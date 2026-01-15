import logging
import os
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle

from slime.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)


def _is_baseline_profile_enabled():
    """Check at runtime, not import time, because Ray sets env vars after import."""
    return os.environ.get("SLIME_BASELINE_PROFILE", "0") == "1"

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)


def _get_awex_classes():
    """Lazy import awex_integration to avoid potential init issues at module load time."""
    from .awex_integration import AwexIntegrationConfig, AwexWeightSender, SlimeAwexTrainEngine, is_awex_enabled
    return AwexIntegrationConfig, AwexWeightSender, SlimeAwexTrainEngine, is_awex_enabled


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    load(dict→GPU) → broadcast PP/EP(GPU NCCL) → gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: GPU→CPU serialize → gather_object(Gloo CPU, collects from rollout_num_gpus_per_engine ranks) → Ray IPC to engine.
    Distributed: GPU NCCL broadcast to remote engines.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
        hf_config: Any = None,
    ) -> None:
        """
        Compute param buckets, create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).

        Args:
            args: Slime command line arguments
            model: Megatron model (list for VPP)
            weights_getter: Callable to get current weights dict
            model_name: Model architecture name
            quantization_config: Optional quantization config
            hf_config: HuggingFace model config (required for AWEX mode)
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0

        # AWEX integration: initialize if enabled
        # Use lazy import to avoid potential init issues at module load time
        AwexIntegrationConfig, AwexWeightSender, SlimeAwexTrainEngine, is_awex_enabled = _get_awex_classes()
        self._use_awex = is_awex_enabled(args)
        self._awex_sender = None
        if self._use_awex:
            if hf_config is None:
                raise ValueError("hf_config is required when --use-awex is enabled")
            logger.info("[AWEX] Initializing AWEX weight sender...")
            awex_config = AwexIntegrationConfig.from_slime_args(args)
            train_engine = SlimeAwexTrainEngine(
                model=model,
                hf_config=hf_config,
                config=awex_config,
                weights_getter=weights_getter,
                offload_train=getattr(args, 'offload_train', False),
            )
            self._awex_sender = AwexWeightSender(train_engine, awex_config)
            logger.info("[AWEX] AWEX weight sender initialized")
        else:
            # Only create HF weight iterator for non-AWEX mode
            self._hf_weight_iterator = HfWeightIteratorBase.create(
                args=args, model=model, model_name=model_name, quantization_config=quantization_config
            )

        # create the group within megatron.
        # In AWEX mode with cross-node colocate, each node's 8 ranks form a group
        # and need to trigger their corresponding engine for weight updates.
        # In non-AWEX mode, use rollout_num_gpus_per_engine as before.
        if self._use_awex:
            # AWEX cross-node: group by node (num_gpus_per_node ranks per group)
            awex_group_size = getattr(self.args, 'actor_num_gpus_per_node', 8)
            logger.info(f"[AWEX] Creating IPC groups with size {awex_group_size} (per-node grouping)")
        else:
            awex_group_size = self.args.rollout_num_gpus_per_engine

        for start_rank in range(0, dist.get_world_size(), awex_group_size):
            end_rank = start_rank + awex_group_size
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

        self._model_update_groups = None
        # Store AWEX group size for engine mapping in connect_rollout_engines
        self._awex_group_size = awex_group_size if self._use_awex else None

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.

        In AWEX cross-node mode, ALL engines are used for colocate (one per node),
        because execute_task_in_model_worker only works within a single scheduler.
        Each node's leader rank triggers its corresponding engine.
        """
        self.rollout_engines = rollout_engines

        # In AWEX mode with cross-node colocate, use all engines
        # because we need each node to independently receive weights
        if self._use_awex and self._awex_group_size:
            # All engines are used for AWEX cross-node weight sync
            self.use_distribute = False
            num_engines = len(rollout_engines)
            world_size = dist.get_world_size()
            # Each engine corresponds to (world_size / num_engines) ranks
            # e.g., 128 ranks / 2 engines = 64 ranks per engine
            ranks_per_engine = world_size // num_engines
            logger.info(
                f"[AWEX] Using {num_engines} engines for cross-node weight sync "
                f"(world_size={world_size}, ranks_per_engine={ranks_per_engine})"
            )

            # Map each training rank to its corresponding engine
            # Engine 0: ranks 0 to (ranks_per_engine-1)
            # Engine 1: ranks (ranks_per_engine) to (2*ranks_per_engine-1)
            # etc.
            my_rank = dist.get_rank()
            engine_idx = my_rank // ranks_per_engine
            if engine_idx < num_engines:
                self._ipc_engine = rollout_engines[engine_idx]
                start_rank = engine_idx * ranks_per_engine
                end_rank = (engine_idx + 1) * ranks_per_engine
                logger.info(
                    f"[AWEX] Rank {my_rank} mapped to engine {engine_idx} "
                    f"(ranks {start_rank}-{end_rank-1})"
                )
                # Store the first rank in this engine's range for Ray trigger
                # Only this rank should trigger the Ray call to avoid duplicate calls
                self._awex_engine_first_rank = start_rank
            else:
                logger.error(
                    f"[AWEX] Rank {my_rank} has no engine! "
                    f"engine_idx={engine_idx} >= num_engines={num_engines}"
                )
            return

        # Non-AWEX mode: original logic
        colocate_engine_nums = (
            self.args.actor_num_nodes * self.args.actor_num_gpus_per_node // self.args.rollout_num_gpus_per_engine
        )
        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "slime"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )

                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args, self._group_name, self.distributed_rollout_engines
                )

        # Here we assume the gpu id of rollout engines and train actors are the same.
        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            if dist.get_rank() in group_ranks:
                self._ipc_engine = engine

    def connect_awex_engine(
        self, engine: ActorHandle, engine_first_rank: int, all_engines: Sequence[ActorHandle] = None
    ) -> None:
        """
        Connect a single engine for AWEX mode (called by engine leaders only).

        In the simplified AWEX flow, only engine leaders need the actual engine handle
        to trigger the Ray call. Other ranks compute engine_first_rank locally.

        Args:
            engine: The SGLang engine ActorHandle for this leader's node
            engine_first_rank: The first rank in this engine's group (e.g., 0, 8, 16, ...)
            all_engines: Optional list of all engines (for compatibility/testing)
        """
        self._ipc_engine = engine
        self._awex_engine_first_rank = engine_first_rank
        self.rollout_engines = all_engines if all_engines else [engine]
        self.use_distribute = False
        logger.info(
            f"[AWEX] Leader rank {dist.get_rank()} connected to engine "
            f"(engine_first_rank={engine_first_rank})"
        )

    def set_awex_leader(self, engine_first_rank: int) -> None:
        """
        Set the engine leader rank for AWEX mode (called by non-leaders).

        Non-leader ranks don't need the actual engine handle - they only need to know
        which rank is their leader for the conditional check in update_weights().

        Args:
            engine_first_rank: The first rank in this rank's engine group
        """
        self._awex_engine_first_rank = engine_first_rank
        self.use_distribute = False
        logger.debug(
            f"[AWEX] Rank {dist.get_rank()} set leader to {engine_first_rank}"
        )

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.

        If AWEX is enabled, delegates to AwexWeightSender which eliminates
        the chunk loop and uses MetaServer + IPC for weight sync.
        """
        self.weight_version += 1
        rank = dist.get_rank()

        # Helper to log GPU memory
        def _log_mem(label):
            if not torch.cuda.is_available():
                return
            device = torch.cuda.current_device()
            allocated = torch.cuda.memory_allocated(device) / (1024**3)
            reserved = torch.cuda.memory_reserved(device) / (1024**3)
            try:
                max_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
                free = max_memory - reserved
            except:
                free = 0
            logger.info(f"[TRAIN_GPU_MEM] {label} | rank={rank} device={device} | allocated={allocated:.2f}GB reserved={reserved:.2f}GB free={free:.2f}GB")

        # AWEX mode: use optimized weight sync
        if self._use_awex:
            _log_mem(f"AWEX update_weights START version={self.weight_version}")
            if rank == 0:
                logger.info(f"[AWEX] Starting weight update version {self.weight_version}")

            # Step 1: Trigger SGLang to start receiving via Ray (non-blocking)
            # Only the first rank in each engine's range triggers the Ray call
            # to avoid duplicate concurrent calls to the same engine
            # e.g., rank 0 triggers engine 0, rank 64 triggers engine 1
            awex_refs = []
            should_trigger = hasattr(self, '_awex_engine_first_rank') and rank == self._awex_engine_first_rank
            if should_trigger:
                from sglang.srt.managers.io_struct import UpdateWeightsFromAwexReqInput
                awex_req = UpdateWeightsFromAwexReqInput(
                    step_id=self.weight_version,
                    weight_version=str(self.weight_version),
                    flush_cache=True,
                )
                awex_refs.append(self._ipc_engine.update_weights_from_awex.remote(awex_req))
                logger.info(f"[AWEX] Rank {rank} triggered SGLang engine for step {self.weight_version} (engine leader)")
            else:
                engine_leader = getattr(self, '_awex_engine_first_rank', 'unknown')
                logger.debug(f"[AWEX] Rank {rank} skipping trigger (engine leader is {engine_leader})")

            # Step 2: Training side writes weights to MetaServer
            # This will block until SGLang finishes receiving
            _log_mem(f"AWEX BEFORE _awex_sender.update_weights() version={self.weight_version}")
            self._awex_sender.update_weights()
            _log_mem(f"AWEX AFTER _awex_sender.update_weights() version={self.weight_version}")

            # Step 3: Wait for Ray call to complete (should be done by now)
            if awex_refs:
                _log_mem(f"AWEX BEFORE ray.get(awex_refs) version={self.weight_version}")
                results = ray.get(awex_refs)
                _log_mem(f"AWEX AFTER ray.get(awex_refs) version={self.weight_version}")
                for result in results:
                    if not result.success:
                        logger.error(f"[AWEX] SGLang weight update failed: {result.message}")

            _log_mem(f"AWEX update_weights END version={self.weight_version}")
            if rank == 0:
                logger.info(f"[AWEX] Completed weight update version {self.weight_version}")
            return

        # Baseline mode: chunk loop
        if _is_baseline_profile_enabled():
            t_cycle_start = time.time()

        if _is_baseline_profile_enabled():
            t_flush_start = time.time()

        if rank == 0:
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        if _is_baseline_profile_enabled():
            flush_time = time.time() - t_flush_start
            t_weights_getter_start = time.time()

        megatron_local_weights = self.weights_getter()

        if _is_baseline_profile_enabled():
            weights_getter_time = time.time() - t_weights_getter_start
            t_chunks_start = time.time()
            total_rayget_time = 0.0
            total_send_time = 0.0

        chunk_count = 0
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            if _is_baseline_profile_enabled():
                t_send_start = time.time()

            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)

            if _is_baseline_profile_enabled():
                send_time = time.time() - t_send_start
                total_send_time += send_time
                t_rayget_start = time.time()

            ray.get(refs)

            if _is_baseline_profile_enabled():
                rayget_time = time.time() - t_rayget_start
                total_rayget_time += rayget_time

            del long_lived_tensors
            chunk_count += 1

        if _is_baseline_profile_enabled() and rank == 0:
            chunks_time = time.time() - t_chunks_start
            total_time = time.time() - t_cycle_start
            print(
                f"[Baseline Profile] Cycle complete: version={self.weight_version} "
                f"chunks={chunk_count} flush={flush_time:.3f}s weights_getter={weights_getter_time:.3f}s "
                f"total_send={total_send_time:.3f}s total_rayget={total_rayget_time:.3f}s "
                f"chunks_loop={chunks_time:.3f}s total={total_time:.3f}s",
                flush=True
            )

        dist.barrier(group=get_gloo_group())

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []

        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
        )
        all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs, long_lived_tensors


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
) -> tuple[list[ObjectRef], Any]:
    """
    Original baseline: gather CUDA IPC handles from all ranks, then send to SGLang.

    Key insight from blog: ForkingPickler serializes CUDA tensors as IPC handles (~KB),
    not full data (~GB). CUDA IPC allows zero-copy access across processes on same GPU.
    """
    long_live_tensors = []
    rank = dist.get_rank()

    # --- PROFILING: Start ---
    if _is_baseline_profile_enabled():
        t_start = time.time()
        total_bytes = 0

    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    serialized_tensors = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        metadata = flattened_tensor_bucket.get_metadata()
        flattened_tensor = flattened_tensor_bucket.get_flattened_tensor()

        # Compute tensor size for profiling
        tensor_bytes = flattened_tensor.numel() * flattened_tensor.element_size()

        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor,
            "metadata": metadata,
        }
        long_live_tensors.append(flattened_tensor_data)
        serialized = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
        serialized_tensors.append(serialized)

        if _is_baseline_profile_enabled():
            total_bytes += tensor_bytes
            # KEY ANALYSIS: Compare tensor size vs serialized string size
            # If ratio << 1: using CUDA IPC handles (small)
            # If ratio ~1.3: full serialization (base64 overhead)
            if rank == 0:
                tensor_mb = tensor_bytes / (1024 * 1024)
                serialized_mb = len(serialized) / (1024 * 1024)
                print(
                    f"[Serialize Analysis] tensor_mb={tensor_mb:.1f} serialized_mb={serialized_mb:.1f} "
                    f"ratio={serialized_mb/tensor_mb:.4f}x device={flattened_tensor.device}",
                    flush=True
                )

    if _is_baseline_profile_enabled():
        serialize_time = time.time() - t_start
        t_gather_start = time.time()

    # Gloo gather: collect serialized data from all ranks to gather_src
    serialized_named_tensors = (
        [None] * dist.get_world_size(ipc_gather_group) if ipc_gather_src == rank else None
    )
    dist.gather_object(
        serialized_tensors,
        object_gather_list=serialized_named_tensors,
        dst=ipc_gather_src,
        group=ipc_gather_group,
    )

    if _is_baseline_profile_enabled():
        gather_time = time.time() - t_gather_start
        t_ray_start = time.time()

    refs = []
    if rank == ipc_gather_src:
        # Send 64 copies to SGLang - each TP worker gets data from corresponding rank
        # This preserves CUDA IPC handle validity (same GPU)
        num_dtypes = len(serialized_named_tensors[0])
        for i in range(num_dtypes):
            kwargs = {
                "serialized_named_tensors": [tensors[i] for tensors in serialized_named_tensors],
                "load_format": "flattened_bucket",
                "weight_version": str(weight_version),
            }
            refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    if _is_baseline_profile_enabled():
        ray_time = time.time() - t_ray_start
        total_time = time.time() - t_start
        world_size = dist.get_world_size(ipc_gather_group)
        total_gathered_mb = (total_bytes * world_size) / (1024 * 1024)
        gather_throughput = total_gathered_mb / gather_time if gather_time > 0 else 0
        if rank == ipc_gather_src:
            print(
                f"[Baseline Profile] rank={rank} n_tensors={len(hf_named_tensors)} "
                f"data_mb={total_gathered_mb:.1f} "
                f"serialize={serialize_time:.3f}s gather={gather_time:.3f}s "
                f"({gather_throughput:.1f}MB/s) ray={ray_time:.3f}s total={total_time:.3f}s",
                flush=True
            )

    return refs, long_live_tensors
