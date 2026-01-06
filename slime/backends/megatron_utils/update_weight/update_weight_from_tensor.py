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

# Module-level marker to verify this file is being loaded
print(f"[WEIGHT-SYNC-MODULE] update_weight_from_tensor.py loaded", flush=True)


def _debug_log(msg: str, rank: int = -1):
    """Write debug log to file for rank 0."""
    if rank != 0:
        return
    import socket
    hostname = socket.gethostname()
    with open(f"/tmp/rank0_debug_{hostname}.log", "a") as f:
        f.write(f"{msg}\n")
        f.flush()
        import os
        os.fsync(f.fileno())  # 强制刷新到磁盘


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
    ) -> None:
        """
        Compute param buckets, create IPC Gloo groups (rollout_num_gpus_per_engine ranks/group).
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )

        # create the group within megatron.
        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            end_rank = start_rank + self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

        self._model_update_groups = None

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines
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

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.

        SIMPLIFIED DESIGN: Only rank 0 sends to ALL engines.
        - Avoids the mystery of why rank 64 doesn't send to engine 1
        - rank 0 already has handles to all engines (see flush_cache)
        - All other ranks only do NCCL communication and serialization
        """
        rank = dist.get_rank()
        _debug_log(f"ENTERING update_weights, version={self.weight_version + 1}, num_engines={len(self.rollout_engines)}", rank)
        logger.info(f"[WEIGHT-SYNC] *** ENTERING update_weights *** rank={rank} version={self.weight_version + 1}")

        if _is_baseline_profile_enabled():
            t_cycle_start = time.time()

        self.weight_version += 1

        if _is_baseline_profile_enabled():
            t_flush_start = time.time()

        _debug_log(f"Starting flush_cache for {len(self.rollout_engines)} engines", rank)
        if rank == 0:
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        _debug_log("flush_cache DONE, entering barrier", rank)
        dist.barrier(group=get_gloo_group())

        if _is_baseline_profile_enabled():
            flush_time = time.time() - t_flush_start
            t_weights_getter_start = time.time()

        _debug_log("Getting weights...", rank)
        megatron_local_weights = self.weights_getter()
        _debug_log("Weights obtained", rank)

        if _is_baseline_profile_enabled():
            weights_getter_time = time.time() - t_weights_getter_start
            t_chunks_start = time.time()

        chunk_count = 0
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            _debug_log(f"Chunk {chunk_count}: got {len(hf_named_tensors)} tensors", rank)

            # All ranks serialize (for CUDA sync)
            t_serialize = time.time()
            serialized = self._serialize_chunk(hf_named_tensors)
            serialize_time = time.time() - t_serialize
            _debug_log(f"Chunk {chunk_count}: serialized in {serialize_time:.3f}s, len={len(serialized)}", rank)

            # Only rank 0 sends to ALL engines
            refs = []
            if rank == 0:
                _debug_log(f"Chunk {chunk_count}: about to send to {len(self.rollout_engines)} engines", rank)
                for i, engine in enumerate(self.rollout_engines):
                    try:
                        _debug_log(f"Chunk {chunk_count}: sending to engine {i} (type={type(engine).__name__})...", rank)
                        t_remote = time.time()
                        ref = engine.update_weights_from_tensor.remote(
                            serialized_named_tensors=[serialized],
                            load_format="flattened_bucket",
                            weight_version=str(self.weight_version),
                        )
                        refs.append(ref)
                        remote_time = time.time() - t_remote
                        _debug_log(f"Chunk {chunk_count}: engine {i} sent in {remote_time:.3f}s, ref={ref}", rank)
                    except Exception as e:
                        _debug_log(f"Chunk {chunk_count}: FAILED engine {i}: {type(e).__name__}: {e}", rank)
                        import traceback
                        _debug_log(f"Traceback: {traceback.format_exc()}", rank)
                        raise
                _debug_log(f"Chunk {chunk_count}: all {len(refs)} refs collected, calling ray.get", rank)

            chunk_count += 1

            # rank 0 waits for Ray, then ALL ranks sync before next chunk
            if rank == 0:
                _debug_log(f"Chunk {chunk_count-1}: ray.get on {len(refs)} refs...", rank)
                t_rayget = time.time()
                ray.get(refs)
                rayget_time = time.time() - t_rayget
                _debug_log(f"Chunk {chunk_count-1}: ray.get done in {rayget_time:.3f}s", rank)

            _debug_log(f"Chunk {chunk_count-1}: entering barrier", rank)
            dist.barrier(group=get_gloo_group())
            _debug_log(f"Chunk {chunk_count-1}: barrier done", rank)

        if _is_baseline_profile_enabled():
            chunks_time = time.time() - t_chunks_start
            total_time = time.time() - t_cycle_start
            print(
                f"[Baseline Profile] Cycle complete: rank={rank} version={self.weight_version} "
                f"chunks={chunk_count} flush={flush_time:.3f}s weights_getter={weights_getter_time:.3f}s "
                f"chunks_loop={chunks_time:.3f}s total={total_time:.3f}s",
                flush=True
            )

        _debug_log(f"All {chunk_count} chunks done, final barrier", rank)
        dist.barrier(group=get_gloo_group())
        _debug_log("update_weights COMPLETE", rank)

    def _serialize_chunk(self, hf_named_tensors: list[tuple[str, torch.Tensor]]) -> str:
        """Serialize a chunk of HF tensors to string."""
        if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
            converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
        else:
            converted_named_tensors_by_dtypes = {}
            for name, tensor in hf_named_tensors:
                dtype = tensor.dtype
                if dtype not in converted_named_tensors_by_dtypes:
                    converted_named_tensors_by_dtypes[dtype] = []
                converted_named_tensors_by_dtypes[dtype].append((name, tensor))

        # Usually only one dtype (bf16), so one serialized string
        for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
            flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
            metadata = flattened_tensor_bucket.get_metadata()
            flattened_tensor = flattened_tensor_bucket.get_flattened_tensor()
            flattened_tensor_data = {
                "flattened_tensor": flattened_tensor,
                "metadata": metadata,
            }
            return MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)

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
    Send HF weights to colocated SGLang engine.

    Optimization: After PP/EP/TP communication, all ranks have identical complete HF weights.
    So we only need ONE rank to send, instead of gathering 64 identical copies.

    Original flow: 64 ranks serialize → Gloo gather → send 64 copies
    Optimized flow: All ranks serialize (for CUDA sync) → only gather_src sends 1 copy
    """
    rank = dist.get_rank()
    long_live_tensors = []
    refs = []

    # --- PROFILING: Start ---
    if _is_baseline_profile_enabled():
        t_start = time.time()
        total_bytes = 0

    # All ranks do serialization (needed for CUDA synchronization)
    # Group tensors by dtype if needed
    if getattr(FlattenedTensorBucket, "supports_multi_dtypes", False):
        converted_named_tensors_by_dtypes = {"dtype": hf_named_tensors}
    else:
        converted_named_tensors_by_dtypes = {}
        for name, tensor in hf_named_tensors:
            dtype = tensor.dtype
            if dtype not in converted_named_tensors_by_dtypes:
                converted_named_tensors_by_dtypes[dtype] = []
            converted_named_tensors_by_dtypes[dtype].append((name, tensor))

    # Serialize tensors (all ranks do this for CUDA sync)
    serialized_tensors = []
    for _dtype, named_tensors in converted_named_tensors_by_dtypes.items():
        flattened_tensor_bucket = FlattenedTensorBucket(named_tensors=named_tensors)
        metadata = flattened_tensor_bucket.get_metadata()
        flattened_tensor = flattened_tensor_bucket.get_flattened_tensor()
        flattened_tensor_data = {
            "flattened_tensor": flattened_tensor,
            "metadata": metadata,
        }
        long_live_tensors.append(flattened_tensor_data)
        serialized_tensors.append(MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True))

        if _is_baseline_profile_enabled():
            total_bytes += flattened_tensor.numel() * flattened_tensor.element_size()

    if _is_baseline_profile_enabled():
        serialize_time = time.time() - t_start
        t_send_start = time.time()

    # Only gather_src rank sends (skip the redundant Gloo gather)
    if rank == ipc_gather_src:
        for i in range(len(serialized_tensors)):
            kwargs = {
                "serialized_named_tensors": [serialized_tensors[i]],  # Single copy, wrapped in list
                "load_format": "flattened_bucket",
                "weight_version": str(weight_version),
            }
            refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    if _is_baseline_profile_enabled():
        send_time = time.time() - t_send_start
        total_time = time.time() - t_start
        total_mb = total_bytes / (1024 * 1024)
        # Only log from gather_src ranks to reduce noise
        if rank == ipc_gather_src:
            print(
                f"[Baseline Profile] rank={rank} n_tensors={len(hf_named_tensors)} "
                f"data_mb={total_mb:.1f} "
                f"serialize={serialize_time:.3f}s send={send_time:.3f}s total={total_time:.3f}s",
                flush=True
            )

    # No barrier here - async mode handles sync at the end of update_weights()
    return refs, long_live_tensors
