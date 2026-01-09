import logging
import os
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import requests
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


def _is_deep_profile_enabled():
    """Check at runtime for deep profiling mode."""
    return os.environ.get("SLIME_DEEP_PROFILE", "0") == "1"

from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)


def _is_awex_enabled(args):
    """Check if awex integration is enabled."""
    return getattr(args, "use_awex", False)


def _is_metaserver_p2p_enabled(args):
    """Check if MetaServer P2P mode is enabled."""
    return getattr(args, "use_metaserver_p2p", False)


def _get_gpu_identity():
    """Get unique GPU identity: hostname_deviceid."""
    import socket
    hostname = socket.gethostname()
    device_id = torch.cuda.current_device()
    # DEBUG: Log for troubleshooting IPC issues
    rank = dist.get_rank() if dist.is_initialized() else 0
    logger.info(
        f"[MetaServer P2P DEBUG] Slime rank={rank}, hostname={hostname}, "
        f"cuda.current_device()={device_id}, identity={hostname}_{device_id}"
    )
    return f"{hostname}_{device_id}"


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

        # Check which mode is enabled
        self._use_awex = _is_awex_enabled(args)
        self._use_metaserver_p2p = _is_metaserver_p2p_enabled(args)
        self._awex_sender = None
        self._ms_client = None

        if self._use_awex:
            self._init_awex(args, model, weights_getter)
        elif self._use_metaserver_p2p:
            self._init_metaserver_p2p(args, model)
        else:
            self._init_baseline(args, model)

        self._model_update_groups = None

    def _init_awex(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable,
    ) -> None:
        """Initialize awex integration for optimized weight sync."""
        from .awex_integration import (
            AwexIntegrationConfig,
            AwexWeightSender,
            SlimeAwexTrainEngine,
        )

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logger.info("[UpdateWeightFromTensor] Initializing awex integration...")

        # Get HF config for awex
        hf_config = self._get_hf_config()

        # Create awex config
        awex_config = AwexIntegrationConfig.from_slime_args(args)

        # Create train engine adapter
        train_engine = SlimeAwexTrainEngine(
            model=model,
            hf_config=hf_config,
            config=awex_config,
            weights_getter=weights_getter,
        )

        # Create weight sender
        self._awex_sender = AwexWeightSender(train_engine, awex_config)

        if rank == 0:
            logger.info("[UpdateWeightFromTensor] Awex integration initialized")

    def _init_baseline(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
    ) -> None:
        """Initialize baseline chunk-based weight sync."""
        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args,
            model=model,
            model_name=self.model_name,
            quantization_config=self.quantization_config,
        )

        # create the group within megatron.
        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            end_rank = start_rank + self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

    def _init_metaserver_p2p(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
    ) -> None:
        """
        Initialize MetaServer P2P mode.

        Key difference from baseline:
        - No Gloo gather (each rank PUTs directly to MetaServer)
        - Use GPU identity (hostname_deviceid) as key instead of rank number
        - SGLang workers GET by their own GPU identity
        """
        # Initialize HF weight iterator (same as baseline)
        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args,
            model=model,
            model_name=self.model_name,
            quantization_config=self.quantization_config,
        )

        # Initialize Gloo groups for gather (needed by _send_hf_params)
        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            end_rank = start_rank + self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

        # Initialize MetaServer client
        meta_server_addr = getattr(args, "awex_meta_server_addr", None)
        if not meta_server_addr:
            raise ValueError("--awex-meta-server-addr is required for --use-metaserver-p2p")

        # Parse address
        if ":" in meta_server_addr:
            host, port = meta_server_addr.rsplit(":", 1)
            port = int(port)
        else:
            raise ValueError(f"Invalid meta_server_addr format: {meta_server_addr}, expected 'ip:port'")

        # Inline lightweight MetaServer client (avoid external dependency)
        import pickle
        import struct

        class LightweightMSClient:
            """Lightweight MetaServer client using only requests + pickle"""
            def __init__(self, address, port):
                self._base_url = f"http://{address}:{port}"
                self._session = requests.Session()

            def put_object(self, key, obj, timeout=120):
                """Store object on server"""
                pickled = pickle.dumps(obj)
                binary = struct.pack("!I", len(pickled)) + pickled
                resp = self._session.put(f"{self._base_url}/v1/put_binary/{key}", data=binary, timeout=timeout)
                resp.raise_for_status()
                return resp.json()

            def delete_if_exists(self, key):
                """Delete data from server if it exists"""
                try:
                    self._session.delete(f"{self._base_url}/v1/delete/{key}", timeout=5)
                except Exception:
                    pass

        self._ms_client = LightweightMSClient(host, port)
        self._ms_addr = meta_server_addr  # Store for passing to SGLang
        self._ms_timeout = getattr(args, "awex_timeout", 600)
        self._gpu_identity = _get_gpu_identity()

        rank = dist.get_rank() if dist.is_initialized() else 0
        if rank == 0:
            logger.info(
                f"[MetaServer P2P] Initialized: addr={meta_server_addr}, "
                f"gpu_identity={self._gpu_identity}, timeout={self._ms_timeout}s"
            )

        # Still need engine mapping for coordination (which engine this rank belongs to)
        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            end_rank = start_rank + self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            if dist.get_rank() in group_ranks:
                self._engine_id = start_rank // self.args.rollout_num_gpus_per_engine

    def _get_hf_config(self):
        """Get HuggingFace config for the model."""
        # Use args.hf_checkpoint which is the correct path
        hf_checkpoint = getattr(self.args, "hf_checkpoint", None)
        if hf_checkpoint is None:
            hf_checkpoint = self.model_name

        if hf_checkpoint is None:
            logger.error("Neither hf_checkpoint nor model_name is set, cannot load HF config")
            return None

        try:
            from transformers import AutoConfig

            return AutoConfig.from_pretrained(hf_checkpoint, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Failed to load HF config from {hf_checkpoint}: {e}")
            return None

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
        Update weights to inference engines.

        Uses awex path if enabled, MetaServer P2P if enabled, otherwise baseline chunk loop.
        """
        if self._use_awex:
            self._update_weights_awex()
        elif self._use_metaserver_p2p:
            self._update_weights_metaserver_p2p()
        else:
            self._update_weights_baseline()

    def _update_weights_awex(self) -> None:
        """
        Awex optimized path: single call replaces 317 chunk loop.

        Expected savings: ~12s (Ray round-trip overhead elimination)
        """
        t_start = time.time()
        rank = dist.get_rank()

        self.weight_version += 1

        if rank == 0:
            logger.info(
                f"[Awex] Starting weight sync version={self.weight_version}..."
            )

        # Flush caches first (same as baseline)
        if rank == 0:
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        # Single awex call replaces the entire chunk loop
        self._awex_sender.update_weights()

        dist.barrier(group=get_gloo_group())

        if rank == 0:
            total_time = time.time() - t_start
            logger.info(
                f"[Awex] Weight sync complete: version={self.weight_version} "
                f"total={total_time:.3f}s"
            )

    def _update_weights_metaserver_p2p(self) -> None:
        """
        P2P MetaServer weight sync: each rank PUTs directly to MetaServer.

        Key design:
        - NO Gloo gather: each rank PUTs to MetaServer with gpu_identity key
        - Only gather_src rank sends lightweight Ray trigger (version, chunk_id)
        - SGLang workers GET by their own gpu_identity (same physical GPU)
        - Double buffer slots + N-2 confirmation for memory safety

        Benefits:
        - Eliminates Gloo gather overhead (6.3s for 317 chunks)
        - Eliminates Ray payload serialization (9.5s for 317 chunks)
        - 64-way parallel PUT/GET
        """
        t_start = time.time()
        rank = dist.get_rank()

        self.weight_version += 1

        if rank == 0:
            logger.info(
                f"[MetaServer P2P] Starting weight sync version={self.weight_version}..."
            )

        # Flush caches first (same as baseline)
        if rank == 0:
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        # Double buffer slots to hold long_lived_tensors (for CUDA IPC validity)
        slots = [None, None]
        futures = {}  # chunk_id -> Ray future (only on gather_src rank)

        chunk_count = 0
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            # 1. N-2 confirmation: ensure Slot[chunk_count % 2] is safe to overwrite
            prev_use_idx = chunk_count - 2
            if prev_use_idx >= 0 and prev_use_idx in futures:
                ray.get(futures[prev_use_idx])  # Block until SGLang consumed
                slots[prev_use_idx % 2] = None  # Release old tensors

            # 2. Serialize chunk (creates CUDA IPC handles, KB-level)
            serialized_tensors, long_lived_tensors = self._serialize_chunk_for_metapipe(
                hf_named_tensors
            )
            slots[chunk_count % 2] = long_lived_tensors  # Hold reference!

            # 3. PUT to MetaServer with gpu_identity key (NO Gloo gather!)
            key = f"weights_{self._gpu_identity}_v{self.weight_version}_c{chunk_count}"
            chunk_data = {
                "serialized_tensors": serialized_tensors,
                "load_format": "flattened_bucket",
                "weight_version": str(self.weight_version),
            }
            self._ms_client.put_object(key, chunk_data, timeout=self._ms_timeout)

            # 4. Only gather_src rank sends lightweight Ray trigger
            if rank == self._ipc_gather_src:
                ref = self._ipc_engine.update_weights_from_metaserver.remote(
                    chunk_id=chunk_count,
                    weight_version=self.weight_version,
                    gpu_identity=self._gpu_identity,  # Pass for logging only
                    meta_server_addr=self._ms_addr,
                    load_format="flattened_bucket",
                    flush_cache=False,
                )
                futures[chunk_count] = ref

            chunk_count += 1

        # Wait for last two chunks to complete
        for i in range(max(0, chunk_count - 2), chunk_count):
            if i in futures:
                ray.get(futures[i])

        dist.barrier(group=get_gloo_group())

        if rank == 0:
            total_time = time.time() - t_start
            logger.info(
                f"[MetaServer P2P] Weight sync complete: version={self.weight_version} "
                f"chunks={chunk_count} total={total_time:.3f}s"
            )

    def _serialize_chunk_for_metapipe(
        self,
        hf_named_tensors: list[tuple[str, torch.Tensor]],
    ) -> tuple[list[str], list[dict]]:
        """
        Serialize chunk using CUDA IPC handles (KB-level, not GB).

        Returns:
            serialized_tensors: List of serialized strings (with IPC handles)
            long_lived_tensors: List of dicts to keep alive (for IPC validity)
        """
        long_lived_tensors = []

        # Group by dtype
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

            flattened_tensor_data = {
                "flattened_tensor": flattened_tensor,
                "metadata": metadata,
            }
            # Keep reference alive for CUDA IPC to work
            long_lived_tensors.append(flattened_tensor_data)

            # Use MultiprocessingSerializer → generates CUDA IPC handles (KB, not GB!)
            serialized = MultiprocessingSerializer.serialize(flattened_tensor_data, output_str=True)
            serialized_tensors.append(serialized)

        return serialized_tensors, long_lived_tensors

    def _update_weights_baseline(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        if _is_baseline_profile_enabled():
            t_cycle_start = time.time()

        self.weight_version += 1

        rank = dist.get_rank()

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
                log_msg = (
                    f"[Serialize Analysis] tensor_mb={tensor_mb:.1f} serialized_mb={serialized_mb:.1f} "
                    f"ratio={serialized_mb/tensor_mb:.4f}x device={flattened_tensor.device}"
                )
                print(log_msg, flush=True)
                # Write to shared storage for reliability
                try:
                    with open("/mnt/hisys-data/yqzhao/slime_profile.log", "a") as f:
                        f.write(log_msg + "\n")
                        f.flush()
                        os.fsync(f.fileno())
                except Exception:
                    pass

    if _is_baseline_profile_enabled():
        serialize_time = time.time() - t_start
        t_gather_start = time.time()

    # Deep profiling: Gloo network monitoring
    if _is_deep_profile_enabled():
        try:
            import psutil
            net0 = psutil.net_io_counters()
        except ImportError:
            net0 = None
        t_gloo_start = time.time()

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

    # Deep profiling: log Gloo network stats
    if _is_deep_profile_enabled():
        gloo_time = time.time() - t_gloo_start
        if net0 is not None:
            try:
                net1 = psutil.net_io_counters()
                bytes_xfer = (net1.bytes_sent - net0.bytes_sent + net1.bytes_recv - net0.bytes_recv)
                if rank == ipc_gather_src:
                    log_msg = f"[Gloo Deep] rank={rank} time={gloo_time*1000:.1f}ms bytes_xfer={bytes_xfer/1e6:.1f}MB"
                    print(log_msg, flush=True)
                    try:
                        with open("/mnt/hisys-data/yqzhao/deep_profile.log", "a") as f:
                            f.write(log_msg + "\n")
                            f.flush()
                            os.fsync(f.fileno())
                    except Exception:
                        pass
            except Exception:
                pass

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
            # Deep profiling: add submit timestamp for Ray latency measurement
            if _is_deep_profile_enabled():
                kwargs["_submit_ts"] = time.time()
            refs.append(ipc_engine.update_weights_from_tensor.remote(**kwargs))

    if _is_baseline_profile_enabled():
        ray_time = time.time() - t_ray_start
        total_time = time.time() - t_start
        world_size = dist.get_world_size(ipc_gather_group)
        total_gathered_mb = (total_bytes * world_size) / (1024 * 1024)
        gather_throughput = total_gathered_mb / gather_time if gather_time > 0 else 0
        if rank == ipc_gather_src:
            log_msg = (
                f"[Baseline Profile] rank={rank} n_tensors={len(hf_named_tensors)} "
                f"data_mb={total_gathered_mb:.1f} "
                f"serialize={serialize_time:.3f}s gather={gather_time:.3f}s "
                f"({gather_throughput:.1f}MB/s) ray={ray_time:.3f}s total={total_time:.3f}s"
            )
            print(log_msg, flush=True)
            try:
                with open("/mnt/hisys-data/yqzhao/slime_profile.log", "a") as f:
                    f.write(log_msg + "\n")
                    f.flush()
                    os.fsync(f.fileno())
            except Exception:
                pass

    return refs, long_live_tensors
