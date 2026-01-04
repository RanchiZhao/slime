"""
Delta weight update implementation for efficient weight synchronization.

This module implements sparse delta updates that only synchronize the changed
elements between training and inference, which is more efficient than full
weight replacement when only a small fraction of elements have changed.

First sync uses the baseline FlattenedTensorBucket method for full weight transfer.
Subsequent syncs compute sparse deltas and only send changed elements.

Key principle: NEVER modify tensors before IPC send completes.
- Sparse path: extract (indices, values), send custom format
- Dense path: send original tensor via FlattenedTensorBucket
- Storage: update _last_synced_weights AFTER ray.get() completes
"""

import logging
import pickle
import base64
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

from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_tensor import _send_to_colocated_engine
from ..sglang import FlattenedTensorBucket, MultiprocessingSerializer

logger = logging.getLogger(__name__)

# Threshold for falling back to dense transfer (changed_ratio > threshold)
DENSE_THRESHOLD = 0.3  # 30%

# Protocol version for Slime↔SGLang delta sync compatibility
# Increment this when making breaking changes to the delta sync protocol
DELTA_SYNC_PROTOCOL_VERSION = "1.0"


def is_moe_expert_param(param_name: str) -> bool:
    """Check if parameter is an MoE expert parameter.

    MoE expert parameters have special fused layout in SGLang and are handled
    separately using dense transfer + load_weights().
    """
    return "experts" in param_name and any(
        proj in param_name for proj in ("gate_proj", "up_proj", "down_proj")
    )


def parse_moe_expert_info(param_name: str) -> tuple[int, str] | None:
    """Parse MoE expert info from HF parameter name.

    Args:
        param_name: HF parameter name like 'model.layers.0.mlp.experts.5.gate_proj.weight'

    Returns:
        Tuple of (global_expert_id, proj_type) or None if not an MoE expert param.
        proj_type is one of: 'gate_proj', 'up_proj', 'down_proj'
    """
    import re

    # Pattern: model.layers.{layer}.mlp.experts.{expert_id}.{proj_type}.weight
    pattern = r"model\.layers\.\d+\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight"
    match = re.search(pattern, param_name)
    if match:
        expert_id = int(match.group(1))
        proj_type = match.group(2)
        return (expert_id, proj_type)
    return None


def get_moe_partition_dim(proj_type: str) -> int:
    """Get partition dimension for MoE expert projection.

    In SGLang's fused MoE layout:
    - gate_proj (w1) and up_proj (w3) are ColumnParallel (partition_dim=0)
    - down_proj (w2) is RowParallel (partition_dim=1)

    Args:
        proj_type: One of 'gate_proj', 'up_proj', 'down_proj'

    Returns:
        0 for ColumnParallel, 1 for RowParallel
    """
    if proj_type in ("gate_proj", "up_proj"):
        return 0  # ColumnParallel
    else:  # down_proj
        return 1  # RowParallel


def get_partition_dim(param_name: str) -> int | None:
    """Determine the partition dimension for a parameter based on its name.

    Returns:
        0: ColumnParallel (partition along output/row dimension)
        1: RowParallel (partition along input/col dimension)
        None: Replicated (no partition)
    """
    # MoE expert params handled separately
    if is_moe_expert_param(param_name):
        return None  # Will use dense path with load_weights()

    # Extract the last component of the parameter name
    parts = param_name.replace(".weight", "").replace(".bias", "").split(".")
    layer_name = parts[-1] if parts else ""

    # ColumnParallel layers (partition_dim=0)
    column_parallel = {
        "q_proj", "k_proj", "v_proj",  # attention projections
        "qkv_proj",  # fused QKV
        "gate_proj", "up_proj",  # MLP gate and up (non-expert)
        "gate_up_proj",  # fused gate+up
        "w1", "w3",  # alternative MLP names (Llama style)
        "embed_tokens",  # embedding (vocab parallel)
        "lm_head",  # output projection
    }

    # RowParallel layers (partition_dim=1)
    row_parallel = {
        "o_proj",  # attention output
        "down_proj",  # MLP down (non-expert)
        "w2",  # alternative MLP name
    }

    if layer_name in column_parallel:
        return 0
    elif layer_name in row_parallel:
        return 1
    else:
        # Replicated: layer_norm, rms_norm, etc.
        return None


def split_indices_by_tp_column(
    global_indices: torch.Tensor,
    values: torch.Tensor,
    out_dim: int,
    in_dim: int,
    tp_size: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Split global indices by TP rank for ColumnParallel (dim=0).

    ColumnParallel partitions along output dimension (rows).
    Each TP rank owns rows [rank * shard_out : (rank+1) * shard_out].

    Args:
        global_indices: Flattened global indices (int32 or int64)
        values: Corresponding values (bf16)
        out_dim: Output dimension (total rows)
        in_dim: Input dimension (columns, not partitioned)
        tp_size: Number of TP ranks

    Returns:
        Dict mapping tp_rank -> (local_indices, local_values)
    """
    shard_out = out_dim // tp_size

    # Pre-allocate lists for each rank
    results: dict[int, tuple[list, list]] = {r: ([], []) for r in range(tp_size)}

    # Vectorized computation
    global_indices_cpu = global_indices.cpu()
    values_cpu = values.cpu()

    rows = global_indices_cpu // in_dim
    cols = global_indices_cpu % in_dim
    tp_ranks = rows // shard_out
    local_rows = rows - tp_ranks * shard_out
    local_indices = local_rows * in_dim + cols

    # Group by TP rank
    for i in range(len(global_indices_cpu)):
        tp_rank = tp_ranks[i].item()
        results[tp_rank][0].append(local_indices[i].item())
        results[tp_rank][1].append(values_cpu[i].item())

    # Convert to tensors
    tensor_results = {}
    for tp_rank, (idx_list, val_list) in results.items():
        if idx_list:
            tensor_results[tp_rank] = (
                torch.tensor(idx_list, dtype=torch.int32),
                torch.tensor(val_list, dtype=values.dtype),
            )
        else:
            tensor_results[tp_rank] = (
                torch.tensor([], dtype=torch.int32),
                torch.tensor([], dtype=values.dtype),
            )

    return tensor_results


def split_indices_by_tp_row(
    global_indices: torch.Tensor,
    values: torch.Tensor,
    out_dim: int,
    in_dim: int,
    tp_size: int,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Split global indices by TP rank for RowParallel (dim=1).

    RowParallel partitions along input dimension (columns).
    Each TP rank owns cols [rank * shard_in : (rank+1) * shard_in].

    Args:
        global_indices: Flattened global indices (int32 or int64)
        values: Corresponding values (bf16)
        out_dim: Output dimension (rows, not partitioned)
        in_dim: Input dimension (total columns)
        tp_size: Number of TP ranks

    Returns:
        Dict mapping tp_rank -> (local_indices, local_values)
    """
    shard_in = in_dim // tp_size

    # Pre-allocate lists for each rank
    results: dict[int, tuple[list, list]] = {r: ([], []) for r in range(tp_size)}

    # Vectorized computation
    global_indices_cpu = global_indices.cpu()
    values_cpu = values.cpu()

    rows = global_indices_cpu // in_dim
    cols = global_indices_cpu % in_dim
    tp_ranks = cols // shard_in
    local_cols = cols - tp_ranks * shard_in
    local_indices = rows * shard_in + local_cols  # Note: local row stride is shard_in

    # Group by TP rank
    for i in range(len(global_indices_cpu)):
        tp_rank = tp_ranks[i].item()
        results[tp_rank][0].append(local_indices[i].item())
        results[tp_rank][1].append(values_cpu[i].item())

    # Convert to tensors
    tensor_results = {}
    for tp_rank, (idx_list, val_list) in results.items():
        if idx_list:
            tensor_results[tp_rank] = (
                torch.tensor(idx_list, dtype=torch.int32),
                torch.tensor(val_list, dtype=values.dtype),
            )
        else:
            tensor_results[tp_rank] = (
                torch.tensor([], dtype=torch.int32),
                torch.tensor([], dtype=values.dtype),
            )

    return tensor_results


# ============================================================================
# Fused Parameter Handling (Non-MoE)
# ============================================================================

# Parameters that should be fused for SGLang compatibility
FUSABLE_QKV_PARAMS = {"q_proj", "k_proj", "v_proj"}
FUSABLE_GATE_UP_PARAMS = {"gate_proj", "up_proj"}


def is_fusable_qkv_param(param_name: str) -> bool:
    """Check if parameter is part of a fusable QKV group (non-MoE only)."""
    if is_moe_expert_param(param_name):
        return False
    parts = param_name.replace(".weight", "").replace(".bias", "").split(".")
    layer_name = parts[-1] if parts else ""
    return layer_name in FUSABLE_QKV_PARAMS


def is_fusable_gate_up_param(param_name: str) -> bool:
    """Check if parameter is part of a fusable gate_up group (non-MoE only)."""
    if is_moe_expert_param(param_name):
        return False
    parts = param_name.replace(".weight", "").replace(".bias", "").split(".")
    layer_name = parts[-1] if parts else ""
    return layer_name in FUSABLE_GATE_UP_PARAMS


def extract_layer_id(param_name: str) -> int | None:
    """Extract layer ID from parameter name.

    Example: 'model.layers.5.self_attn.q_proj.weight' -> 5
    """
    import re
    match = re.search(r"layers\.(\d+)\.", param_name)
    if match:
        return int(match.group(1))
    return None


def fuse_qkv_sparse_deltas(
    q_data: tuple[torch.Tensor, torch.Tensor, tuple] | None,
    k_data: tuple[torch.Tensor, torch.Tensor, tuple] | None,
    v_data: tuple[torch.Tensor, torch.Tensor, tuple] | None,
    tp_size: int,
) -> tuple[str, dict[int, tuple[torch.Tensor, torch.Tensor]], tuple] | None:
    """Fuse q/k/v sparse deltas into qkv_proj format.

    IMPORTANT: All three components (q, k, v) must be present for correct fusion.
    If any component is missing, returns None and the caller should handle fallback.

    Args:
        q_data: (indices, values, shape) for q_proj or None
        k_data: (indices, values, shape) for k_proj or None
        v_data: (indices, values, shape) for v_proj or None
        tp_size: Number of TP ranks

    Returns:
        ("qkv_proj.weight", per_tp_data, fused_shape) or None if incomplete
    """
    # CRITICAL: All three must be present for correct fused layout
    # SGLang's qkv_proj expects [q_dim + k_dim + v_dim, hidden]
    # If any component is missing, we cannot compute correct offsets
    if q_data is None or k_data is None or v_data is None:
        if q_data is not None or k_data is not None or v_data is not None:
            logger.debug(
                f"Incomplete qkv set for fusion: q={q_data is not None}, "
                f"k={k_data is not None}, v={v_data is not None}. Skipping fusion."
            )
        return None

    # All should have same hidden_size (in_dim)
    # Shape is [out_dim, in_dim] where out_dim varies (q_dim, k_dim, v_dim)
    in_dim = q_data[2][1]  # hidden_size

    # For Qwen3, q/k/v have same out_dim (num_heads * head_dim)
    # For models with MQA/GQA, k/v might have smaller out_dim
    q_out = q_data[2][0]
    k_out = k_data[2][0]
    v_out = v_data[2][0]

    # Fused shape: [(q_out + k_out + v_out), in_dim]
    fused_out = q_out + k_out + v_out
    fused_shape = (fused_out, in_dim)

    # Compute offsets in fused tensor (flattened)
    q_offset = 0
    k_offset = q_out * in_dim
    v_offset = (q_out + k_out) * in_dim

    # All three (q, k, v) are guaranteed present at this point
    q_indices, q_values, _ = q_data
    k_indices, k_values, _ = k_data
    v_indices, v_values, _ = v_data

    # Combine indices with offsets
    fused_indices = torch.cat([
        q_indices + q_offset,
        k_indices + k_offset,
        v_indices + v_offset,
    ])
    fused_values = torch.cat([q_values, k_values, v_values])

    # Now split by TP (qkv_proj is ColumnParallel)
    per_tp_data = split_indices_by_tp_column(
        fused_indices, fused_values, fused_out, in_dim, tp_size
    )

    return ("qkv_proj.weight", per_tp_data, fused_shape)


def fuse_gate_up_sparse_deltas(
    gate_data: tuple[torch.Tensor, torch.Tensor, tuple] | None,
    up_data: tuple[torch.Tensor, torch.Tensor, tuple] | None,
    tp_size: int,
) -> tuple[str, dict[int, tuple[torch.Tensor, torch.Tensor]], tuple] | None:
    """Fuse gate/up sparse deltas into gate_up_proj format.

    IMPORTANT: Both components (gate, up) must be present for correct fusion.
    If any component is missing, returns None and the caller should handle fallback.

    Args:
        gate_data: (indices, values, shape) for gate_proj or None
        up_data: (indices, values, shape) for up_proj or None
        tp_size: Number of TP ranks

    Returns:
        ("gate_up_proj.weight", per_tp_data, fused_shape) or None if incomplete
    """
    # CRITICAL: Both must be present for correct fused layout
    # SGLang's gate_up_proj expects [gate_dim + up_dim, hidden]
    # If either component is missing, we cannot compute correct offsets
    if gate_data is None or up_data is None:
        if gate_data is not None or up_data is not None:
            logger.debug(
                f"Incomplete gate_up set for fusion: gate={gate_data is not None}, "
                f"up={up_data is not None}. Skipping fusion."
            )
        return None

    in_dim = gate_data[2][1]  # hidden_size

    gate_out = gate_data[2][0]
    up_out = up_data[2][0]

    # Fused shape: [(gate_out + up_out), in_dim]
    fused_out = gate_out + up_out
    fused_shape = (fused_out, in_dim)

    # Compute offsets in fused tensor (flattened)
    gate_offset = 0
    up_offset = gate_out * in_dim

    # Both gate and up are guaranteed present at this point
    gate_indices, gate_values, _ = gate_data
    up_indices, up_values, _ = up_data

    # Combine indices with offsets
    fused_indices = torch.cat([gate_indices + gate_offset, up_indices + up_offset])
    fused_values = torch.cat([gate_values, up_values])

    # gate_up_proj is ColumnParallel
    per_tp_data = split_indices_by_tp_column(
        fused_indices, fused_values, fused_out, in_dim, tp_size
    )

    return ("gate_up_proj.weight", per_tp_data, fused_shape)


class UpdateWeightFromDelta:
    """
    Update rollout engines using sparse delta updates.

    Only synchronizes the elements that have changed since the last sync,
    which is more efficient for RL training where only ~1% of weights change per step.

    Key principle: NEVER modify original tensors before IPC send completes.
    """

    # Key dense parameters to verify (non-MoE)
    VERIFY_PARAMS = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0

        # Enable verification if --delta-verify is set
        self._verify_enabled = getattr(args, 'delta_verify', False)

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )

        # Store the last synced weights for delta comparison (CPU)
        self._last_synced_weights: dict[str, torch.Tensor] = {}

        # Track if this is the first sync (use full sync for first time)
        self._is_first_sync = True

        # TP size for index splitting
        self._tp_size = args.rollout_num_gpus_per_engine

        # Create IPC gather group for colocated engines
        for start_rank in range(0, dist.get_world_size(), self.args.rollout_num_gpus_per_engine):
            end_rank = start_rank + self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            new_group = dist.new_group(ranks=group_ranks, backend="gloo")
            if dist.get_rank() in group_ranks:
                self._ipc_gather_group = new_group
                self._ipc_gather_src = start_rank

    def connect_rollout_engines(
        self, rollout_engines: Sequence[ActorHandle], rollout_engine_lock: ActorHandle
    ) -> None:
        """Connect to rollout engines."""
        self.rollout_engines = rollout_engines
        colocate_engine_nums = (
            self.args.actor_num_nodes * self.args.actor_num_gpus_per_node // self.args.rollout_num_gpus_per_engine
        )

        # For now, only support colocated engines for delta sync
        if len(rollout_engines) > colocate_engine_nums:
            raise NotImplementedError("Delta weight sync does not support distributed engines yet")

        # Map ranks to colocated IPC engines
        for i, engine in enumerate(self.rollout_engines):
            start_rank = i * self.args.rollout_num_gpus_per_engine
            end_rank = (i + 1) * self.args.rollout_num_gpus_per_engine
            group_ranks = list(range(start_rank, end_rank))
            if dist.get_rank() in group_ranks:
                self._ipc_engine = engine

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Update weights using delta sync.

        First sync: Uses baseline FlattenedTensorBucket for full weight transfer.
        Subsequent syncs: Computes element-level sparse delta, sends only changed elements.
        """
        self.weight_version += 1
        rank = dist.get_rank()

        logger.info(f"[Rank {rank}] Delta update_weights start, version={self.weight_version}, is_first={self._is_first_sync}")

        if rank == 0:
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        # Collect param names for verification before sync
        verify_params = []
        if self._verify_enabled:
            verify_params = self._collect_verify_params(megatron_local_weights)

        if self._is_first_sync:
            # First sync: use baseline method (FlattenedTensorBucket) for full weight transfer
            self._first_sync_full_weights(megatron_local_weights)
            self._is_first_sync = False
        else:
            # Subsequent syncs: use element-level sparse delta
            self._delta_sync_weights(megatron_local_weights)

        # IMPORTANT: Final barrier to ensure all ranks complete sync before continuing.
        # All ranks must reach this point before any rank proceeds to training/rollout.
        dist.barrier(group=get_gloo_group())

        # === Verification (方案B: 挪出 collective 窗口) ===
        # After the final barrier, all collective operations are done.
        # Verification runs ONLY on gather_src rank and does NOT block other ranks.
        # Other ranks can immediately proceed to training/rollout.
        #
        # Key design decisions:
        # 1. NO barrier after verify - verification is informational only
        # 2. gather_src rank does slow ray.get() calls, other ranks don't wait
        # 3. If verification fails, AssertionError is raised on gather_src rank only
        # 4. This avoids NCCL timeout because no collective ops after verify
        if self._verify_enabled and verify_params:
            # _verify_sync_correctness already checks rank == gather_src internally
            self._verify_sync_correctness(verify_params, megatron_local_weights)

    def _first_sync_full_weights(self, megatron_local_weights) -> None:
        """First sync using baseline FlattenedTensorBucket method.

        After send completes, store weights for future comparison.
        """
        rank = dist.get_rank()
        tensors_to_store = []

        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            # Send using baseline function
            refs, long_lived_tensors = _send_to_colocated_engine(
                hf_named_tensors,
                ipc_engine=self._ipc_engine,
                ipc_gather_src=self._ipc_gather_src,
                ipc_gather_group=self._ipc_gather_group,
                weight_version=self.weight_version,
            )
            ray.get(refs)
            del long_lived_tensors

            # Collect tensors to store (AFTER send completes)
            tensors_to_store.extend(hf_named_tensors)

        # Store all weights for future comparison (AFTER all sends complete)
        for name, tensor in tensors_to_store:
            self._last_synced_weights[name] = tensor.detach().cpu().clone()

        logger.info(f"[Rank {rank}] First sync completed, stored {len(self._last_synced_weights)} params")

    def _delta_sync_weights(self, megatron_local_weights) -> None:
        """Subsequent syncs using element-level sparse delta.

        Flow:
        1. For each chunk, classify params into sparse/dense
        2. Send sparse params first (custom format)
        3. Send dense params using baseline FlattenedTensorBucket
        4. After all sends complete, update stored weights
        """
        rank = dist.get_rank()
        all_tensors_to_store = []
        chunk_idx = 0

        # Statistics counters (clear naming for log output)
        total_non_moe_sparse_params = 0    # Non-MoE params updated via sparse path
        total_non_moe_dense_params = 0     # Non-MoE params updated via dense/baseline path
        total_skipped_params = 0           # Params with no changes
        total_non_moe_sparse_elements = 0  # Total elements in non-MoE sparse updates
        total_non_moe_dense_elements = 0   # Total elements in non-MoE dense updates
        total_moe_sparse_params = 0        # MoE expert params updated via sparse path
        total_moe_dense_params = 0         # MoE expert params updated via dense path (threshold fallback)

        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            chunk_start = time.time()
            diff_time = 0.0  # Time spent on diff/indices computation
            send_time = 0.0  # Time spent on serialize+gather+sglang_apply

            # Phase 1: Classify params and compute deltas (read-only operations)
            sparse_params = []  # [(name, per_tp_data, shape, partition_dim)]
            dense_params = []   # [(name, tensor)]
            moe_expert_dense_params = []  # [(name, tensor, moe_info)] - MoE dense/full update
            moe_expert_sparse_params = []  # [(name, per_tp_data, shape, moe_info)] - MoE sparse update
            tensors_for_storage = []  # [(name, tensor)] - collect for post-send storage

            # Buffers for fusable params (keyed by layer_id)
            # Each stores (indices, values, shape) for the raw delta BEFORE TP split
            qkv_buffer: dict[int, dict[str, tuple]] = {}  # layer_id -> {"q": ..., "k": ..., "v": ...}
            gate_up_buffer: dict[int, dict[str, tuple]] = {}  # layer_id -> {"gate": ..., "up": ...}

            for name, tensor in hf_named_tensors:
                tensors_for_storage.append((name, tensor))

                # MoE expert params: Step 5.1 + 5.3 - expert write path with sparse support
                if is_moe_expert_param(name):
                    moe_info = parse_moe_expert_info(name)
                    if moe_info is not None:
                        expert_id, proj_type = moe_info
                        # Step 5.1+5.2+5.4: Enable all MoE expert projections
                        # - gate_proj, up_proj: ColumnParallel (dim 0)
                        # - down_proj: RowParallel (dim 1)
                        if proj_type in ("gate_proj", "up_proj", "down_proj"):
                            moe_info_dict = {
                                "expert_id": expert_id,
                                "proj_type": proj_type,
                            }

                            # Step 5.3: MoE sparse diff - check if we have old weights
                            if name in self._last_synced_weights:
                                old_cpu = self._last_synced_weights[name]
                                if old_cpu.shape == tensor.shape:
                                    # Do bitwise diff for MoE expert
                                    old_gpu = old_cpu.to(tensor.device, non_blocking=False)
                                    changed_mask = (tensor != old_gpu)
                                    num_changed = changed_mask.sum().item()

                                    if num_changed == 0:
                                        # No changes - skip
                                        total_skipped_params += 1
                                        del old_gpu
                                        continue

                                    changed_ratio = num_changed / tensor.numel()

                                    if changed_ratio <= DENSE_THRESHOLD:
                                        # MoE sparse path: extract indices and values
                                        changed_indices = torch.nonzero(changed_mask.view(-1), as_tuple=True)[0]
                                        changed_values = tensor.view(-1)[changed_indices].clone()

                                        # Determine partition dim for MoE
                                        # gate_proj/up_proj: ColumnParallel (dim 0)
                                        # down_proj: RowParallel (dim 1)
                                        if proj_type in ("gate_proj", "up_proj"):
                                            partition_dim = 0
                                        else:  # down_proj
                                            partition_dim = 1

                                        shape = tensor.shape
                                        out_dim, in_dim = shape

                                        if partition_dim == 0:
                                            per_tp_data = split_indices_by_tp_column(
                                                changed_indices, changed_values, out_dim, in_dim, self._tp_size
                                            )
                                        else:
                                            per_tp_data = split_indices_by_tp_row(
                                                changed_indices, changed_values, out_dim, in_dim, self._tp_size
                                            )

                                        moe_expert_sparse_params.append((name, per_tp_data, shape, moe_info_dict))
                                        total_moe_sparse_params += 1
                                        del old_gpu
                                        continue

                                    del old_gpu

                            # Fallback to dense MoE update (first sync or high change ratio)
                            moe_expert_dense_params.append((name, tensor, moe_info_dict))
                            total_moe_dense_params += 1
                            continue
                    # Fallback to dense for parse failure
                    dense_params.append((name, tensor))
                    total_non_moe_dense_params += 1
                    total_non_moe_dense_elements += tensor.numel()
                    continue

                # New param (not in storage) - use dense
                if name not in self._last_synced_weights:
                    dense_params.append((name, tensor))
                    total_non_moe_dense_params += 1
                    total_non_moe_dense_elements += tensor.numel()
                    continue

                # Compare with stored weights
                old_cpu = self._last_synced_weights[name]

                if old_cpu.shape != tensor.shape:
                    # Shape changed - use dense
                    dense_params.append((name, tensor))
                    total_non_moe_dense_params += 1
                    total_non_moe_dense_elements += tensor.numel()
                    continue

                # GPU comparison (read-only, doesn't modify original tensor)
                old_gpu = old_cpu.to(tensor.device, non_blocking=False)
                changed_mask = (tensor != old_gpu)
                num_changed = changed_mask.sum().item()

                if num_changed == 0:
                    # No changes - skip entirely
                    total_skipped_params += 1
                    continue

                changed_ratio = num_changed / tensor.numel()

                if changed_ratio > DENSE_THRESHOLD:
                    # Too many changes - use dense
                    dense_params.append((name, tensor))
                    total_non_moe_dense_params += 1
                    total_non_moe_dense_elements += tensor.numel()
                    del old_gpu
                    continue

                # Sparse path: extract indices and values
                changed_indices = torch.nonzero(changed_mask.view(-1), as_tuple=True)[0]
                changed_values = tensor.view(-1)[changed_indices].clone()  # Clone to decouple from original

                shape = tensor.shape
                layer_id = extract_layer_id(name)

                # Check if this is a fusable param (qkv or gate_up)
                if is_fusable_qkv_param(name) and layer_id is not None:
                    # Store raw delta for later fusion (don't split by TP yet)
                    if layer_id not in qkv_buffer:
                        qkv_buffer[layer_id] = {}
                    parts = name.replace(".weight", "").split(".")
                    component = parts[-1]  # q_proj, k_proj, or v_proj
                    key = component.replace("_proj", "")  # q, k, or v
                    qkv_buffer[layer_id][key] = (changed_indices, changed_values, shape)
                    total_non_moe_sparse_params += 1
                    total_non_moe_sparse_elements += num_changed
                    del old_gpu
                    continue

                if is_fusable_gate_up_param(name) and layer_id is not None:
                    # Store raw delta for later fusion
                    if layer_id not in gate_up_buffer:
                        gate_up_buffer[layer_id] = {}
                    parts = name.replace(".weight", "").split(".")
                    component = parts[-1]  # gate_proj or up_proj
                    key = component.replace("_proj", "")  # gate or up
                    gate_up_buffer[layer_id][key] = (changed_indices, changed_values, shape)
                    total_non_moe_sparse_params += 1
                    total_non_moe_sparse_elements += num_changed
                    del old_gpu
                    continue

                # Non-fusable param: Split by TP immediately
                partition_dim = get_partition_dim(name)

                if len(shape) == 2 and partition_dim is not None:
                    out_dim, in_dim = shape
                    if partition_dim == 0:
                        per_tp_data = split_indices_by_tp_column(
                            changed_indices, changed_values, out_dim, in_dim, self._tp_size
                        )
                    else:  # partition_dim == 1
                        per_tp_data = split_indices_by_tp_row(
                            changed_indices, changed_values, out_dim, in_dim, self._tp_size
                        )
                else:
                    # Replicated or 1D: all ranks get same data
                    per_tp_data = {
                        r: (changed_indices.cpu().to(torch.int32), changed_values.cpu())
                        for r in range(self._tp_size)
                    }

                sparse_params.append((name, per_tp_data, shape, partition_dim))
                total_non_moe_sparse_params += 1
                total_non_moe_sparse_elements += num_changed

                del old_gpu

            # Record diff time (Phase 1 complete)
            diff_time = time.time() - chunk_start

            # Phase 1b: Fuse buffered qkv and gate_up params
            for layer_id, components in qkv_buffer.items():
                q_data = components.get("q")
                k_data = components.get("k")
                v_data = components.get("v")
                fused = fuse_qkv_sparse_deltas(q_data, k_data, v_data, self._tp_size)
                if fused:
                    fused_name, per_tp_data, fused_shape = fused
                    # Construct full param name: model.layers.{layer_id}.self_attn.qkv_proj.weight
                    full_name = f"model.layers.{layer_id}.self_attn.{fused_name}"
                    sparse_params.append((full_name, per_tp_data, fused_shape, 0))  # ColumnParallel

            for layer_id, components in gate_up_buffer.items():
                gate_data = components.get("gate")
                up_data = components.get("up")
                fused = fuse_gate_up_sparse_deltas(gate_data, up_data, self._tp_size)
                if fused:
                    fused_name, per_tp_data, fused_shape = fused
                    # Construct full param name: model.layers.{layer_id}.mlp.gate_up_proj.weight
                    full_name = f"model.layers.{layer_id}.mlp.{fused_name}"
                    sparse_params.append((full_name, per_tp_data, fused_shape, 0))  # ColumnParallel

            # Phase 2a: Send sparse params (custom format, per-TP)
            send_start = time.time()
            if sparse_params:
                self._send_sparse_params(sparse_params)

            # Phase 2a.5: Send MoE expert params (Step 5.1+5.3)
            # Send sparse MoE params first (if any)
            if moe_expert_sparse_params:
                self._send_moe_expert_sparse_params(moe_expert_sparse_params)

            # Send dense MoE params (first sync or high change ratio)
            if moe_expert_dense_params:
                self._send_moe_expert_params(moe_expert_dense_params)

            # Phase 2b: Send dense params (baseline FlattenedTensorBucket)
            if dense_params:
                refs, long_lived = _send_to_colocated_engine(
                    dense_params,
                    ipc_engine=self._ipc_engine,
                    ipc_gather_src=self._ipc_gather_src,
                    ipc_gather_group=self._ipc_gather_group,
                    weight_version=self.weight_version,
                )
                ray.get(refs)
                del long_lived
            send_time = time.time() - send_start

            # Collect for storage
            all_tensors_to_store.extend(tensors_for_storage)

            chunk_time = time.time() - chunk_start
            fuse_time = chunk_time - diff_time - send_time  # Time for fusion operations
            if rank == 0 and chunk_idx % 20 == 0:
                logger.info(
                    f"[Rank {rank}] Chunk {chunk_idx}: "
                    f"non_moe_sparse={len(sparse_params)}, moe_sparse={len(moe_expert_sparse_params)}, "
                    f"time={chunk_time:.2f}s (diff={diff_time:.2f}s, fuse={fuse_time:.2f}s, send={send_time:.2f}s)"
                )
            chunk_idx += 1

        # Phase 3: Update storage (AFTER all sends complete)
        for name, tensor in all_tensors_to_store:
            self._last_synced_weights[name] = tensor.detach().cpu().clone()

        logger.info(
            f"[Rank {rank}] Delta sync done: "
            f"non_moe_sparse={total_non_moe_sparse_params} ({total_non_moe_sparse_elements} elements), "
            f"non_moe_dense={total_non_moe_dense_params} ({total_non_moe_dense_elements} elements), "
            f"moe_sparse={total_moe_sparse_params}, moe_dense={total_moe_dense_params}, "
            f"skipped={total_skipped_params}"
        )

    def _send_sparse_params(
        self,
        sparse_params: list[tuple[str, dict[int, tuple[torch.Tensor, torch.Tensor]], tuple, int | None]],
    ) -> None:
        """Send sparse parameters to colocated engine.

        Each TP rank receives only its own local indices/values.

        Args:
            sparse_params: List of (name, per_tp_data, shape, partition_dim)
                per_tp_data: Dict mapping tp_rank -> (local_indices, local_values)
        """
        rank = dist.get_rank()

        # Determine which TP rank this process is
        tp_rank_in_group = rank - self._ipc_gather_src

        # Build sparse delta chunks for this TP rank
        delta_chunks = []
        for name, per_tp_data, shape, partition_dim in sparse_params:
            local_indices, local_values = per_tp_data.get(tp_rank_in_group, (
                torch.tensor([], dtype=torch.int32),
                torch.tensor([], dtype=torch.bfloat16),
            ))

            # Compute local shape for this TP rank
            if partition_dim == 0 and len(shape) == 2:
                # ColumnParallel: out_dim is partitioned
                local_shape = (shape[0] // self._tp_size, shape[1])
            elif partition_dim == 1 and len(shape) == 2:
                # RowParallel: in_dim is partitioned
                local_shape = (shape[0], shape[1] // self._tp_size)
            else:
                # Replicated
                local_shape = shape

            delta_chunks.append({
                "param_name": name,
                "is_sparse": True,
                "local_indices": local_indices,
                "local_values": local_values,
                "local_shape": local_shape,
            })

        # Serialize and gather
        serialized_delta = base64.b64encode(pickle.dumps(delta_chunks)).decode('ascii')

        serialized_delta_list = (
            [None] * dist.get_world_size(self._ipc_gather_group)
            if self._ipc_gather_src == rank else None
        )
        dist.gather_object(
            serialized_delta,
            object_gather_list=serialized_delta_list,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )

        # Send to engine
        if rank == self._ipc_gather_src:
            refs = [
                self._ipc_engine.update_weights_from_delta.remote(
                    serialized_delta_chunks=serialized_delta_list,
                    flush_cache=False,
                    weight_version=str(self.weight_version),
                    protocol_version=DELTA_SYNC_PROTOCOL_VERSION,
                )
            ]
            ray.get(refs)

    def _send_moe_expert_params(
        self,
        moe_expert_params: list[tuple[str, torch.Tensor, dict]],
    ) -> None:
        """Send MoE expert parameters for direct expert write (Step 5.1).

        This method sends expert weights with moe_info metadata, bypassing load_weights()
        for more efficient direct write to fused tensor positions.

        Args:
            moe_expert_params: List of (name, tensor, moe_info)
                moe_info: Dict with 'expert_id' (global) and 'proj_type'
        """
        rank = dist.get_rank()

        # Build MoE expert chunks for this rank
        # For Step 5.1, we send the whole expert tensor (no TP splitting yet)
        moe_chunks = []
        for name, tensor, moe_info in moe_expert_params:
            moe_chunks.append({
                "param_name": name,
                "is_sparse": False,
                "is_moe_expert": True,
                "tensor": tensor.cpu(),  # Move to CPU for serialization
                "shape": tuple(tensor.shape),
                "moe_info": moe_info,  # Contains expert_id and proj_type
            })

        # Serialize and gather
        serialized_moe = base64.b64encode(pickle.dumps(moe_chunks)).decode('ascii')

        serialized_moe_list = (
            [None] * dist.get_world_size(self._ipc_gather_group)
            if self._ipc_gather_src == rank else None
        )
        dist.gather_object(
            serialized_moe,
            object_gather_list=serialized_moe_list,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )

        # Send to engine
        if rank == self._ipc_gather_src:
            refs = [
                self._ipc_engine.update_weights_from_delta.remote(
                    serialized_delta_chunks=serialized_moe_list,
                    flush_cache=False,
                    weight_version=str(self.weight_version),
                    protocol_version=DELTA_SYNC_PROTOCOL_VERSION,
                )
            ]
            ray.get(refs)

    def _send_moe_expert_sparse_params(
        self,
        moe_expert_sparse_params: list[tuple[str, dict[int, tuple[torch.Tensor, torch.Tensor]], tuple, dict]],
    ) -> None:
        """Send sparse MoE expert parameters (Step 5.3).

        This method sends only changed indices/values for MoE experts, enabling
        efficient sparse updates instead of full tensor replacement.

        Args:
            moe_expert_sparse_params: List of (name, per_tp_data, shape, moe_info)
                per_tp_data: Dict mapping tp_rank -> (local_indices, local_values)
                shape: Original tensor shape (for computing offsets)
                moe_info: Dict with 'expert_id' (global) and 'proj_type'
        """
        rank = dist.get_rank()

        # Build sparse MoE chunks for this rank
        moe_sparse_chunks = []
        for name, per_tp_data, shape, moe_info in moe_expert_sparse_params:
            # Get the local indices/values for each TP rank
            per_tp_serialized = {}
            for tp_rank, (indices, values) in per_tp_data.items():
                per_tp_serialized[tp_rank] = {
                    "indices": indices.cpu() if isinstance(indices, torch.Tensor) else indices,
                    "values": values.cpu() if isinstance(values, torch.Tensor) else values,
                }

            moe_sparse_chunks.append({
                "param_name": name,
                "is_sparse": True,
                "is_moe_expert": True,
                "per_tp_data": per_tp_serialized,
                "shape": tuple(shape),
                "moe_info": moe_info,  # Contains expert_id and proj_type
            })

        # Serialize and gather
        serialized_moe = base64.b64encode(pickle.dumps(moe_sparse_chunks)).decode('ascii')

        serialized_moe_list = (
            [None] * dist.get_world_size(self._ipc_gather_group)
            if self._ipc_gather_src == rank else None
        )
        dist.gather_object(
            serialized_moe,
            object_gather_list=serialized_moe_list,
            dst=self._ipc_gather_src,
            group=self._ipc_gather_group,
        )

        # Send to engine
        if rank == self._ipc_gather_src:
            refs = [
                self._ipc_engine.update_weights_from_delta.remote(
                    serialized_delta_chunks=serialized_moe_list,
                    flush_cache=False,
                    weight_version=str(self.weight_version),
                    protocol_version=DELTA_SYNC_PROTOCOL_VERSION,
                )
            ]
            ray.get(refs)

    def _collect_verify_params(self, megatron_local_weights) -> list[str]:
        """Collect parameter names to verify (sample from key dense params)."""
        verify_params = []

        # Get first chunk to find param names
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            for name, tensor in hf_named_tensors:
                # Only verify non-MoE params with key names
                if is_moe_expert_param(name):
                    continue

                # Check if this is a key param type
                for key_param in self.VERIFY_PARAMS:
                    if key_param in name:
                        verify_params.append(name)
                        break

                # Limit to first few layers for efficiency
                if len(verify_params) >= 10:
                    break

            if len(verify_params) >= 10:
                break

        return verify_params

    def _verify_sync_correctness(
        self,
        verify_params: list[str],
        megatron_local_weights,
    ) -> None:
        """Verify sync correctness by comparing BITWISE hashes between Slime and SGLang.

        Uses int16 view -> int32 sum for deterministic, collision-resistant hashing.
        Samples 3 regions per parameter and compares hash values.
        Raises AssertionError if mismatch detected, with detailed dump.
        """
        rank = dist.get_rank()

        if rank != self._ipc_gather_src:
            # Only the gather source rank does verification
            return

        logger.info(f"[Rank {rank}] Verifying sync correctness for {len(verify_params)} params...")

        # Get hashes from SGLang
        try:
            response = ray.get(self._ipc_engine.get_param_sample_hashes.remote(verify_params))
            sglang_hashes = response.get("hashes_by_rank", [])
        except Exception as e:
            logger.error(f"Failed to get SGLang hashes: {e}")
            return

        if not sglang_hashes:
            logger.warning("No hashes received from SGLang")
            return

        # Get HF tensors for comparison
        hf_tensors_dict = {}
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            for name, tensor in hf_named_tensors:
                if name in verify_params:
                    hf_tensors_dict[name] = tensor

        # Compare with each TP rank's hashes
        mismatches = []
        for tp_idx, tp_hashes in enumerate(sglang_hashes):
            if not tp_hashes:
                continue

            for name, info in tp_hashes.items():
                if name not in hf_tensors_dict:
                    continue

                hf_tensor = hf_tensors_dict[name]
                sglang_hash = info.get("hash", 0)
                sglang_shape = info.get("shape", ())
                sglang_numel = info.get("numel", 0)
                sglang_bits = info.get("sample_bits", [])
                sglang_indices = info.get("sample_indices", [])

                # Get TP shard and compute bitwise hash
                partition_dim = get_partition_dim(name)
                local_tensor = self._get_tp_shard(hf_tensor, partition_dim, tp_idx)

                if local_tensor is None:
                    continue

                # Compute BITWISE hash matching SGLang's method
                local_hash, local_bits, local_indices = self._compute_bitwise_hash(local_tensor)

                # Compare hashes (exact match for bitwise)
                if local_hash != sglang_hash:
                    mismatches.append({
                        "param_name": name,
                        "tp_rank": tp_idx,
                        "slime_hash": local_hash,
                        "sglang_hash": sglang_hash,
                        "slime_shape": tuple(local_tensor.shape),
                        "sglang_shape": sglang_shape,
                        "slime_bits": local_bits,
                        "sglang_bits": sglang_bits,
                        "slime_indices": local_indices,
                        "sglang_indices": sglang_indices,
                        "partition_dim": partition_dim,
                    })

        if mismatches:
            logger.error(f"[Rank {rank}] VERIFICATION FAILED! {len(mismatches)} mismatches:")
            for m in mismatches[:5]:  # Show first 5
                logger.error(f"  MISMATCH: {m['param_name']}")
                logger.error(f"    tp_rank={m['tp_rank']}, partition_dim={m['partition_dim']}")
                logger.error(f"    shapes: slime={m['slime_shape']} sglang={m['sglang_shape']}")
                logger.error(f"    hash: slime={m['slime_hash']} sglang={m['sglang_hash']}")
                logger.error(f"    sample_indices: slime={m['slime_indices']} sglang={m['sglang_indices']}")
                logger.error(f"    sample_bits: slime={m['slime_bits']} sglang={m['sglang_bits']}")
            raise AssertionError(f"Delta sync verification failed: {len(mismatches)} parameter mismatches")
        else:
            logger.info(f"[Rank {rank}] Verification PASSED for {len(verify_params)} params")

    def _get_tp_shard(
        self,
        tensor: torch.Tensor,
        partition_dim: int | None,
        tp_rank: int,
    ) -> torch.Tensor | None:
        """Get the TP shard of a tensor for a specific rank."""
        if partition_dim is None or self._tp_size == 1:
            return tensor

        if len(tensor.shape) != 2:
            return tensor

        out_dim, in_dim = tensor.shape

        if partition_dim == 0:
            # ColumnParallel: partition along output dim
            shard_out = out_dim // self._tp_size
            start = tp_rank * shard_out
            end = start + shard_out
            return tensor[start:end, :]
        elif partition_dim == 1:
            # RowParallel: partition along input dim
            shard_in = in_dim // self._tp_size
            start = tp_rank * shard_in
            end = start + shard_in
            return tensor[:, start:end]
        else:
            return tensor

    def _compute_bitwise_hash(
        self, tensor: torch.Tensor
    ) -> tuple[int, list[int], list[int]]:
        """Compute BITWISE sampling hash matching SGLang's method.

        Uses int16 view -> int32 sum for deterministic, collision-resistant hashing.
        BF16 is 16-bit, so viewing as int16 is safe.

        Returns:
            (hash_value, sample_bits, sample_indices)
            - hash_value: int32 sum of int16-viewed sample values
            - sample_bits: first 5 raw int16 values (for debug)
            - sample_indices: first 10 sample indices (for debug)
        """
        flat = tensor.view(-1)
        numel = flat.numel()

        # Same sampling as SGLang: 3 regions, 10 elements each
        sample_size = min(10, numel // 3)
        indices = []
        # Start region
        indices.extend(range(0, sample_size))
        # Middle region
        mid_start = numel // 2 - sample_size // 2
        indices.extend(range(mid_start, mid_start + sample_size))
        # End region
        indices.extend(range(numel - sample_size, numel))

        # Filter valid indices
        indices = [idx for idx in indices if 0 <= idx < numel]
        if not indices:
            return 0, [], []

        indices_tensor = torch.tensor(indices, dtype=torch.int64, device=tensor.device)
        sample_values = flat[indices_tensor]

        # BITWISE hash: view as int16, sum as int32 (deterministic, no float issues)
        # bf16 is 16-bit, so view as int16 is safe
        sample_int16 = sample_values.view(torch.int16)
        hash_value = sample_int16.to(torch.int32).sum().item()

        # Store raw bits for debugging (first 5)
        sample_bits = (
            sample_int16[:5].cpu().tolist()
            if len(sample_int16) >= 5
            else sample_int16.cpu().tolist()
        )

        # First 10 indices for debug
        sample_indices_debug = indices[:10]

        return hash_value, sample_bits, sample_indices_debug
