import dataclasses
import os
import time
from argparse import Namespace
from collections.abc import Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from tqdm import tqdm

from slime.utils.distributed_utils import get_gloo_group
from slime.utils.types import ParamInfo

from ..megatron_to_hf import convert_to_hf
from ..sglang import monkey_patch_torch_reductions
from .common import all_gather_params_async, named_params_and_buffers
from .hf_weight_iterator_base import HfWeightIteratorBase


def _is_baseline_profile_enabled():
    """Check at runtime, not import time, because Ray sets env vars after import."""
    return os.environ.get("SLIME_BASELINE_PROFILE", "0") == "1"


def _is_deep_profile_enabled():
    """Check at runtime for deep profiling mode."""
    return os.environ.get("SLIME_DEEP_PROFILE", "0") == "1"


class HfWeightIteratorDirect(HfWeightIteratorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.megatron_local_param_info_buckets = _get_megatron_local_param_info_buckets(self.args, self.model)

    def get_hf_weight_chunks(self, megatron_local_weights):
        rank = dist.get_rank()

        if _is_baseline_profile_enabled():
            total_nccl_time = 0.0
            total_hf_convert_time = 0.0

        for megatron_local_param_infos in tqdm(
            self.megatron_local_param_info_buckets, disable=rank != 0, desc="Update weights"
        ):
            if _is_baseline_profile_enabled():
                t_nccl_start = time.time()

            megatron_full_params = _get_megatron_full_params(megatron_local_param_infos, megatron_local_weights)

            if _is_baseline_profile_enabled():
                nccl_time = time.time() - t_nccl_start
                total_nccl_time += nccl_time
                t_hf_start = time.time()

            hf_named_tensors = self._convert_to_hf_named_tensors(megatron_full_params, megatron_local_param_infos)

            if _is_baseline_profile_enabled():
                hf_convert_time = time.time() - t_hf_start
                total_hf_convert_time += hf_convert_time

            yield hf_named_tensors
            del megatron_full_params

        if _is_baseline_profile_enabled():
            # 从所有 rank 打印总结信息（只打印一次，Ray 会聚合）
            print(
                f"[HF Iterator Profile] rank={rank} total_nccl={total_nccl_time:.3f}s total_hf_convert={total_hf_convert_time:.3f}s",
                flush=True
            )

    def _convert_to_hf_named_tensors(self, megatron_full_params: Sequence[torch.Tensor], param_infos: list[ParamInfo]):
        hf_named_tensors = []
        for info, param in zip(param_infos, megatron_full_params, strict=False):
            hf_named_tensors.extend(
                convert_to_hf(self.args, self.model_name, info.name, param, self.quantization_config)
            )
        return hf_named_tensors


def _get_megatron_full_params(
    megatron_local_param_infos: Sequence[ParamInfo],
    megatron_local_weights,
) -> Sequence[torch.Tensor]:
    monkey_patch_torch_reductions()
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()
    rank = dist.get_rank()

    if _is_baseline_profile_enabled():
        t_start = time.time()

    # init params:
    params = []
    for info in megatron_local_param_infos:
        if dist.get_rank() == info.src_rank:
            params.append(
                torch.nn.Parameter(
                    megatron_local_weights[info.name].to(device=torch.cuda.current_device(), non_blocking=True),
                    requires_grad=False,
                )
            )
        else:
            params.append(torch.empty(info.shape, dtype=info.dtype, device=torch.cuda.current_device()))
    torch.cuda.synchronize()

    if _is_baseline_profile_enabled():
        init_time = time.time() - t_start
        t_pp_start = time.time()

    # broadcast params across pp ranks
    if pp_size > 1:
        handles = []
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            if info.src_rank in dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group()):
                handles.append(
                    torch.distributed.broadcast(
                        param, src=info.src_rank, group=mpu.get_pipeline_model_parallel_group(), async_op=True
                    )
                )
        for handle in handles:
            handle.wait()

    if _is_baseline_profile_enabled():
        pp_time = time.time() - t_pp_start
        t_ep_start = time.time()

    # broadcast params across ep ranks
    if ep_size > 1:
        handles = []
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            if ".experts." in info.name:
                src_rank = (
                    info.src_rank
                    if info.src_rank in dist.get_process_group_ranks(mpu.get_expert_model_parallel_group())
                    else rank
                )
                handles.append(
                    torch.distributed.broadcast(
                        param, src=src_rank, group=mpu.get_expert_model_parallel_group(), async_op=True
                    )
                )
        for handle in handles:
            handle.wait()

    if _is_baseline_profile_enabled():
        ep_time = time.time() - t_ep_start
        t_tp_start = time.time()

    # Set tp attrs for all params
    for info, param in zip(megatron_local_param_infos, params, strict=False):
        for key, value in info.attrs.items():
            setattr(param, key, value)

    # Batch async all_gather for all parameters
    gathered_params = all_gather_params_async(list(zip(megatron_local_param_infos, params, strict=False)))

    if _is_baseline_profile_enabled():
        tp_time = time.time() - t_tp_start
        total_time = time.time() - t_start
        # 计算这个 chunk 的数据量
        total_bytes = sum(p.numel() * p.element_size() for p in gathered_params)
        total_mb = total_bytes / (1024 * 1024)
        # 从所有 rank 打印（Ray 会聚合重复的日志）
        print(
            f"[NCCL Profile] rank={rank} n_params={len(megatron_local_param_infos)} data_mb={total_mb:.1f} "
            f"init={init_time:.4f}s pp={pp_time:.4f}s ep={ep_time:.4f}s tp={tp_time:.4f}s total={total_time:.4f}s",
            flush=True
        )

    return gathered_params


def _get_megatron_local_param_info_buckets(args: Namespace, model: Sequence[torch.nn.Module]) -> list[list[ParamInfo]]:
    """
    Partition params into buckets ≤ update_weight_buffer_size (with TP replication).
    """
    param_infos = _get_megatron_local_param_infos(args, model)
    param_info_buckets = [[]]  # Start with one empty bucket
    buffer_size = 0  # Track current bucket size in bytes

    for info in param_infos:
        # Expert params use expert-TP size, others use regular-TP size
        if ".experts." in info.name:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()

        # Full param size = shard size × TP replicas (all-gather will reconstruct full param)
        param_size = info.size * tp_size

        # If adding this param exceeds limit AND current bucket has params: start new bucket
        if buffer_size + param_size > args.update_weight_buffer_size and len(param_info_buckets[-1]) > 0:
            param_info_buckets.append([])
            buffer_size = 0

        # Add param to current bucket and update size
        param_info_buckets[-1].append(info)
        buffer_size += param_size

    return param_info_buckets


def _get_megatron_local_param_infos(args: Namespace, model: Sequence[torch.nn.Module]) -> list[ParamInfo]:
    """
    Build global param metadata: collect → exchange PP/EP → resolve duplicates (MTP virtual PP)
    by min src_rank → validate. Returns sorted ParamInfo identical across all ranks.
    """
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()

    param_infos = {}
    rank = dist.get_rank()
    for name, param in named_params_and_buffers(args, model):
        param_infos[name] = ParamInfo(
            name=name,
            dtype=param.dtype,
            shape=param.shape,
            attrs={
                "tensor_model_parallel": getattr(param, "tensor_model_parallel", False),
                "partition_dim": getattr(param, "partition_dim", -1),
                "partition_stride": getattr(param, "partition_stride", 1),
                "parallel_mode": getattr(param, "parallel_mode", None),
            },
            size=param.numel() * param.element_size(),
            src_rank=rank,
        )

    if pp_size > 1:
        param_infos_list = [None] * pp_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_pipeline_model_parallel_group()
        )
        for src_rank, infos in param_infos_list:
            if src_rank == rank:
                continue
            for name, info in infos.items():
                if name in param_infos:
                    assert args.mtp_num_layers is not None
                    old_info = param_infos[name]
                    if old_info.src_rank > src_rank:
                        param_infos[name] = info
                else:
                    param_infos[name] = info

    if ep_size > 1:
        param_infos_list = [None] * ep_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_expert_model_parallel_group()
        )
        for src_rank, infos in param_infos_list:
            for name, info in infos.items():
                if name not in param_infos:
                    # here we need to set the src_rank to the rank within the expert model parallel group
                    info = dataclasses.replace(info, src_rank=src_rank)
                    param_infos[name] = info

    param_infos = list(param_infos.values())
    param_infos = sorted(param_infos, key=lambda info: info.name)

    # Check all ranks has the same parameter info
    all_param_info_list = [None] * dist.get_world_size()
    dist.all_gather_object(
        obj=param_infos,
        object_list=all_param_info_list,
        group=get_gloo_group(),
    )
    for i, param_info in enumerate(param_infos):
        for infos in all_param_info_list:
            assert infos[i].name == param_info.name, f"Parameter name mismatch: {infos[i].name} != {param_info.name}"
            assert (
                infos[i].shape == param_info.shape
            ), f"Parameter shape mismatch: {infos[i].shape} != {param_info.shape}"
            assert (
                infos[i].dtype == param_info.dtype
            ), f"Parameter dtype mismatch: {infos[i].dtype} != {param_info.dtype}"

    return param_infos
