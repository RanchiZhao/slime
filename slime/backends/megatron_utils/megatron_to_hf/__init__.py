import os
import time

from .deepseekv3 import convert_deepseekv3_to_hf
from .glm4 import convert_glm4_to_hf
from .glm4moe import convert_glm4moe_to_hf
from .llama import convert_llama_to_hf
from .mimo import convert_mimo_to_hf
from .processors.padding_remover import remove_padding
from .processors.quantizer import quantize_params
from .qwen2 import convert_qwen2_to_hf
from .qwen3_next import convert_qwen3_next_to_hf
from .qwen3moe import convert_qwen3moe_to_hf


def _is_hf_convert_profile_enabled():
    """Check at runtime for HF convert profiling."""
    return os.environ.get("SLIME_HF_CONVERT_PROFILE", "0") == "1"


# Global accumulators for HF convert profiling (per chunk)
_hf_convert_profile = {
    "remove_padding_time": 0.0,
    "convert_core_time": 0.0,
    "quantize_time": 0.0,
    "param_count": 0,
}


def reset_hf_convert_profile():
    """Reset the profiling accumulators. Call at the start of each chunk."""
    global _hf_convert_profile
    _hf_convert_profile = {
        "remove_padding_time": 0.0,
        "convert_core_time": 0.0,
        "quantize_time": 0.0,
        "param_count": 0,
    }


def get_hf_convert_profile():
    """Get the current profiling data."""
    return _hf_convert_profile.copy()


# TODO unify w/ `convert_to_hf`
def postprocess_hf_param(args, megatron_param_name, hf_param_name, param):
    param = remove_padding(megatron_param_name, param, args.vocab_size)
    # TODO support quant
    return param


# TODO optimize code details
def convert_to_hf(args, model_name, name, param, quantization_config=None):
    global _hf_convert_profile
    profile_enabled = _is_hf_convert_profile_enabled()

    if profile_enabled:
        t0 = time.time()

    param = remove_padding(name, param, args.vocab_size)

    if profile_enabled:
        t1 = time.time()
        _hf_convert_profile["remove_padding_time"] += t1 - t0

    converted_named_tensors = _convert_to_hf_core(args, model_name, name, param)

    if profile_enabled:
        t2 = time.time()
        _hf_convert_profile["convert_core_time"] += t2 - t1
        _hf_convert_profile["param_count"] += 1

    if not quantization_config:
        return converted_named_tensors

    result = quantize_params(args, name, converted_named_tensors, quantization_config)

    if profile_enabled:
        t3 = time.time()
        _hf_convert_profile["quantize_time"] += t3 - t2

    return result


# TODO optimize
_cached_tensors = {}


# TODO optimize code details
def _convert_to_hf_core(args, model_name, name, param):
    if "glm4moe" in model_name:
        converted_named_tensors = convert_glm4moe_to_hf(args, name, param)
    elif "glm4" in model_name:
        converted_named_tensors = convert_glm4_to_hf(args, name, param)
    elif "qwen3moe" in model_name:
        converted_named_tensors = convert_qwen3moe_to_hf(args, name, param)
    elif "qwen3next" in model_name:
        converted_named_tensors = convert_qwen3_next_to_hf(args, name, param)
    elif "qwen2" in model_name or "qwen3" in model_name:
        converted_named_tensors = convert_qwen2_to_hf(args, name, param)
    elif "deepseekv3" in model_name:
        converted_named_tensors = convert_deepseekv3_to_hf(args, name, param)

    elif "llama" in model_name:
        converted_named_tensors = convert_llama_to_hf(args, name, param)
    elif "mimo" in model_name:
        converted_named_tensors = convert_mimo_to_hf(args, name, param)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # to compatible with sglang implementation
    if args.q_lora_rank is not None:
        old_converted_named_tensors = converted_named_tensors
        converted_named_tensors = []
        for converted_name, converted_param in old_converted_named_tensors:
            if "q_a_proj" in converted_name:
                pair_name = converted_name.replace("q_a_proj", "kv_a_proj_with_mqa")
                if pair_name in _cached_tensors:
                    converted_named_tensors += [
                        (converted_name, converted_param),
                        (pair_name, _cached_tensors[pair_name]),
                    ]
                    del _cached_tensors[pair_name]
                else:
                    _cached_tensors[converted_name] = converted_param
            elif "kv_a_proj_with_mqa" in converted_name:
                pair_name = converted_name.replace("kv_a_proj_with_mqa", "q_a_proj")
                if pair_name in _cached_tensors:
                    converted_named_tensors += [
                        (converted_name, converted_param),
                        (pair_name, _cached_tensors[pair_name]),
                    ]
                    del _cached_tensors[pair_name]
                else:
                    _cached_tensors[converted_name] = converted_param
            else:
                converted_named_tensors.append((converted_name, converted_param))
    return converted_named_tensors
