"""
Awex integration for Slime weight synchronization.

This module provides adapter classes to integrate awex's optimized weight
synchronization into slime's training framework.

Key optimizations over baseline:
- Eliminates chunk loop (317 chunks × 38.5ms = 12s+ savings)
- Groups tensors to reduce IPC handles
- Uses NCCL P2P for efficient transfer
"""

import logging
import os
from argparse import Namespace
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _get_torch_memory_saver():
    """Lazy import torch_memory_saver to avoid init at module load time."""
    from torch_memory_saver import torch_memory_saver
    return torch_memory_saver


@dataclass
class AwexIntegrationConfig:
    """
    Awex integration config for Training/Inference.

    Args:
        meta_server_addr: MetaServer address, format "ip:port"
        enable_colocate_mode: Enable colocate mode (training/inference share GPU)
        ipc_backend: IPC backend, "cuda" or "cpu"
        timeout: Operation timeout in seconds
        enable_debug_mode: Enable debug mode
    """

    meta_server_addr: str
    enable_colocate_mode: bool = True
    ipc_backend: str = "cuda"  # "cuda" or "cpu"
    timeout: int = 600
    enable_debug_mode: bool = False

    def __post_init__(self):
        if self.meta_server_addr is None:
            raise ValueError("meta_server_addr cannot be None")
        if self.ipc_backend not in {"cuda", "cpu"}:
            raise ValueError(f"ipc_backend must be 'cuda' or 'cpu', got {self.ipc_backend}")
        if self.timeout <= 0:
            raise ValueError(f"timeout must be positive, got {self.timeout}")

    @staticmethod
    def from_slime_args(args: Namespace) -> "AwexIntegrationConfig":
        """Build config from slime command line args."""
        meta_server_addr = getattr(args, "awex_meta_server_addr", None)
        if meta_server_addr is None:
            raise ValueError("--awex-meta-server-addr is required when using awex")

        return AwexIntegrationConfig(
            meta_server_addr=meta_server_addr,
            enable_colocate_mode=getattr(args, "awex_colocate_mode", True),
            ipc_backend=getattr(args, "awex_ipc_backend", "cuda"),
            timeout=getattr(args, "awex_timeout", 600),
            enable_debug_mode=getattr(args, "awex_debug", False),
        )

    def to_awex_train_config(self) -> dict:
        """Convert to awex train engine config dict."""
        return {
            "meta_server_addr": self.meta_server_addr,
            "enable_colocate_mode": self.enable_colocate_mode,
            "weights_exchange_ipc_backend": self.ipc_backend,
            "enable_debug_mode": self.enable_debug_mode,
        }


class SlimeAwexTrainEngine:
    """
    Slime-side awex adapter that wraps Megatron model to satisfy awex WeightsWriter interface.

    Args:
        model: Megatron model list (supports VPP)
        hf_config: HuggingFace model config
        config: AwexIntegrationConfig
        weights_getter: Callable to get current weights
        offload_train: Whether training uses offload mode (torch_memory_saver)
    """

    def __init__(
        self,
        model: Sequence[torch.nn.Module],
        hf_config: Any,
        config: AwexIntegrationConfig,
        weights_getter: Callable,
        offload_train: bool = False,
    ):
        if not model:
            raise ValueError("model list cannot be empty")
        if hf_config is None:
            raise ValueError("hf_config cannot be None")
        if config is None:
            raise ValueError("config cannot be None")

        self.model = model
        self.hf_config = hf_config
        self.config = config.to_awex_train_config()
        self.meta_server_addr = config.meta_server_addr
        self.enable_colocate_mode = config.enable_colocate_mode
        self.enable_debug_mode = config.enable_debug_mode
        self.engine_name = "mcore"  # Megatron-Core
        self._weights_getter = weights_getter
        self._offload_train = offload_train

        # Track memory state for idempotent operations
        # In Slime's offload_train mode:
        # - torch_memory_saver.pause() is called in actor.sleep()
        # - torch_memory_saver.disable() in update_weights() does NOT resume paused memory
        # - We need to call resume()/pause() directly for AWEX's memory management to work
        self._memory_resumed = False

    def release_memory_occupation(self, tags: list[str] | str | None = None):
        """Release memory occupation (needed in colocate mode).

        In Slime's offload_train mode, this calls torch_memory_saver.pause()
        to actually offload GPU memory. The tags parameter is ignored since
        torch_memory_saver doesn't support fine-grained control.
        """
        if not self._offload_train:
            return

        if self._memory_resumed:
            logger.info(f"[SlimeAwexTrainEngine] Releasing memory (tags={tags}), calling torch_memory_saver.pause()")
            _get_torch_memory_saver().pause()
            self._memory_resumed = False
        else:
            logger.debug(f"[SlimeAwexTrainEngine] release_memory_occupation(tags={tags}) - already paused, skipping")

    def resume_memory_occupation(self, tags: list[str] | str | None = None):
        """Resume memory occupation.

        In Slime's offload_train mode, this calls torch_memory_saver.resume()
        to restore GPU memory. The tags parameter is ignored since
        torch_memory_saver doesn't support fine-grained control.

        CRITICAL: In Slime, actor.py wraps update_weights() with torch_memory_saver.disable(),
        but disable() only disables allocation hooks - it does NOT resume paused memory!
        We must call resume() explicitly for AWEX to access model weights.
        """
        if not self._offload_train:
            return

        if not self._memory_resumed:
            logger.info(f"[SlimeAwexTrainEngine] Resuming memory (tags={tags}), calling torch_memory_saver.resume()")
            _get_torch_memory_saver().resume()
            self._memory_resumed = True
        else:
            logger.debug(f"[SlimeAwexTrainEngine] resume_memory_occupation(tags={tags}) - already resumed, skipping")

    def release_grad_memory(self):
        """Release gradient memory."""
        for m in self.model:
            for param in m.parameters():
                if param.grad is not None:
                    param.grad = None

    def save_hf_checkpoint(self, path: str):
        """Save HF checkpoint (for validation mode)."""
        # This is called by awex for weights validation
        # Delegate to slime's checkpoint saving logic if needed
        logger.warning(f"save_hf_checkpoint called with path={path}, not implemented")
        pass


class AwexWeightSender:
    """
    Slime-side weight sender that wraps awex NCCLWeightsWriter.

    Args:
        train_engine: SlimeAwexTrainEngine instance
        config: AwexIntegrationConfig
    """

    def __init__(self, train_engine: SlimeAwexTrainEngine, config: AwexIntegrationConfig):
        if train_engine is None:
            raise ValueError("train_engine cannot be None")
        if config is None:
            raise ValueError("config cannot be None")

        self._train_engine = train_engine
        self._config = config
        self._writer = None  # Lazy init
        self._initialized = False
        self._weight_version = 0

    def _create_writer(self):
        """Create awex NCCLWeightsWriter (lazy initialization)."""
        try:
            from awex.writer.nccl_writer import NCCLWeightsWriter

            return NCCLWeightsWriter(self._train_engine)
        except ImportError as e:
            raise ImportError(
                "awex is not installed. Please install it with: "
                "cd /mnt/hisys-data/yqzhao/asystem-awex && pip install -e ."
            ) from e

    def initialize(self):
        """Lazy initialization, wait for inference side to be ready."""
        if not self._initialized:
            rank = dist.get_rank() if dist.is_initialized() else 0
            logger.info(f"[Rank {rank}] Initializing AwexWeightSender...")

            self._writer = self._create_writer()
            self._writer.initialize()
            self._initialized = True

            logger.info(f"[Rank {rank}] AwexWeightSender initialized successfully")

    def update_weights(self) -> None:
        """
        Send weights to inference side.

        This replaces the baseline chunk loop with a single awex call.
        """
        rank = dist.get_rank() if dist.is_initialized() else 0
        logger.info(f"[AwexWeightSender] Rank {rank} entering update_weights()")
        
        logger.info(f"[AwexWeightSender] Rank {rank} calling initialize()")
        self.initialize()
        logger.info(f"[AwexWeightSender] Rank {rank} initialize() completed")

        self._weight_version += 1
        logger.info(f"[AwexWeightSender] Rank {rank} incremented weight_version to {self._weight_version}")

        if rank == 0:
            logger.info(
                f"[AwexWeightSender] Sending weights version {self._weight_version}..."
            )

        logger.info(f"[AwexWeightSender] Rank {rank} about to call _writer.write_weights(step_id={self._weight_version})")
        self._writer.write_weights(step_id=self._weight_version)
        logger.info(f"[AwexWeightSender] Rank {rank} returned from _writer.write_weights()")

        if rank == 0:
            logger.info(
                f"[AwexWeightSender] Weights version {self._weight_version} sent successfully"
            )

        assert self._weight_version > 0, "Post-condition failed: weight_version > 0"
        logger.info(f"[AwexWeightSender] Rank {rank} update_weights() completed")


def is_awex_enabled(args: Namespace) -> bool:
    """Check if awex integration is enabled."""
    return getattr(args, "use_awex", False)


def get_awex_meta_server_addr() -> str | None:
    """Get MetaServer address from environment variable."""
    return os.environ.get("AWEX_META_SERVER_ADDR")
