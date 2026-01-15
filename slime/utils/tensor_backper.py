from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Callable, Iterable

import torch

_SourceGetter = Callable[[], Iterable[tuple[str, torch.Tensor]]]


class TensorBackuper(ABC):
    @staticmethod
    def create(source_getter, single_tag):
        if single_tag is None:
            return _TensorBackuperNormal(source_getter=source_getter)
        else:
            return _TensorBackuperNoop(source_getter=source_getter, single_tag=single_tag)

    def __init__(self, source_getter: _SourceGetter):
        self._source_getter = source_getter

    @property
    @abstractmethod
    def backup_tags(self):
        raise NotImplementedError

    @abstractmethod
    def get(self, tag: str):
        raise NotImplementedError

    @abstractmethod
    def backup(self, tag: str):
        raise NotImplementedError

    def copy(self, *, src_tag: str, dst_tag: str):
        raise NotImplementedError

    @abstractmethod
    def restore(self, tag: str):
        raise NotImplementedError


class _TensorBackuperNormal(TensorBackuper):
    def __init__(self, source_getter):
        super().__init__(source_getter=source_getter)
        self._backups: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

    @property
    def backup_tags(self):
        return list(self._backups)

    def get(self, tag: str):
        return self._backups[tag]

    @torch.no_grad()
    def backup(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        for name, param in self._source_getter():
            if name not in backup_dict:
                backup_dict[name] = torch.empty_like(param, device=torch.device("cpu"), pin_memory=True)
            backup_dict[name].copy_(param.detach(), non_blocking=True)
        torch.cuda.synchronize()

    @torch.no_grad()
    def copy(self, *, src_tag: str, dst_tag: str):
        for name in self._backups[dst_tag]:
            self._backups[dst_tag][name].copy_(self._backups[src_tag][name])

    @torch.no_grad()
    def restore(self, tag: str) -> None:
        backup_dict = self._backups[tag]
        import logging
        logger = logging.getLogger(__name__)
        for i, (name, param) in enumerate(self._source_getter()):
            assert name in backup_dict, f"Parameter {name} not found in backup"
            backup_tensor = backup_dict[name]
            # CUDA pointer diagnostics for first few parameters
            if i < 3 or "embed" in name:
                logger.info(f"[restore {tag}] {name}")
                logger.info(f"  param.data_ptr()    = {param.data_ptr()}")
                logger.info(f"  param.storage_ptr() = {param.storage().data_ptr()}")
                logger.info(f"  param.is_contiguous = {param.is_contiguous()}")
                logger.info(f"  backup device       = {backup_tensor.device}")

            try:
                param.copy_(backup_tensor, non_blocking=True)
            except RuntimeError as e:
                logger.error(f"[restore {tag}] FAILED at {name} (index {i})")
                logger.error(f"  Error: {e}")
                logger.error(f"  param.data_ptr()    = {param.data_ptr()}")
                logger.error(f"  param.storage_ptr() = {param.storage().data_ptr()}")
                logger.error(f"  param.is_contiguous = {param.is_contiguous()}")
                logger.error(f"  param shape:  {param.shape}, dtype: {param.dtype}")
                logger.error(f"  backup shape: {backup_tensor.shape}, dtype: {backup_tensor.dtype}")
                logger.error(f"  CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                logger.error(f"  CUDA memory reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
                raise
        torch.cuda.synchronize()


class _TensorBackuperNoop(TensorBackuper):
    def __init__(self, source_getter, single_tag):
        super().__init__(source_getter=source_getter)
        self._single_tag = single_tag
        # Sanity check for safety
        self._backup_hash_dict = None

    @property
    def backup_tags(self):
        return [self._single_tag]

    def get(self, tag: str):
        ans = dict(self._source_getter())
        ans = {k: v.detach() for k, v in ans.items()}
        assert _compute_hash_dict(ans) == self._backup_hash_dict
        return ans

    def backup(self, tag: str) -> None:
        assert tag == self._single_tag
        self._backup_hash_dict = _compute_hash_dict(dict(self._source_getter()))
        torch.cuda.synchronize()

    def restore(self, tag: str) -> None:
        assert tag == self._single_tag
        assert _compute_hash_dict(dict(self._source_getter())) == self._backup_hash_dict
        torch.cuda.synchronize()


def _compute_hash_dict(tensors: dict[str, torch.Tensor]):
    return {k: _compute_hash_tensor(v) for k, v in tensors.items()}


def _compute_hash_tensor(x: torch.Tensor):
    # Not a real/good hash, but pretty fast
    x = x.contiguous()
    x = x.view(-1)
    x = x.view(torch.uint32)
    x = x.sum()
    return x.item()
