import logging
import torch
import ray
from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

try:
    from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH
except ImportError:
    GPU_MEMORY_TYPE_CUDA_GRAPH = None

from slime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from slime.utils.arguments import parse_args
from slime.utils.logging_utils import configure_logger
from slime.utils.misc import should_run_periodic_action
from slime.utils.tracking_utils import init_tracking

logger = logging.getLogger(__name__)


def _log_gpu_memory_train(label: str):
    """Log GPU memory from training process (rank 0 only)."""
    try:
        if not torch.cuda.is_available():
            return

        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / (1024**3)
        reserved = torch.cuda.memory_reserved(device) / (1024**3)

        try:
            max_memory = torch.cuda.get_device_properties(device).total_memory / (1024**3)
            free = max_memory - reserved
        except:
            max_memory = 0
            free = 0

        logger.info(
            f"[TRAIN_GPU_MEM] {label} | device={device} | "
            f"allocated={allocated:.2f}GB reserved={reserved:.2f}GB free={free:.2f}GB total={max_memory:.2f}GB"
        )
    except Exception as e:
        pass  # Silently ignore errors in training process memory logging


def train(args):
    configure_logger()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = create_training_models(args, pgs, rollout_manager)

    logger.info(f"[TRAIN_LOOP] offload_rollout={args.offload_rollout} offload_train={getattr(args, 'offload_train', False)} use_awex={getattr(args, 'use_awex', False)}")

    if args.offload_rollout:
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    # always update weight first so that sglang has the loaded weights from training.
    logger.info("[TRAIN_LOOP] BEFORE initial update_weights")
    actor_model.update_weights()
    logger.info("[TRAIN_LOOP] AFTER initial update_weights")

    if args.check_weight_update_equal:
        ray.get(rollout_manager.check_weights.remote(action="compare"))

    if args.offload_rollout:
        if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
        ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        ray.get(rollout_manager.eval.remote(rollout_id=0))

    def offload_train():
        if args.offload_train:
            if args.use_critic:
                critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    actor_model.offload()
            else:
                actor_model.offload()
        else:
            actor_model.clear_memory()

    def onload_rollout():
        if args.offload_rollout:
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_WEIGHTS]))

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    for rollout_id in range(args.start_rollout_id, args.num_rollout):
        logger.info(f"[TRAIN_LOOP] ========== ITERATION {rollout_id} START ==========")
        _log_gpu_memory_train(f"ITERATION {rollout_id} START")

        if args.eval_interval is not None and rollout_id == 0:
            ray.get(rollout_manager.eval.remote(rollout_id))

        logger.info(f"[TRAIN_LOOP] BEFORE generate() rollout_id={rollout_id}")
        rollout_data_ref = ray.get(rollout_manager.generate.remote(rollout_id))
        logger.info(f"[TRAIN_LOOP] AFTER generate() rollout_id={rollout_id}")

        if args.offload_rollout:
            logger.info(f"[TRAIN_LOOP] BEFORE offload_rollout() rollout_id={rollout_id}")
            ray.get(rollout_manager.offload.remote())
            logger.info(f"[TRAIN_LOOP] AFTER offload_rollout() rollout_id={rollout_id}")

        logger.info(f"[TRAIN_LOOP] BEFORE train() rollout_id={rollout_id}")
        _log_gpu_memory_train(f"BEFORE train() rollout_id={rollout_id}")
        if args.use_critic:
            critic_train_handle = critic_model.async_train(rollout_id, rollout_data_ref)
            if rollout_id >= args.num_critic_only_steps:
                ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
            ray.get(critic_train_handle)
        else:
            ray.get(actor_model.async_train(rollout_id, rollout_data_ref))
        logger.info(f"[TRAIN_LOOP] AFTER train() rollout_id={rollout_id}")
        _log_gpu_memory_train(f"AFTER train() rollout_id={rollout_id}")

        if should_run_periodic_action(rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout):
            if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
                actor_model.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)
            if args.use_critic:
                critic_model.save_model(rollout_id, force_sync=rollout_id == args.num_rollout - 1)
            if args.rollout_global_dataset:
                ray.get(rollout_manager.save.remote(rollout_id))

        logger.info(f"[TRAIN_LOOP] BEFORE offload_train() rollout_id={rollout_id}")
        offload_train()
        logger.info(f"[TRAIN_LOOP] AFTER offload_train() rollout_id={rollout_id}")

        logger.info(f"[TRAIN_LOOP] BEFORE onload_rollout() rollout_id={rollout_id}")
        onload_rollout()
        logger.info(f"[TRAIN_LOOP] AFTER onload_rollout() rollout_id={rollout_id}")

        logger.info(f"[TRAIN_LOOP] BEFORE update_weights() rollout_id={rollout_id}")
        _log_gpu_memory_train(f"BEFORE update_weights() rollout_id={rollout_id}")
        actor_model.update_weights()
        logger.info(f"[TRAIN_LOOP] AFTER update_weights() rollout_id={rollout_id}")
        _log_gpu_memory_train(f"AFTER update_weights() rollout_id={rollout_id}")

        if args.offload_rollout:
            if GPU_MEMORY_TYPE_CUDA_GRAPH is not None:
                ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_CUDA_GRAPH]))
            ray.get(rollout_manager.onload.remote(tags=[GPU_MEMORY_TYPE_KV_CACHE]))

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            ray.get(rollout_manager.eval.remote(rollout_id))

        logger.info(f"[TRAIN_LOOP] ========== ITERATION {rollout_id} END ==========")
        _log_gpu_memory_train(f"ITERATION {rollout_id} END")

    ray.get(rollout_manager.dispose.remote())


if __name__ == "__main__":
    args = parse_args()
    train(args)
