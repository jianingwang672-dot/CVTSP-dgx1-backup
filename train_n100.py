##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0
REWARD_PARALLEL = True
REWARD_BACKEND = "persistent_pool"
REWARD_PARALLEL_WORKERS = 64
PARALLEL_SOLVER_THREADS = 1
SOLVER_BACKEND = "cvxpylayer"


##########################################################################################
# import

import logging
import os

from src.TSPTrainer import OnlineTSPTrainer as Trainer


##########################################################################################
# parameters

env_params = {
    "dataset_dir": "instance/Data",
    "min_problem_size": 20,
    "max_problem_size": 100,
    "problem_sizes": None,
    "problem_size": 100,
    "pomo_size": 100,
    "pomo_divisor": None,
    "start_node_strategy": "nearest_depot",
}

model_params = {
    "embedding_dim": 128,
    "encoder_layer_num": 6,
    "qkv_dim": 16,
    "head_num": 8,
    "logit_clipping": 10,
    "ff_hidden_dim": 512,
}

optimizer_params = {
    "optimizer": {
        "lr": 5e-5,
        "weight_decay": 1e-6,
    },
    "scheduler": {
        "milestones": [501],
        "gamma": 0.1,
    },
}

trainer_params = {
    "use_cuda": USE_CUDA,
    "cuda_device_num": CUDA_DEVICE_NUM,
    "epochs": 1,
    "resume_extra_epochs": True,
    "train_episodes": 100 * 1000,
    "train_batch_size": 512,
    "checkpoint_interval": 25,
    "solver_backend": SOLVER_BACKEND,
    "gurobi_threads": 64,
    "cvxpylayer_solver_args": {
        "eps": 1e-5,
        "max_iters": 10000,
    },
    "cvxpylayer_dtype": "float64",
    "cvxpylayer_objective_loss_enable": True,
    "cvxpylayer_hard_objective_loss_enable": False,
    "cvxpylayer_aux_enable": False,
    "cvxpylayer_aux_weight": 0.0,
    "cvxpylayer_route_candidates": 1,
    "cvxpylayer_route_max_instances_per_batch": 2,
    "cvxpylayer_route_selection": "best",
    "cvxpylayer_route_device": "cpu",
    "cvxpylayer_route_normalize_by_size": False,
    "cvxpylayer_soft_route_method": "unmasked_sinkhorn",
    "cvxpylayer_fixed_start_logit": 20.0,
    "cvxpylayer_hard_route_bias": 5.0,
    "sinkhorn_temperature": 1.0,
    "sinkhorn_iters": 100,
    "penalty_reward": -1e6,
    "grad_clip": 1.0,
    "seed": 1234,
    "result_folder": os.path.join("outputs", "train__cvxpylayer_unmasked_sinkhorn_objective_1epoch"),
    "log_level": "INFO",
    "progress_log_percent": 1.0,
    "progress_bar_width": 24,
    "reward_backend": REWARD_BACKEND,
    "reward_parallel_workers": REWARD_PARALLEL_WORKERS if REWARD_PARALLEL else 0,
    "parallel_solver_threads": PARALLEL_SOLVER_THREADS,
    "reward_parallel_chunksize": 1,
    "checkpoint_path": "/home/Mingfan/wjn/CVTSP/outputs/train__fresh/best.pt",
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    _print_config()

    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params,
    )
    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params["epochs"] = 2
    trainer_params["train_episodes"] = 10
    trainer_params["train_batch_size"] = 4


def _print_config():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("root")
    logger.info("DEBUG_MODE: %s", DEBUG_MODE)
    logger.info("USE_CUDA: %s, CUDA_DEVICE_NUM: %s", USE_CUDA, CUDA_DEVICE_NUM)
    logger.info("env_params=%s", env_params)
    logger.info("model_params=%s", model_params)
    logger.info("optimizer_params=%s", optimizer_params)
    logger.info("trainer_params=%s", trainer_params)


##########################################################################################

if __name__ == "__main__":
    main()
