##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0
SOLVER_BACKEND = "cvxpylayer"


##########################################################################################
# import

import argparse
import logging
import os

from src.TSPTester import OnlineTSPTester as Tester


##########################################################################################
# parameters

env_params = {
    "dataset_dir": "instance/Data",
    "min_problem_size": 20,
    "max_problem_size": 100,
    "problem_sizes": [20, 40, 60, 80, 100],
    "problem_size": 100,
    "pomo_size": 100,
    "pomo_divisor": 4,
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

tester_params = {
    "use_cuda": USE_CUDA,
    "cuda_device_num": CUDA_DEVICE_NUM,
    "model_load": {
        "path": os.path.join("outputs", "train__cvtsp_random_n100"),
        "epoch": 1000,
    },
    "test_mode": "real529",
    "real_split": "all",
    "test_episodes": 100 * 1000,
    "test_batch_size": 64,
    "decode_type": "sample",
    "augmentation_enable": False,
    "aug_factor": 8,
    "solver_backend": SOLVER_BACKEND,
    "gurobi_threads": 32,
    "cvxpylayer_solver_args": {
        "eps": 1e-5,
        "max_iters": 10000,
    },
    "cvxpylayer_dtype": "float64",
    "seed": 1234,
    "result_folder": os.path.join("outputs", "test__cvtsp_real529_n100"),
    "log_level": "INFO",
}


##########################################################################################
# main

def main():
    args = _build_parser().parse_args()
    _apply_overrides(args)

    if DEBUG_MODE:
        _set_debug_mode()

    _print_config()

    tester = Tester(
        env_params=env_params,
        model_params=model_params,
        tester_params=tester_params,
    )
    tester.run()


def _set_debug_mode():
    global tester_params
    tester_params["test_episodes"] = 10
    tester_params["test_batch_size"] = 4


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test the online-random CVTSP model on random instances or real 529 instances."
    )
    parser.add_argument(
        "--mode",
        choices=["random", "real529"],
        help="random: evaluate online random instances; real529: evaluate real dataset instances.",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "val", "test"],
        help="split used when --mode real529.",
    )
    parser.add_argument("--episodes", type=int, help="number of random instances to evaluate in random mode.")
    parser.add_argument("--batch-size", type=int, help="test batch size.")
    parser.add_argument("--checkpoint-path", help="direct checkpoint path.")
    parser.add_argument("--checkpoint-dir", help="checkpoint directory containing checkpoint-<epoch>.pt.")
    parser.add_argument("--checkpoint-epoch", type=int, help="checkpoint epoch when using --checkpoint-dir.")
    parser.add_argument("--output-dir", help="override result folder.")
    parser.add_argument("--decode-type", choices=["greedy", "sample"], help="decoder type.")
    parser.add_argument("--augmentation", action="store_true", help="enable 8-fold test augmentation.")
    parser.add_argument("--aug-factor", type=int, help="number of augmented views to evaluate (1-8).")
    parser.add_argument("--seed", type=int, help="random seed.")
    parser.add_argument("--cpu", action="store_true", help="force CPU evaluation.")
    return parser


def _apply_overrides(args: argparse.Namespace) -> None:
    global tester_params

    if args.mode:
        tester_params["test_mode"] = args.mode
    if args.split:
        tester_params["real_split"] = args.split
    if args.episodes is not None:
        tester_params["test_episodes"] = args.episodes
    if args.batch_size is not None:
        tester_params["test_batch_size"] = args.batch_size
    if args.decode_type:
        tester_params["decode_type"] = args.decode_type
    if args.augmentation:
        tester_params["augmentation_enable"] = True
    if args.aug_factor is not None:
        tester_params["aug_factor"] = args.aug_factor
    if args.seed is not None:
        tester_params["seed"] = args.seed
    if args.cpu:
        tester_params["use_cuda"] = False

    if args.checkpoint_path:
        tester_params["model_load"] = {"checkpoint_path": args.checkpoint_path}
    else:
        if args.checkpoint_dir:
            tester_params["model_load"]["path"] = args.checkpoint_dir
        if args.checkpoint_epoch is not None:
            tester_params["model_load"]["epoch"] = args.checkpoint_epoch

    if args.output_dir:
        tester_params["result_folder"] = args.output_dir
    elif tester_params["test_mode"] == "random":
        tester_params["result_folder"] = os.path.join("outputs", "test__cvtsp_random_n100")
    else:
        split_name = tester_params.get("real_split", "all")
        tester_params["result_folder"] = os.path.join("outputs", f"test__cvtsp_real529_{split_name}_n100")


def _print_config():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    logger = logging.getLogger("root")
    logger.info("DEBUG_MODE: %s", DEBUG_MODE)
    logger.info("USE_CUDA: %s, CUDA_DEVICE_NUM: %s", USE_CUDA, CUDA_DEVICE_NUM)
    logger.info("env_params=%s", env_params)
    logger.info("model_params=%s", model_params)
    logger.info("tester_params=%s", tester_params)


##########################################################################################

if __name__ == "__main__":
    main()
