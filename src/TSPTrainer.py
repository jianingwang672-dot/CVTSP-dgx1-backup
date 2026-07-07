from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from src.CVXPYLayerSolver import (
    solve_fixed_sequence_objective_numpy,
    solve_ordered_targets_objective_torch,
    solve_ordered_targets_torch,
)
from src.CVPSolver import SPSolution, SolverConfig, solve
from src.RewardWorkerPool import RewardWorkerPool
from src.TSPEnv import RolloutResult, TSPEnv, rollout_batch
from src.TSPModel import ModelConfig, TSPModel
from src.TSPTester import RewardCache, evaluate_dataset
from src.TSProblemDef import RandomCVTSPGenerator, iter_grouped_batches, load_dataset
from src.TSPUtils import (
    AppConfig,
    load_checkpoint,
    load_config,
    save_checkpoint,
    save_config_snapshot,
    set_seed,
    setup_logger,
)


def _resolve_max_pomo_size(env_params: dict[str, Any], model_params: dict[str, Any], default: int) -> int:
    pomo_divisor = env_params.get("pomo_divisor")
    pomo_size_limit = int(env_params.get("pomo_size", model_params.get("max_pomo_size", default)))
    if pomo_divisor is None:
        return pomo_size_limit
    max_problem_size = int(env_params.get("max_problem_size", env_params.get("problem_size", pomo_size_limit)))
    return min(pomo_size_limit, max(1, max_problem_size // int(pomo_divisor)))


def _solve_reward_task(payload: tuple[Any, ...]) -> tuple[str, tuple[int, ...], SPSolution]:
    instance, sequence, solver_config = payload
    solution = solve(instance, sequence, solver_config)
    return instance.instance_id, tuple(sequence), solution


def _solve_cvxpylayer_reward_chunk_task(payload: tuple[Any, ...]) -> list[tuple[float, bool, str, float]]:
    route_items, solver_args, dtype_name = payload
    if not route_items:
        return []

    route_dtype = torch.float64 if str(dtype_name).lower() == "float64" else torch.float32
    route_device = torch.device("cpu")
    chunk_start = time.perf_counter()
    try:
        ordered_targets = torch.stack(
            [
                torch.as_tensor(
                    instance.targets[sequence],
                    dtype=route_dtype,
                    device=route_device,
                )
                for instance, sequence in route_items
            ],
            dim=0,
        )
        depots = torch.stack(
            [
                torch.as_tensor(instance.depot, dtype=route_dtype, device=route_device)
                for instance, _ in route_items
            ],
            dim=0,
        )
        carrier_speeds = torch.as_tensor(
            [float(instance.carrier_speed) for instance, _ in route_items],
            dtype=route_dtype,
            device=route_device,
        )
        uav_speeds = torch.as_tensor(
            [float(instance.uav_speed) for instance, _ in route_items],
            dtype=route_dtype,
            device=route_device,
        )
        endurances = torch.as_tensor(
            [float(instance.endurance) for instance, _ in route_items],
            dtype=route_dtype,
            device=route_device,
        )
        with torch.no_grad():
            objectives = solve_ordered_targets_objective_torch(
                depot=depots,
                ordered_targets=ordered_targets,
                carrier_speed=carrier_speeds,
                uav_speed=uav_speeds,
                endurance=endurances,
                solver_args=solver_args,
                dtype=route_dtype,
            ).detach().cpu()
        if not torch.isfinite(objectives).all():
            raise RuntimeError("batched cvxpylayer objective contains non-finite values")
        solve_time = time.perf_counter() - chunk_start
        per_route_time = solve_time / max(len(route_items), 1)
        return [
            (float(objective), True, "CVXPYLAYER_CHUNK_OBJECTIVE", per_route_time)
            for objective in objectives.tolist()
        ]
    except Exception:
        results: list[tuple[float, bool, str, float]] = []
        for instance, sequence in route_items:
            single_start = time.perf_counter()
            try:
                result = solve_fixed_sequence_objective_numpy(
                    depot=instance.depot,
                    targets=instance.targets,
                    sequence=list(sequence),
                    carrier_speed=instance.carrier_speed,
                    uav_speed=instance.uav_speed,
                    endurance=instance.endurance,
                    solver_args=solver_args,
                    dtype=dtype_name,
                )
                results.append(
                    (
                        float(result["obj"]),
                        True,
                        str(result.get("status", "CVXPYLAYER_OBJECTIVE")),
                        time.perf_counter() - single_start,
                    )
                )
            except Exception as exc:  # pragma: no cover - worker safety
                results.append(
                    (
                        float("inf"),
                        False,
                        f"ERROR:{type(exc).__name__}",
                        time.perf_counter() - single_start,
                    )
                )
        return results


def _format_progress_bar(completed: int, total: int, width: int = 24) -> str:
    if total <= 0:
        total = 1
    ratio = min(max(completed / total, 0.0), 1.0)
    filled = int(ratio * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class TSPTrainer:
    def __init__(self, config: AppConfig):
        self.config = config
        self.output_dir = Path(config.runtime.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        save_config_snapshot(config, self.output_dir / "resolved_config.yaml")

        self.device = torch.device(config.runtime.device)
        self.model = TSPModel(config.model).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay,
        )

        self.start_epoch = 1
        self.best_val = float("inf")
        if config.runtime.checkpoint_path and Path(config.runtime.checkpoint_path).exists():
            checkpoint = load_checkpoint(
                config.runtime.checkpoint_path,
                self.model,
                optimizer=self.optimizer,
                map_location=self.device,
            )
            self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
            self.best_val = float(checkpoint.get("best_metric", float("inf")))
            logger.info("Resumed training from epoch {}", self.start_epoch)

        self.val_instances = load_dataset(
            config.data.dataset_dir,
            "val",
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            split_seed=config.data.split_seed,
        )
        self.train_instances = load_dataset(
            config.data.dataset_dir,
            "train",
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            split_seed=config.data.split_seed,
        )
        self.test_instances = load_dataset(
            config.data.dataset_dir,
            "test",
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            split_seed=config.data.split_seed,
        )
        self.cache = RewardCache()
        self.metrics_path = self.output_dir / "metrics.csv"

        split_manifest = {
            "train_count": len(self.train_instances),
            "val_count": len(self.val_instances),
            "test_count": len(self.test_instances),
            "train_instances": [instance.instance_id for instance in self.train_instances],
            "val_instances": [instance.instance_id for instance in self.val_instances],
            "test_instances": [instance.instance_id for instance in self.test_instances],
        }
        (self.output_dir / "split_manifest.json").write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    def _compute_batch_rewards_serial(self, instances, rollout_result):
        sequence_lists = rollout_result.to_sequence_lists()
        batch_size = len(sequence_lists)
        pomo_size = len(sequence_lists[0]) if sequence_lists else 0
        rewards = torch.empty((batch_size, pomo_size), dtype=torch.float32, device=self.device)
        feasible_count = 0
        solver_calls = 0

        for batch_index, instance in enumerate(instances):
            for pomo_index, sequence in enumerate(sequence_lists[batch_index]):
                solution, from_cache = self.cache.get_or_solve(instance, sequence, self.config.solver)
                if not from_cache:
                    solver_calls += 1
                if solution.success:
                    rewards[batch_index, pomo_index] = -float(solution.objective)
                    feasible_count += 1
                else:
                    rewards[batch_index, pomo_index] = float(self.config.train.penalty_reward)
        feasible_ratio = feasible_count / max(batch_size * max(pomo_size, 1), 1)
        return rewards, feasible_ratio, solver_calls

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_objective = 0.0
        total_feasible = 0.0
        total_batches = 0
        total_solver_calls = 0
        max_batches = self.config.train.max_batches_per_epoch

        for batch_instances in iter_grouped_batches(
            list(self.train_instances),
            batch_size=self.config.train.batch_size,
            shuffle=True,
            seed=self.config.runtime.seed + epoch,
        ):
            if max_batches is not None and total_batches >= max_batches:
                break
            rollout_result = rollout_batch(
                model=self.model,
                instances=batch_instances,
                device=self.device,
                decode_type="sample",
                use_pomo_start=True,
            )
            rewards, feasible_ratio, solver_calls = self._compute_batch_rewards(batch_instances, rollout_result)

            advantage = rewards - rewards.mean(dim=1, keepdim=True)
            loss = -(advantage * rollout_result.log_probs).mean()

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.train.grad_clip)
            self.optimizer.step()

            best_reward = rewards.max(dim=1).values
            total_loss += float(loss.item())
            total_objective += float((-best_reward).mean().item())
            total_feasible += feasible_ratio
            total_batches += 1
            total_solver_calls += solver_calls

        return {
            "epoch": float(epoch),
            "train_loss": total_loss / max(total_batches, 1),
            "train_objective": total_objective / max(total_batches, 1),
            "train_feasible_ratio": total_feasible / max(total_batches, 1),
            "solver_calls": float(total_solver_calls),
            "cache_hits": float(self.cache.hits),
            "cache_misses": float(self.cache.misses),
        }

    def _append_metrics_row(self, row: dict[str, Any]) -> None:
        self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.metrics_path.exists()
        with self.metrics_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def run(self) -> dict[str, Any]:
        summary: dict[str, Any] = {}

        for epoch in range(self.start_epoch, self.config.train.epochs + 1):
            train_metrics = self._train_one_epoch(epoch)
            log_payload = dict(train_metrics)

            if self.val_instances and epoch % self.config.train.validate_every == 0:
                self.model.eval()
                with torch.no_grad():
                    _, val_summary = evaluate_dataset(
                        self.val_instances,
                        config=self.config,
                        model=self.model,
                        device=self.device,
                        cache=self.cache,
                    )
                log_payload.update({f"val_{key}": value for key, value in val_summary.items()})
                current_val = float(val_summary["avg_objective"])
                if current_val < self.best_val:
                    self.best_val = current_val
                    save_checkpoint(
                        self.output_dir / "best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        best_metric=self.best_val,
                        metrics=log_payload,
                    )
                    logger.info("Saved best checkpoint at epoch {} with avg objective {:.4f}", epoch, self.best_val)
            elif not self.val_instances:
                current_metric = float(train_metrics["train_objective"])
                if current_metric < self.best_val:
                    self.best_val = current_metric
                    save_checkpoint(
                        self.output_dir / "best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        best_metric=self.best_val,
                        metrics=log_payload,
                    )
                    logger.info("Saved best checkpoint at epoch {} with train objective {:.4f}", epoch, self.best_val)

            if epoch % self.config.train.checkpoint_every == 0:
                save_checkpoint(
                    self.output_dir / "last.pt",
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    best_metric=self.best_val,
                    metrics=log_payload,
                )

            self._append_metrics_row(log_payload)
            logger.info(
                "Epoch {} | train_loss={:.4f} train_obj={:.4f} train_feas={:.3f}",
                epoch,
                log_payload["train_loss"],
                log_payload["train_objective"],
                log_payload["train_feasible_ratio"],
            )
            if "val_avg_objective" in log_payload:
                logger.info(
                    "Epoch {} | val_obj={:.4f} val_feas={:.3f}",
                    epoch,
                    log_payload["val_avg_objective"],
                    log_payload["val_feasible_ratio"],
                )
            summary = log_payload

        return {
            "output_dir": str(self.output_dir),
            "best_val_objective": self.best_val,
            "last_metrics": summary,
            "config": asdict(self.config),
        }


class OnlineTSPTrainer:
    def __init__(
        self,
        env_params: dict[str, Any],
        model_params: dict[str, Any],
        optimizer_params: dict[str, Any],
        trainer_params: dict[str, Any],
    ):
        self.env_params = dict(env_params)
        self.model_params = dict(model_params)
        self.optimizer_params = dict(optimizer_params)
        self.trainer_params = dict(trainer_params)

        seed = int(self.trainer_params.get("seed", 1234))
        set_seed(seed)
        self._problem_size_rng = torch.Generator().manual_seed(seed)

        self.output_dir = Path(self.trainer_params.get("result_folder", "outputs/train_n100"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_logger(self.output_dir, self.trainer_params.get("log_level", "INFO"))

        use_cuda = bool(self.trainer_params.get("use_cuda", True)) and torch.cuda.is_available()
        cuda_device_num = int(self.trainer_params.get("cuda_device_num", 0))
        self.device = torch.device("cuda", cuda_device_num) if use_cuda else torch.device("cpu")
        if use_cuda:
            torch.cuda.set_device(cuda_device_num)

        default_model = ModelConfig()
        self.model_config = ModelConfig(
            node_feature_dim=int(self.model_params.get("node_feature_dim", default_model.node_feature_dim)),
            embedding_dim=int(self.model_params.get("embedding_dim", default_model.embedding_dim)),
            encoder_layer_num=int(self.model_params.get("encoder_layer_num", default_model.encoder_layer_num)),
            qkv_dim=int(self.model_params.get("qkv_dim", default_model.qkv_dim)),
            head_num=int(self.model_params.get("head_num", default_model.head_num)),
            logit_clipping=float(self.model_params.get("logit_clipping", default_model.logit_clipping)),
            ff_hidden_dim=int(self.model_params.get("ff_hidden_dim", default_model.ff_hidden_dim)),
            max_pomo_size=_resolve_max_pomo_size(self.env_params, self.model_params, default_model.max_pomo_size),
            pomo_divisor=(
                int(self.env_params["pomo_divisor"])
                if self.env_params.get("pomo_divisor") is not None
                else default_model.pomo_divisor
            ),
            start_node_strategy=str(
                self.env_params.get("start_node_strategy", default_model.start_node_strategy)
            ),
        )
        self.model = TSPModel(self.model_config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            **self.optimizer_params.get("optimizer", {"lr": 1e-4, "weight_decay": 1e-6}),
        )
        scheduler_params = self.optimizer_params.get("scheduler", {})
        self.scheduler = Scheduler(self.optimizer, **scheduler_params) if scheduler_params else None

        self.solver_config = SolverConfig(
            solver_backend=str(self.trainer_params.get("solver_backend", "gurobi")),
            gurobi_time_limit=self.trainer_params.get("gurobi_time_limit"),
            gurobi_threads=int(self.trainer_params.get("gurobi_threads", 16)),
            output_flag=int(self.trainer_params.get("output_flag", 0)),
            cvxpylayer_solver_args=dict(self.trainer_params.get("cvxpylayer_solver_args", {}) or {}),
            cvxpylayer_dtype=str(self.trainer_params.get("cvxpylayer_dtype", "float64")),
        )
        self.cvxpylayer_objective_loss_enable = bool(
            self.trainer_params.get("cvxpylayer_objective_loss_enable", False)
        )
        self.cvxpylayer_hard_objective_loss_enable = bool(
            self.trainer_params.get("cvxpylayer_hard_objective_loss_enable", False)
        )
        self.cvxpylayer_rebar_loss_enable = bool(
            self.trainer_params.get("cvxpylayer_rebar_loss_enable", False)
        )
        self.cvxpylayer_aux_enable = bool(self.trainer_params.get("cvxpylayer_aux_enable", False))
        self.cvxpylayer_aux_weight = float(self.trainer_params.get("cvxpylayer_aux_weight", 0.0))
        if (
            self.cvxpylayer_objective_loss_enable
            or self.cvxpylayer_hard_objective_loss_enable
            or (self.cvxpylayer_aux_enable and self.cvxpylayer_aux_weight > 0)
        ):
            raise ValueError(
                "Sinkhorn/direct cvxpylayer route losses have been removed. "
                "Use cvxpylayer_rebar_loss_enable=True, or disable cvxpylayer "
                "route loss for plain POMO policy-gradient training."
            )
        self.cvxpylayer_route_candidates = max(
            1,
            int(
                self.trainer_params.get(
                    "cvxpylayer_route_candidates",
                    self.trainer_params.get("cvxpylayer_aux_candidates", 1),
                )
            ),
        )
        self.cvxpylayer_route_max_instances_per_batch = int(
            self.trainer_params.get(
                "cvxpylayer_route_max_instances_per_batch",
                self.trainer_params.get("cvxpylayer_aux_max_instances_per_batch", 0),
            )
        )
        self.cvxpylayer_route_selection = str(
            self.trainer_params.get(
                "cvxpylayer_route_selection",
                self.trainer_params.get("cvxpylayer_aux_selection", "best"),
            )
        ).lower()
        self.cvxpylayer_route_device = str(
            self.trainer_params.get(
                "cvxpylayer_route_device",
                self.trainer_params.get("cvxpylayer_aux_device", "cpu"),
            )
        ).lower()
        self.cvxpylayer_route_normalize_by_size = bool(
            self.trainer_params.get(
                "cvxpylayer_route_normalize_by_size",
                self.trainer_params.get("cvxpylayer_aux_normalize_by_size", False),
            )
        )
        self.cvxpylayer_rebar_eta = float(self.trainer_params.get("cvxpylayer_rebar_eta", 1.0))
        self.cvxpylayer_rebar_temperature = float(
            self.trainer_params.get("cvxpylayer_rebar_temperature", 1.0)
        )
        self.cvxpylayer_rebar_conditional_bias = float(
            self.trainer_params.get("cvxpylayer_rebar_conditional_bias", 8.0)
        )
        self.cvxpylayer_rebar_use_gumbel = bool(
            self.trainer_params.get("cvxpylayer_rebar_use_gumbel", True)
        )
        self.penalty_reward = float(self.trainer_params.get("penalty_reward", -1e6))
        self.grad_clip = float(self.trainer_params.get("grad_clip", 1.0))
        self.train_batch_size = int(self.trainer_params.get("train_batch_size", 64))
        self.train_episodes = int(self.trainer_params.get("train_episodes", 100000))
        self.epochs = int(self.trainer_params.get("epochs", 1000))
        self.resume_extra_epochs = bool(self.trainer_params.get("resume_extra_epochs", False))
        self.checkpoint_interval = int(self.trainer_params.get("checkpoint_interval", 100))
        self.log_first_batch_count = int(self.trainer_params.get("log_first_batch_count", 10))
        self.progress_log_percent = float(self.trainer_params.get("progress_log_percent", 1.0))
        self.progress_bar_width = int(self.trainer_params.get("progress_bar_width", 24))
        self.best_score = float("inf")
         # ===== 这里开始是新增的续训逻辑 =====
        self.start_epoch = 1
        self.best_score = float("inf")
        checkpoint_path = self.trainer_params.get("checkpoint_path")

        if checkpoint_path and Path(checkpoint_path).exists():
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

            if "model_state_dict" in checkpoint:
                self.model.load_state_dict(checkpoint["model_state_dict"])
            else:
                raise KeyError(f"checkpoint missing 'model_state_dict': {checkpoint_path}")

            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            # 如果以后你保存了 scheduler，也可以自动恢复
            if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            self.start_epoch = int(checkpoint.get("epoch", 0)) + 1
            self.best_score = float(checkpoint.get("best_metric", float("inf")))

            logger.info("Resumed training from epoch {}", self.start_epoch)
            if self.resume_extra_epochs:
                requested_extra_epochs = self.epochs
                self.epochs = self.start_epoch + requested_extra_epochs - 1
                logger.info(
                    "Training {} additional epoch(s), ending at epoch {}",
                    requested_extra_epochs,
                    self.epochs,
                )
        # ===== 新增逻辑结束 =====
        self.cache = RewardCache()
        self.metrics_path = self.output_dir / "metrics.csv"
        self.reward_parallel_workers = int(self.trainer_params.get("reward_parallel_workers", 0))
        self.parallel_solver_threads = int(self.trainer_params.get("parallel_solver_threads", 1))
        self.reward_parallel_chunksize = int(self.trainer_params.get("reward_parallel_chunksize", 1))
        self.cvxpylayer_reward_batch_size = int(
            self.trainer_params.get("cvxpylayer_reward_batch_size", 128)
        )
        self.reward_backend = str(
            self.trainer_params.get(
                "reward_backend",
                "executor" if self.reward_parallel_workers > 0 else "serial",
            )
        ).lower()
        if self.reward_backend in {"cvxpylayer_batch", "cvxpylayer_chunk_pool"} and str(
            self.solver_config.solver_backend
        ).lower() not in {
            "cvxpylayer",
            "cvxpy_layer",
            "cvxpy",
        }:
            raise ValueError(
                "cvxpylayer reward backends require solver_backend='cvxpylayer'"
            )
        self.reward_pool_queue_size = int(
            self.trainer_params.get("reward_pool_queue_size", max(self.reward_parallel_workers * 4, 1))
        )
        self.reward_pool_poll_timeout = float(self.trainer_params.get("reward_pool_poll_timeout", 5.0))
        self._reward_executor: ProcessPoolExecutor | None = None
        self._reward_worker_pool: RewardWorkerPool | None = None
        if self.reward_backend in {"executor", "cvxpylayer_chunk_pool"} and self.reward_parallel_workers > 0:
            self._reward_executor = ProcessPoolExecutor(
                max_workers=self.reward_parallel_workers,
                mp_context=mp.get_context("spawn"),
            )
        elif self.reward_backend == "persistent_pool" and self.reward_parallel_workers > 0:
            self._reward_worker_pool = RewardWorkerPool(
                worker_count=self.reward_parallel_workers,
                env_output_flag=self.solver_config.output_flag,
                queue_size=self.reward_pool_queue_size,
                poll_timeout=self.reward_pool_poll_timeout,
            )

        min_problem_size = int(self.env_params.get("min_problem_size", self.env_params.get("problem_size", 100)))
        max_problem_size = int(self.env_params.get("max_problem_size", self.env_params.get("problem_size", 100)))
        self.env = TSPEnv(
            problem_size=self.env_params.get("problem_size"),
            pomo_size=int(self.env_params.get("pomo_size", 100)),
            pomo_divisor=(
                int(self.env_params["pomo_divisor"])
                if self.env_params.get("pomo_divisor") is not None
                else None
            ),
            use_pomo_start=True,
            start_node_strategy=str(self.env_params.get("start_node_strategy", "spread")),
            dataset_dir=self.env_params.get("dataset_dir", "instance/Data"),
            min_problem_size=min_problem_size,
            max_problem_size=max_problem_size,
            problem_sizes=self.env_params.get("problem_sizes"),
            seed=seed,
            online_random=True,
        )
        if self.env.generator is None:
            raise RuntimeError("online random generator was not initialized")
        self.generator = self.env.generator

        snapshot = {
            "env_params": self.env_params,
            "model_params": self.model_params,
            "optimizer_params": self.optimizer_params,
            "trainer_params": self.trainer_params,
            "generator_stats": self.generator.stats.to_dict(),
        }
        (self.output_dir / "params.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
        

    def _sample_problem_size(self) -> int:
        problem_sizes = self.env_params.get("problem_sizes")
        if problem_sizes:
            choices = [int(size) for size in problem_sizes]
            index = int(torch.randint(0, len(choices), (1,), generator=self._problem_size_rng).item())
            return choices[index]
        low = int(self.env_params.get("min_problem_size", self.env_params.get("problem_size", 100)))
        high = int(self.env_params.get("max_problem_size", self.env_params.get("problem_size", low)))
        if low == high:
            return low
        return int(torch.randint(low, high + 1, (1,), generator=self._problem_size_rng).item())

    def _compute_batch_rewards_serial(self, instances, rollout_result):
        sequence_lists = rollout_result.to_sequence_lists()
        batch_size = len(sequence_lists)
        pomo_size = len(sequence_lists[0]) if sequence_lists else 0
        rewards = torch.empty((batch_size, pomo_size), dtype=torch.float32, device=self.device)
        feasible_count = 0
        solver_calls = 0

        for batch_index, instance in enumerate(instances):
            for pomo_index, sequence in enumerate(sequence_lists[batch_index]):
                solution, from_cache = self.cache.get_or_solve(instance, sequence, self.solver_config)
                if not from_cache:
                    solver_calls += 1
                if solution.success:
                    rewards[batch_index, pomo_index] = -float(solution.objective)
                    feasible_count += 1
                else:
                    rewards[batch_index, pomo_index] = self.penalty_reward
        feasible_ratio = feasible_count / max(batch_size * max(pomo_size, 1), 1)
        return rewards, feasible_ratio, solver_calls

    def _prepare_pending_rewards(self, instances, rollout_result):
        sequence_lists = rollout_result.to_sequence_lists()
        batch_size = len(sequence_lists)
        pomo_size = len(sequence_lists[0]) if sequence_lists else 0
        rewards = torch.empty((batch_size, pomo_size), dtype=torch.float32, device=self.device)
        feasible_count = 0
        pending: dict[tuple[str, tuple[int, ...]], dict[str, Any]] = {}

        for batch_index, instance in enumerate(instances):
            for pomo_index, sequence in enumerate(sequence_lists[batch_index]):
                cached = self.cache.get_cached(instance, sequence)
                if cached is not None:
                    if cached.success:
                        rewards[batch_index, pomo_index] = -float(cached.objective)
                        feasible_count += 1
                    else:
                        rewards[batch_index, pomo_index] = self.penalty_reward
                    continue

                key = RewardCache.make_key(instance.instance_id, sequence)
                if key not in pending:
                    pending[key] = {
                        "instance": instance,
                        "sequence": list(sequence),
                        "positions": [],
                    }
                pending[key]["positions"].append((batch_index, pomo_index))

        return rewards, feasible_count, pending, batch_size, pomo_size

    def _compute_batch_rewards_parallel(self, instances, rollout_result):
        if self._reward_executor is None:
            return self._compute_batch_rewards_serial(instances, rollout_result)

        rewards, feasible_count, pending, batch_size, pomo_size = self._prepare_pending_rewards(instances, rollout_result)

        solver_calls = len(pending)
        if pending:
            parallel_solver_config = SolverConfig(
                solver_backend=self.solver_config.solver_backend,
                gurobi_time_limit=self.solver_config.gurobi_time_limit,
                gurobi_threads=self.parallel_solver_threads,
                output_flag=self.solver_config.output_flag,
                cvxpylayer_solver_args=dict(self.solver_config.cvxpylayer_solver_args),
                cvxpylayer_dtype=self.solver_config.cvxpylayer_dtype,
            )
            tasks = [
                (entry["instance"], entry["sequence"], parallel_solver_config)
                for entry in pending.values()
            ]
            solutions = self._reward_executor.map(
                _solve_reward_task,
                tasks,
                chunksize=max(self.reward_parallel_chunksize, 1),
            )
            for entry, (_, _, solution) in zip(pending.values(), solutions):
                instance = entry["instance"]
                sequence = entry["sequence"]
                self.cache.put(instance, sequence, solution)
                for batch_index, pomo_index in entry["positions"]:
                    if solution.success:
                        rewards[batch_index, pomo_index] = -float(solution.objective)
                        feasible_count += 1
                    else:
                        rewards[batch_index, pomo_index] = self.penalty_reward

        feasible_ratio = feasible_count / max(batch_size * max(pomo_size, 1), 1)
        return rewards, feasible_ratio, solver_calls

    def _compute_batch_rewards_persistent_pool(self, instances, rollout_result):
        if self._reward_worker_pool is None:
            return self._compute_batch_rewards_serial(instances, rollout_result)

        rewards, feasible_count, pending, batch_size, pomo_size = self._prepare_pending_rewards(instances, rollout_result)
        solver_calls = len(pending)
        if pending:
            parallel_solver_config = SolverConfig(
                solver_backend=self.solver_config.solver_backend,
                gurobi_time_limit=self.solver_config.gurobi_time_limit,
                gurobi_threads=self.parallel_solver_threads,
                output_flag=self.solver_config.output_flag,
                cvxpylayer_solver_args=dict(self.solver_config.cvxpylayer_solver_args),
                cvxpylayer_dtype=self.solver_config.cvxpylayer_dtype,
            )
            tasks = [
                (entry["instance"], entry["sequence"], parallel_solver_config)
                for entry in pending.values()
            ]
            solutions = self._reward_worker_pool.map(tasks)
            for entry, (_, _, solution) in zip(pending.values(), solutions):
                instance = entry["instance"]
                sequence = entry["sequence"]
                self.cache.put(instance, sequence, solution)
                for batch_index, pomo_index in entry["positions"]:
                    if solution.success:
                        rewards[batch_index, pomo_index] = -float(solution.objective)
                        feasible_count += 1
                    else:
                        rewards[batch_index, pomo_index] = self.penalty_reward

        feasible_ratio = feasible_count / max(batch_size * max(pomo_size, 1), 1)
        return rewards, feasible_ratio, solver_calls

    def _compute_batch_rewards_cvxpylayer_chunk_pool(self, instances, rollout_result):
        if self._reward_executor is None:
            return self._compute_batch_rewards_cvxpylayer_batch(instances, rollout_result)

        rewards, feasible_count, pending, batch_size, pomo_size = self._prepare_pending_rewards(instances, rollout_result)
        solver_calls = len(pending)
        if not pending:
            feasible_ratio = feasible_count / max(batch_size * max(pomo_size, 1), 1)
            return rewards, feasible_ratio, solver_calls

        chunk_size = self.cvxpylayer_reward_batch_size
        if chunk_size <= 0:
            chunk_size = len(pending)

        grouped_entries: dict[int, list[dict[str, Any]]] = {}
        for entry in pending.values():
            grouped_entries.setdefault(int(entry["instance"].J), []).append(entry)

        entry_chunks: list[list[dict[str, Any]]] = []
        tasks = []
        for entries in grouped_entries.values():
            for start in range(0, len(entries), chunk_size):
                chunk = entries[start : start + chunk_size]
                entry_chunks.append(chunk)
                route_items = [
                    (entry["instance"], tuple(entry["sequence"]))
                    for entry in chunk
                ]
                tasks.append(
                    (
                        route_items,
                        dict(self.solver_config.cvxpylayer_solver_args),
                        self.solver_config.cvxpylayer_dtype,
                    )
                )

        mapped_results = self._reward_executor.map(
            _solve_cvxpylayer_reward_chunk_task,
            tasks,
            chunksize=max(self.reward_parallel_chunksize, 1),
        )
        for chunk, chunk_results in zip(entry_chunks, mapped_results):
            if len(chunk_results) != len(chunk):
                logger.warning(
                    "cvxpylayer chunk worker returned {} results for {} routes",
                    len(chunk_results),
                    len(chunk),
                )
            for entry, result in zip(chunk, chunk_results):
                objective, success, status, solve_time = result
                solution = SPSolution(
                    objective=float(objective),
                    makespan=float(objective),
                    success=bool(success),
                    status=str(status),
                    solve_time=float(solve_time),
                    sequence=list(entry["sequence"]),
                    raw_debug={
                        "backend": "cvxpylayer_chunk_pool",
                        "chunk_size": len(chunk),
                        "solver_args": dict(self.solver_config.cvxpylayer_solver_args),
                    },
                )
                instance = entry["instance"]
                sequence = entry["sequence"]
                self.cache.put(instance, sequence, solution)
                for batch_index, pomo_index in entry["positions"]:
                    if solution.success:
                        rewards[batch_index, pomo_index] = -float(solution.objective)
                        feasible_count += 1
                    else:
                        rewards[batch_index, pomo_index] = self.penalty_reward

        feasible_ratio = feasible_count / max(batch_size * max(pomo_size, 1), 1)
        return rewards, feasible_ratio, solver_calls

    def _compute_batch_rewards_cvxpylayer_batch(self, instances, rollout_result):
        rewards, feasible_count, pending, batch_size, pomo_size = self._prepare_pending_rewards(instances, rollout_result)
        solver_calls = len(pending)
        if not pending:
            feasible_ratio = feasible_count / max(batch_size * max(pomo_size, 1), 1)
            return rewards, feasible_ratio, solver_calls

        route_dtype = torch.float64 if self.solver_config.cvxpylayer_dtype == "float64" else torch.float32
        route_device = torch.device("cpu")
        chunk_size = self.cvxpylayer_reward_batch_size
        if chunk_size <= 0:
            chunk_size = len(pending)

        grouped_entries: dict[int, list[dict[str, Any]]] = {}
        for entry in pending.values():
            grouped_entries.setdefault(int(entry["instance"].J), []).append(entry)

        fallback_solver_config = SolverConfig(
            solver_backend=self.solver_config.solver_backend,
            gurobi_time_limit=self.solver_config.gurobi_time_limit,
            gurobi_threads=self.parallel_solver_threads,
            output_flag=self.solver_config.output_flag,
            cvxpylayer_solver_args=dict(self.solver_config.cvxpylayer_solver_args),
            cvxpylayer_dtype=self.solver_config.cvxpylayer_dtype,
        )

        def write_solution(entry: dict[str, Any], solution: SPSolution) -> None:
            nonlocal feasible_count
            instance = entry["instance"]
            sequence = entry["sequence"]
            self.cache.put(instance, sequence, solution)
            for batch_index, pomo_index in entry["positions"]:
                if solution.success:
                    rewards[batch_index, pomo_index] = -float(solution.objective)
                    feasible_count += 1
                else:
                    rewards[batch_index, pomo_index] = self.penalty_reward

        for problem_size, entries in grouped_entries.items():
            for start in range(0, len(entries), chunk_size):
                chunk = entries[start : start + chunk_size]
                chunk_start = time.perf_counter()
                try:
                    ordered_targets = torch.stack(
                        [
                            torch.as_tensor(
                                entry["instance"].targets[entry["sequence"]],
                                dtype=route_dtype,
                                device=route_device,
                            )
                            for entry in chunk
                        ],
                        dim=0,
                    )
                    depots = torch.stack(
                        [
                            torch.as_tensor(entry["instance"].depot, dtype=route_dtype, device=route_device)
                            for entry in chunk
                        ],
                        dim=0,
                    )
                    carrier_speeds = torch.as_tensor(
                        [float(entry["instance"].carrier_speed) for entry in chunk],
                        dtype=route_dtype,
                        device=route_device,
                    )
                    uav_speeds = torch.as_tensor(
                        [float(entry["instance"].uav_speed) for entry in chunk],
                        dtype=route_dtype,
                        device=route_device,
                    )
                    endurances = torch.as_tensor(
                        [float(entry["instance"].endurance) for entry in chunk],
                        dtype=route_dtype,
                        device=route_device,
                    )
                    with torch.no_grad():
                        objectives = solve_ordered_targets_objective_torch(
                            depot=depots,
                            ordered_targets=ordered_targets,
                            carrier_speed=carrier_speeds,
                            uav_speed=uav_speeds,
                            endurance=endurances,
                            solver_args=self.solver_config.cvxpylayer_solver_args,
                            dtype=route_dtype,
                        ).detach().cpu()
                    if not torch.isfinite(objectives).all():
                        raise RuntimeError("batched cvxpylayer objective contains non-finite values")
                    solve_time = time.perf_counter() - chunk_start
                    per_route_time = solve_time / max(len(chunk), 1)
                    for entry, objective in zip(chunk, objectives.tolist()):
                        solution = SPSolution(
                            objective=float(objective),
                            makespan=float(objective),
                            success=True,
                            status="CVXPYLAYER_BATCH",
                            solve_time=per_route_time,
                            sequence=list(entry["sequence"]),
                            raw_debug={
                                "problem_size": problem_size,
                                "batch_size": len(chunk),
                                "solver_args": dict(self.solver_config.cvxpylayer_solver_args),
                            },
                        )
                        write_solution(entry, solution)
                except Exception as exc:
                    logger.warning(
                        "Batched cvxpylayer reward failed for J={} chunk={} ({}); falling back to single solves",
                        problem_size,
                        len(chunk),
                        type(exc).__name__,
                    )
                    for entry in chunk:
                        solution = solve(entry["instance"], entry["sequence"], fallback_solver_config)
                        write_solution(entry, solution)

        feasible_ratio = feasible_count / max(batch_size * max(pomo_size, 1), 1)
        return rewards, feasible_ratio, solver_calls

    def _compute_batch_rewards(self, instances, rollout_result):
        if self.reward_backend == "cvxpylayer_chunk_pool":
            return self._compute_batch_rewards_cvxpylayer_chunk_pool(instances, rollout_result)
        if self.reward_backend == "cvxpylayer_batch":
            return self._compute_batch_rewards_cvxpylayer_batch(instances, rollout_result)
        if self.reward_backend == "persistent_pool" and self.reward_parallel_workers > 0:
            return self._compute_batch_rewards_persistent_pool(instances, rollout_result)
        if self.reward_backend == "executor" and self.reward_parallel_workers > 0:
            return self._compute_batch_rewards_parallel(instances, rollout_result)
        return self._compute_batch_rewards_serial(instances, rollout_result)

    @staticmethod
    def _sample_gumbel_like(reference: torch.Tensor) -> torch.Tensor:
        uniform = torch.rand_like(reference).clamp_(1e-9, 1.0 - 1e-9)
        return -torch.log(-torch.log(uniform))

    def _build_stepwise_relaxed_perm(
        self,
        selected_decoder_probs: torch.Tensor,
        hard_perm: torch.Tensor,
        temperature: float,
        gumbels: torch.Tensor | None = None,
        hard_bias: float = 0.0,
    ) -> torch.Tensor:
        """Build a row-wise Concrete relaxation for the sampled route.

        The first POMO start row is fixed. Later rows use the decoder's masked
        probabilities, so already visited hard-route nodes stay unavailable.
        """
        if selected_decoder_probs.size(1) != hard_perm.size(1) - 1:
            raise ValueError(
                "decoder probability steps must contain exactly problem_size - 1 rows"
            )

        valid = selected_decoder_probs > 0
        very_negative = torch.full_like(selected_decoder_probs, -1e9)
        log_scores = torch.where(
            valid,
            selected_decoder_probs.clamp_min(1e-12).log(),
            very_negative,
        )
        if gumbels is not None:
            log_scores = torch.where(valid, log_scores + gumbels, very_negative)
        if hard_bias:
            log_scores = log_scores + hard_bias * hard_perm[:, 1:, :].detach()

        soft_rest = torch.softmax(log_scores / max(temperature, 1e-6), dim=-1)
        return torch.cat([hard_perm[:, :1, :], soft_rest], dim=1)

    @staticmethod
    def _hard_permutation_from_sequence(
        sequences: torch.Tensor,
        problem_size: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        hard_perm = torch.zeros(
            (sequences.size(0), problem_size, problem_size),
            dtype=dtype,
            device=device,
        )
        return hard_perm.scatter(2, sequences.to(device=device)[:, :, None], 1.0)

    def _build_cvxpylayer_route_indices(self, rewards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, pomo_size = rewards.shape
        max_instances = self.cvxpylayer_route_max_instances_per_batch
        if max_instances <= 0:
            max_instances = batch_size
        max_instances = min(batch_size, max_instances)

        batch_indices = torch.arange(max_instances, device=rewards.device)
        if self.cvxpylayer_route_selection == "first":
            pomo_indices = torch.zeros(
                (max_instances, self.cvxpylayer_route_candidates),
                dtype=torch.long,
                device=rewards.device,
            )
        else:
            candidate_count = min(self.cvxpylayer_route_candidates, pomo_size)
            pomo_indices = rewards[:max_instances].topk(candidate_count, dim=1).indices

        expanded_batch_indices = batch_indices[:, None].expand_as(pomo_indices).reshape(-1)
        expanded_pomo_indices = pomo_indices.reshape(-1)
        return expanded_batch_indices, expanded_pomo_indices

    def _cvxpylayer_route_loss_requested(self) -> bool:
        return self.cvxpylayer_rebar_loss_enable

    def _cvxpylayer_objective_is_main_loss(self) -> bool:
        return self.cvxpylayer_rebar_loss_enable

    def _loss_mode_name(self) -> str:
        if self.cvxpylayer_rebar_loss_enable:
            return "cvxpylayer_rebar"
        return "pomo_policy"

    def _compute_cvxpylayer_route_loss(
        self,
        instances,
        sequences: torch.Tensor,
        decoder_prob_steps: list[torch.Tensor],
        rewards: torch.Tensor,
        rollout_log_probs: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        zero = torch.zeros((), dtype=torch.float32, device=self.device)
        if (
            not self._cvxpylayer_route_loss_requested()
            or sequences.size(-1) <= 1
        ):
            return zero, 0
        if not decoder_prob_steps:
            return zero, 0
        if rollout_log_probs is None:
            raise RuntimeError("REBAR loss requires rollout log probabilities")

        problem_size = int(sequences.size(-1))
        batch_indices, pomo_indices = self._build_cvxpylayer_route_indices(rewards.detach())
        route_count = int(batch_indices.numel())
        if route_count == 0:
            return zero, 0

        route_device = torch.device("cpu") if self.cvxpylayer_route_device == "cpu" else self.device
        route_dtype = torch.float64 if self.solver_config.cvxpylayer_dtype == "float64" else torch.float32

        selected_sequences = sequences[batch_indices, pomo_indices].to(device=route_device)
        hard_perm = self._hard_permutation_from_sequence(
            selected_sequences,
            problem_size=problem_size,
            dtype=route_dtype,
            device=route_device,
        )

        batch_ids_cpu = batch_indices.detach().cpu().tolist()
        selected_instances = [instances[int(index)] for index in batch_ids_cpu]
        targets = torch.stack(
            [
                torch.as_tensor(instance.targets, dtype=route_dtype, device=route_device)
                for instance in selected_instances
            ],
            dim=0,
        )
        depots = torch.stack(
            [
                torch.as_tensor(instance.depot, dtype=route_dtype, device=route_device)
                for instance in selected_instances
            ],
            dim=0,
        )
        carrier_speeds = torch.as_tensor(
            [float(instance.carrier_speed) for instance in selected_instances],
            dtype=route_dtype,
            device=route_device,
        )
        uav_speeds = torch.as_tensor(
            [float(instance.uav_speed) for instance in selected_instances],
            dtype=route_dtype,
            device=route_device,
        )
        endurances = torch.as_tensor(
            [float(instance.endurance) for instance in selected_instances],
            dtype=route_dtype,
            device=route_device,
        )

        def solve_route_perm(route_perm: torch.Tensor) -> torch.Tensor:
            ordered_targets = torch.bmm(route_perm, targets)
            result = solve_ordered_targets_torch(
                depot=depots,
                ordered_targets=ordered_targets,
                carrier_speed=carrier_speeds,
                uav_speed=uav_speeds,
                endurance=endurances,
                solver_args=self.solver_config.cvxpylayer_solver_args,
                dtype=route_dtype,
            )
            route_objectives_inner = result["objective"]
            if self.cvxpylayer_route_normalize_by_size:
                route_objectives_inner = route_objectives_inner / max(problem_size, 1)
            return route_objectives_inner

        decoder_probs = torch.stack(decoder_prob_steps, dim=2)
        selected_decoder_probs = decoder_probs[batch_indices, pomo_indices].to(
            device=route_device,
            dtype=route_dtype,
        )
        gumbels = (
            self._sample_gumbel_like(selected_decoder_probs)
            if self.cvxpylayer_rebar_use_gumbel
            else None
        )
        soft_perm = self._build_stepwise_relaxed_perm(
            selected_decoder_probs=selected_decoder_probs,
            hard_perm=hard_perm,
            temperature=self.cvxpylayer_rebar_temperature,
            gumbels=gumbels,
            hard_bias=0.0,
        )
        conditional_perm = self._build_stepwise_relaxed_perm(
            selected_decoder_probs=selected_decoder_probs,
            hard_perm=hard_perm,
            temperature=self.cvxpylayer_rebar_temperature,
            gumbels=gumbels,
            hard_bias=self.cvxpylayer_rebar_conditional_bias,
        )
        soft_objectives = solve_route_perm(soft_perm)
        conditional_objectives = solve_route_perm(conditional_perm)

        hard_objectives = (-rewards[batch_indices, pomo_indices]).to(
            device=self.device,
            dtype=torch.float32,
        )
        baseline_objectives = (-rewards[batch_indices]).mean(dim=1).to(
            device=self.device,
            dtype=torch.float32,
        )
        selected_log_probs = rollout_log_probs[batch_indices, pomo_indices].to(
            device=self.device,
            dtype=torch.float32,
        )
        eta = float(self.cvxpylayer_rebar_eta)
        soft_objectives = soft_objectives.to(device=self.device, dtype=torch.float32)
        conditional_objectives = conditional_objectives.to(device=self.device, dtype=torch.float32)
        policy_control = (
            hard_objectives.detach()
            - eta * conditional_objectives.detach()
            - baseline_objectives.detach()
        )
        policy_term = (policy_control * selected_log_probs).mean()
        pathwise_term = eta * (soft_objectives - conditional_objectives).mean()
        return policy_term + pathwise_term, route_count

    def _train_one_batch(self, batch_size: int) -> tuple[float, float, float, float, int, float, int]:
        self.model.train()
        problem_size = self._sample_problem_size()
        self.env.load_problems(batch_size, self.device, problem_size=problem_size)
        instances = list(self.env.instances)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state.node_features)

        state, _, done = self.env.pre_step()
        batch_size_eff = len(instances)
        pomo_size = self.env.pomo_size
        log_prob_sum = torch.zeros((batch_size_eff, pomo_size), device=self.device)
        decoder_prob_steps: list[torch.Tensor] = []

        while not done:
            selected_count_before = state.selected_count
            selected, prob, all_probs = self.model(
                state,
                decode_type="sample",
                use_pomo_start=True,
                return_all_probs=True,
            )
            if selected_count_before >= 2:
                decoder_prob_steps.append(all_probs[:, :, 1:])
            state, _, done = self.env.step(selected)
            log_prob_sum = log_prob_sum + prob.clamp_min(1e-12).log()

        raw_node_indices = self.env.selected_node_list
        sequences = raw_node_indices[:, :, 1:] - 1
        rollout_result = RolloutResult(
            sequences=sequences,
            log_probs=log_prob_sum,
            raw_node_indices=raw_node_indices,
        )
        rewards, feasible_ratio, solver_calls = self._compute_batch_rewards(instances, rollout_result)

        advantage = rewards - rewards.mean(dim=1, keepdim=True)
        policy_loss = -(advantage * rollout_result.log_probs).mean()
        cvxpylayer_route_loss, cvxpylayer_route_count = self._compute_cvxpylayer_route_loss(
            instances=instances,
            sequences=sequences,
            decoder_prob_steps=decoder_prob_steps,
            rewards=rewards,
            rollout_log_probs=rollout_result.log_probs,
        )
        if self.cvxpylayer_rebar_loss_enable:
            if cvxpylayer_route_count == 0:
                raise RuntimeError(
                    "REBAR cvxpylayer loss was enabled but no cvxpylayer loss was computed"
                )
            loss = cvxpylayer_route_loss
        else:
            loss = policy_loss + self.cvxpylayer_aux_weight * cvxpylayer_route_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()

        best_reward = rewards.max(dim=1).values
        score_mean = float((-best_reward).mean().item())
        return (
            score_mean,
            float(loss.item()),
            feasible_ratio,
            float(problem_size),
            solver_calls,
            float(cvxpylayer_route_loss.detach().item()),
            cvxpylayer_route_count,
        )

    def _append_metrics_row(self, row: dict[str, Any]) -> None:
        write_header = not self.metrics_path.exists()
        with self.metrics_path.open("a", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def close(self) -> None:
        if self._reward_executor is not None:
            self._reward_executor.shutdown(wait=True)
            self._reward_executor = None
        if self._reward_worker_pool is not None:
            self._reward_worker_pool.close()
            self._reward_worker_pool = None

    def _save_checkpoint(self, epoch: int, metrics: dict[str, Any], path: Path) -> None:
        save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=epoch,
            best_metric=self.best_score,
            metrics=metrics,
        )

    def _train_one_epoch(self, epoch: int) -> dict[str, float]:
        total_score = 0.0
        total_loss = 0.0
        total_cvxpylayer_route_loss = 0.0
        total_cvxpylayer_route_count = 0
        total_feasible = 0.0
        total_solver_calls = 0
        total_batches = 0
        total_problem_size = 0.0
        episode = 0
        progress_step = max(int(self.train_episodes * (self.progress_log_percent / 100.0)), 1)
        next_progress_log = progress_step

        while episode < self.train_episodes:
            remaining = self.train_episodes - episode
            batch_size = min(self.train_batch_size, remaining)
            score, loss, feasible_ratio, problem_size, solver_calls, cvxpylayer_route_loss, cvxpylayer_route_count = (
                self._train_one_batch(batch_size)
            )

            total_score += score * batch_size
            total_loss += loss * batch_size
            total_cvxpylayer_route_loss += cvxpylayer_route_loss * batch_size
            total_cvxpylayer_route_count += cvxpylayer_route_count
            total_feasible += feasible_ratio * batch_size
            total_problem_size += problem_size * batch_size
            total_solver_calls += solver_calls
            total_batches += 1
            episode += batch_size

            should_log_initial = epoch == 1 and total_batches <= self.log_first_batch_count
            should_log_progress = episode >= next_progress_log or episode == self.train_episodes
            if should_log_progress:
                while next_progress_log <= episode:
                    next_progress_log += progress_step

            if should_log_initial or should_log_progress:
                progress_bar = _format_progress_bar(episode, self.train_episodes, width=self.progress_bar_width)
                logger.info(
                    "Epoch {:3d}: {} {:6d}/{:6d}({:5.1f}%)  J: {:3.0f}  Score: {:.4f}  Loss: {:.4f}  CVX: {:.4f}",
                    epoch,
                    progress_bar,
                    episode,
                    self.train_episodes,
                    100.0 * episode / self.train_episodes,
                    problem_size,
                    total_score / max(episode, 1),
                    total_loss / max(episode, 1),
                    total_cvxpylayer_route_loss / max(episode, 1),
                )

        avg_cvxpylayer_route_loss = total_cvxpylayer_route_loss / max(self.train_episodes, 1)
        return {
            "epoch": float(epoch),
            "train_score": total_score / max(self.train_episodes, 1),
            "train_loss": total_loss / max(self.train_episodes, 1),
            "loss_mode": self._loss_mode_name(),
            "cvxpylayer_route_loss": avg_cvxpylayer_route_loss,
            "cvxpylayer_route_count": float(total_cvxpylayer_route_count),
            "cvxpylayer_aux_loss": 0.0 if self._cvxpylayer_objective_is_main_loss() else avg_cvxpylayer_route_loss,
            "cvxpylayer_aux_count": 0.0
            if self._cvxpylayer_objective_is_main_loss()
            else float(total_cvxpylayer_route_count),
            "cvxpylayer_objective_loss": avg_cvxpylayer_route_loss
            if self._cvxpylayer_objective_is_main_loss()
            else 0.0,
            "cvxpylayer_objective_count": float(total_cvxpylayer_route_count)
            if self._cvxpylayer_objective_is_main_loss()
            else 0.0,
            "train_feasible_ratio": total_feasible / max(self.train_episodes, 1),
            "avg_problem_size": total_problem_size / max(self.train_episodes, 1),
            "solver_calls": float(total_solver_calls),
            "cache_hits": float(self.cache.hits),
            "cache_misses": float(self.cache.misses),
        }

    def run(self) -> dict[str, Any]:
        last_metrics: dict[str, Any] = {}
        logger.info("Generator stats: {}", self.generator.stats.to_dict())
        try:
            # for epoch in range(1, self.epochs + 1):
            for epoch in range(self.start_epoch, self.epochs + 1):
                metrics = self._train_one_epoch(epoch)
                self._append_metrics_row(metrics)
                last_metrics = metrics

                current_score = float(metrics["train_score"])
                if current_score < self.best_score:
                    self.best_score = current_score
                    self._save_checkpoint(epoch, metrics, self.output_dir / "best.pt")

                if epoch % self.checkpoint_interval == 0 or epoch == self.epochs:
                    self._save_checkpoint(epoch, metrics, self.output_dir / f"checkpoint-{epoch}.pt")
                    self._save_checkpoint(epoch, metrics, self.output_dir / "last.pt")

                logger.info(
                    "Epoch {:4d}/{:4d}: Mode={} Score={:.4f} Loss={:.4f} CVX={:.4f} Feas={:.3f} AvgJ={:.2f}",
                    epoch,
                    self.epochs,
                    metrics["loss_mode"],
                    metrics["train_score"],
                    metrics["train_loss"],
                    metrics["cvxpylayer_route_loss"],
                    metrics["train_feasible_ratio"],
                    metrics["avg_problem_size"],
                )

                if self.scheduler is not None:
                    self.scheduler.step()

            summary = {
                "output_dir": str(self.output_dir),
                "best_train_score": self.best_score,
                "last_metrics": last_metrics,
                "generator_stats": self.generator.stats.to_dict(),
                "env_params": self.env_params,
                "model_params": self.model_params,
                "optimizer_params": self.optimizer_params,
                "trainer_params": self.trainer_params,
            }
            (self.output_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            return summary
        finally:
            self.close()


def run_training(config: AppConfig) -> dict[str, Any]:
    return TSPTrainer(config).run()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the CVTSP master policy.")
    parser.add_argument("--config")
    parser.add_argument("--dataset-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--max-batches-per-epoch", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--log-level")
    return parser


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.dataset_dir:
        config.data.dataset_dir = args.dataset_dir
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.checkpoint_path:
        config.runtime.checkpoint_path = args.checkpoint_path
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.batch_size is not None:
        config.train.batch_size = args.batch_size
    if args.max_batches_per_epoch is not None:
        config.train.max_batches_per_epoch = args.max_batches_per_epoch
    if args.seed is not None:
        config.runtime.seed = args.seed
    if args.device:
        config.runtime.device = args.device
    if args.log_level:
        config.runtime.log_level = args.log_level
    return config


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    config = _apply_overrides(load_config(args.config), args)
    setup_logger(config.runtime.output_dir, config.runtime.log_level)
    set_seed(config.runtime.seed)

    summary = run_training(config)
    summary_path = Path(config.runtime.output_dir) / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
