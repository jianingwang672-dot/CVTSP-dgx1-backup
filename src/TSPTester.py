from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from typing import Any, Iterable

import torch
from loguru import logger

from src.CVPSolver import SPSolution, SolverConfig, solve
from src.TSPEnv import TSPEnv, generate_sequences
from src.TSPModel import ModelConfig, TSPModel
from src.TSProblemDef import CVTSPInstance, augment_instance_by_8_fold, load_dataset, load_instance
from src.TSPUtils import (
    AppConfig,
    load_checkpoint,
    load_config,
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


@dataclass
class CandidateEvaluation:
    sequence: list[int]
    solution: SPSolution

    def to_dict(self) -> dict[str, Any]:
        return {"sequence": list(self.sequence), "solution": self.solution.to_dict()}


@dataclass
class EvaluationResult:
    instance_id: str
    decode_type: str
    num_candidates: int
    success: bool
    best_sequence: list[int]
    best_objective: float
    best_status: str
    candidates: list[CandidateEvaluation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        best_solution = None
        for candidate in self.candidates:
            if list(candidate.sequence) == list(self.best_sequence):
                best_solution = candidate.solution.to_dict()
                break

        return {
            "instance_id": self.instance_id,
            "decode_type": self.decode_type,
            "num_candidates": self.num_candidates,
            "success": self.success,
            "best_sequence": list(self.best_sequence),
            "best_target_sequence_0based": list(self.best_sequence),
            "best_target_sequence_1based": [node + 1 for node in self.best_sequence],
            "best_route_with_depot": best_solution.get("route_with_depot") if best_solution else None,
            "best_objective": float(self.best_objective),
            "best_status": self.best_status,
            "candidates": [candidate.to_dict() for candidate in self.candidates],
        }


class RewardCache:
    def __init__(self):
        self._cache: dict[tuple[str, tuple[int, ...]], SPSolution] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def make_key(instance_id: str, sequence: list[int]) -> tuple[str, tuple[int, ...]]:
        return (instance_id, tuple(sequence))

    def get_cached(self, instance: CVTSPInstance, sequence: list[int]) -> SPSolution | None:
        key = self.make_key(instance.instance_id, sequence)
        solution = self._cache.get(key)
        if solution is not None:
            self.hits += 1
        return solution

    def put(self, instance: CVTSPInstance, sequence: list[int], solution: SPSolution) -> None:
        key = self.make_key(instance.instance_id, sequence)
        self._cache[key] = solution
        self.misses += 1

    def get_or_solve(
        self,
        instance: CVTSPInstance,
        sequence: list[int],
        solver_config: SolverConfig,
    ) -> tuple[SPSolution, bool]:
        cached = self.get_cached(instance, sequence)
        if cached is not None:
            return cached, True
        solution = solve(instance, sequence, solver_config)
        self.put(instance, sequence, solution)
        return solution, False


def _deduplicate_sequences(sequences) -> list[list[int]]:
    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for sequence in sequences:
        key = tuple(sequence)
        if key in seen:
            continue
        seen.add(key)
        unique.append(list(sequence))
    return unique


def _normalize_sequence_input(sequences: Iterable[Iterable[int]]) -> list[list[int]]:
    normalized = [list(map(int, sequence)) for sequence in sequences]
    return _deduplicate_sequences(normalized)


def _build_augmented_instances(instance: CVTSPInstance, aug_factor: int) -> list[CVTSPInstance]:
    if aug_factor <= 1:
        return [instance]
    if aug_factor > 8:
        raise ValueError(f"aug_factor must be in [1, 8], got {aug_factor}")
    return augment_instance_by_8_fold(instance)[:aug_factor]


def evaluate_sequences(
    instance: CVTSPInstance,
    sequences: Iterable[Iterable[int]],
    solver_config: SolverConfig,
    cache: RewardCache | None = None,
) -> list[CandidateEvaluation]:
    reward_cache = cache or RewardCache()
    candidates = []
    for sequence in _normalize_sequence_input(sequences):
        solution, _ = reward_cache.get_or_solve(instance, sequence, solver_config)
        candidates.append(CandidateEvaluation(sequence=sequence, solution=solution))
    return candidates


def _select_best_candidate(candidates: list[CandidateEvaluation]) -> CandidateEvaluation:
    successful = [candidate for candidate in candidates if candidate.solution.success]
    if successful:
        return min(successful, key=lambda item: item.solution.objective)
    return min(candidates, key=lambda item: item.solution.objective)


class TSPTester:
    def __init__(
        self,
        config: AppConfig,
        model: TSPModel | None = None,
        device: torch.device | None = None,
        cache: RewardCache | None = None,
    ):
        self.config = config
        self.device = device or torch.device(config.runtime.device)
        self.model = model
        self.cache = cache or RewardCache()

    def _ensure_model(self) -> TSPModel:
        if self.model is None:
            self.model, _ = load_model_for_inference(self.config, self.device)
        return self.model

    def evaluate_instance(
        self,
        instance: CVTSPInstance,
        sequences: Iterable[Iterable[int]] | None = None,
    ) -> EvaluationResult:
        if sequences is not None:
            candidate_sequences = _normalize_sequence_input(sequences)
            decode_type = self.config.decode.decode_type
        else:
            model = self._ensure_model()
            if self.config.decode.augmentation_enable:
                candidate_sequences = []
                for augmented_instance in _build_augmented_instances(instance, self.config.decode.aug_factor):
                    candidate_sequences.extend(
                        generate_sequences(
                            model=model,
                            instance=augmented_instance,
                            device=self.device,
                            decode_type=self.config.decode.decode_type,
                            num_candidates=self.config.decode.num_candidates,
                            sample_max_rollouts=self.config.decode.sample_max_rollouts,
                        )
                    )
                candidate_sequences = _normalize_sequence_input(candidate_sequences)
            else:
                candidate_sequences = generate_sequences(
                    model=model,
                    instance=instance,
                    device=self.device,
                    decode_type=self.config.decode.decode_type,
                    num_candidates=self.config.decode.num_candidates,
                    sample_max_rollouts=self.config.decode.sample_max_rollouts,
                )
            decode_type = self.config.decode.decode_type

        candidate_records = evaluate_sequences(
            instance,
            candidate_sequences,
            solver_config=self.config.solver,
            cache=self.cache,
        )
        best_candidate = _select_best_candidate(candidate_records)
        return EvaluationResult(
            instance_id=instance.instance_id,
            decode_type=decode_type,
            num_candidates=len(candidate_records),
            success=best_candidate.solution.success,
            best_sequence=list(best_candidate.sequence),
            best_objective=float(best_candidate.solution.objective),
            best_status=best_candidate.solution.status,
            candidates=candidate_records,
        )

    def evaluate_dataset(self, instances: list[CVTSPInstance]) -> tuple[list[EvaluationResult], dict[str, float]]:
        results = [self.evaluate_instance(instance) for instance in instances]
        objectives = [result.best_objective if isfinite(result.best_objective) else 1e6 for result in results]
        feasible = [1.0 if result.success else 0.0 for result in results]
        summary = {
            "num_instances": float(len(results)),
            "avg_objective": float(sum(objectives) / max(len(objectives), 1)),
            "feasible_ratio": float(sum(feasible) / max(len(feasible), 1)),
        }
        return results, summary


class OnlineTSPTester:
    def __init__(
        self,
        env_params: dict[str, Any],
        model_params: dict[str, Any],
        tester_params: dict[str, Any],
    ):
        self.env_params = dict(env_params)
        self.model_params = dict(model_params)
        self.tester_params = dict(tester_params)

        seed = int(self.tester_params.get("seed", 1234))
        set_seed(seed)

        self.output_dir = Path(self.tester_params.get("result_folder", "outputs/test_n100"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_logger(self.output_dir, self.tester_params.get("log_level", "INFO"))

        use_cuda = bool(self.tester_params.get("use_cuda", True)) and torch.cuda.is_available()
        cuda_device_num = int(self.tester_params.get("cuda_device_num", 0))
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

        model_load = self.tester_params.get("model_load", {})
        checkpoint_path = model_load.get("checkpoint_path")
        if not checkpoint_path:
            model_dir = model_load.get("path", "")
            model_epoch = model_load.get("epoch")
            if model_dir and model_epoch is not None:
                checkpoint_path = str(Path(model_dir) / f"checkpoint-{model_epoch}.pt")
        if not checkpoint_path:
            raise ValueError("tester_params.model_load must provide checkpoint_path or path+epoch")
        load_checkpoint(checkpoint_path, self.model, map_location=self.device)
        self.model.eval()

        self.solver_config = SolverConfig(
            solver_backend=str(self.tester_params.get("solver_backend", "gurobi")),
            gurobi_time_limit=self.tester_params.get("gurobi_time_limit"),
            gurobi_threads=int(self.tester_params.get("gurobi_threads", 16)),
            output_flag=int(self.tester_params.get("output_flag", 0)),
            cvxpylayer_solver_args=dict(self.tester_params.get("cvxpylayer_solver_args", {}) or {}),
            cvxpylayer_dtype=str(self.tester_params.get("cvxpylayer_dtype", "float64")),
        )
        self.test_episodes = int(self.tester_params.get("test_episodes", 1000))
        self.test_batch_size = int(self.tester_params.get("test_batch_size", 64))
        self.decode_type = str(self.tester_params.get("decode_type", "greedy"))
        self.test_mode = str(self.tester_params.get("test_mode", "random")).lower()
        self.cache = RewardCache()

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

        snapshot = {
            "env_params": self.env_params,
            "model_params": self.model_params,
            "tester_params": self.tester_params,
            "generator_stats": self.env.generator.stats.to_dict(),
        }
        (self.output_dir / "params.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")

    def _build_real_eval_config(self) -> AppConfig:
        config = AppConfig()
        config.data.dataset_dir = str(self.env_params.get("dataset_dir", "instance/Data"))
        config.data.eval_split = str(self.tester_params.get("real_split", "all"))
        config.data.train_ratio = float(self.tester_params.get("train_ratio", config.data.train_ratio))
        config.data.val_ratio = float(self.tester_params.get("val_ratio", config.data.val_ratio))
        config.data.test_ratio = float(self.tester_params.get("test_ratio", config.data.test_ratio))
        config.data.split_seed = int(self.tester_params.get("split_seed", config.data.split_seed))
        config.model = self.model_config
        config.decode.decode_type = self.decode_type
        config.decode.num_candidates = int(
            self.tester_params.get("num_candidates", self.model_config.max_pomo_size)
        )
        config.decode.sample_max_rollouts = int(
            self.tester_params.get("sample_max_rollouts", config.decode.sample_max_rollouts)
        )
        config.decode.augmentation_enable = bool(
            self.tester_params.get("augmentation_enable", config.decode.augmentation_enable)
        )
        config.decode.aug_factor = int(self.tester_params.get("aug_factor", config.decode.aug_factor))
        config.solver = self.solver_config
        config.runtime.device = str(self.device)
        config.runtime.output_dir = str(self.output_dir)
        config.runtime.log_level = str(self.tester_params.get("log_level", "INFO"))
        return config

    def _generate_batch_candidates(self, batch_size: int) -> tuple[list[CVTSPInstance], list[list[list[int]]]]:
        self.env.load_problems(batch_size, self.device)
        instances = list(self.env.instances)

        with torch.no_grad():
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state.node_features)
            state, _, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state, decode_type=self.decode_type, use_pomo_start=True)
                state, _, done = self.env.step(selected)

        sequences = self.env.selected_node_list[:, :, 1:] - 1
        candidate_sequences = [
            [list(map(int, sequence)) for sequence in batch_sequences]
            for batch_sequences in sequences.cpu().tolist()
        ]
        return instances, candidate_sequences

    def _evaluate_batch(self, batch_size: int) -> list[EvaluationResult]:
        instances, candidate_sequences = self._generate_batch_candidates(batch_size)
        results: list[EvaluationResult] = []
        for instance, sequences in zip(instances, candidate_sequences):
            candidate_records = evaluate_sequences(
                instance,
                sequences,
                solver_config=self.solver_config,
                cache=self.cache,
            )
            best_candidate = _select_best_candidate(candidate_records)
            results.append(
                EvaluationResult(
                    instance_id=instance.instance_id,
                    decode_type=self.decode_type,
                    num_candidates=len(candidate_records),
                    success=best_candidate.solution.success,
                    best_sequence=list(best_candidate.sequence),
                    best_objective=float(best_candidate.solution.objective),
                    best_status=best_candidate.solution.status,
                    candidates=candidate_records,
                )
            )
        return results

    def _run_random(self) -> dict[str, Any]:
        results: list[EvaluationResult] = []
        episode = 0
        while episode < self.test_episodes:
            remaining = self.test_episodes - episode
            batch_size = min(self.test_batch_size, remaining)
            batch_results = self._evaluate_batch(batch_size)
            results.extend(batch_results)
            episode += len(batch_results)
            if episode <= min(self.test_episodes, self.test_batch_size * 2):
                avg = sum(item.best_objective for item in results) / max(len(results), 1)
                logger.info(
                    "Random Test {:6d}/{:6d}({:4.1f}%) AvgObj={:.4f}",
                    episode,
                    self.test_episodes,
                    100.0 * episode / self.test_episodes,
                    avg,
                )

        objectives = [result.best_objective if isfinite(result.best_objective) else 1e6 for result in results]
        feasible = [1.0 if result.success else 0.0 for result in results]
        summary = {
            "num_instances": float(len(results)),
            "avg_objective": float(sum(objectives) / max(len(objectives), 1)),
            "feasible_ratio": float(sum(feasible) / max(len(feasible), 1)),
            "cache_hits": float(self.cache.hits),
            "cache_misses": float(self.cache.misses),
            "test_mode": "random",
        }
        payload = {"summary": summary, "results": [result.to_dict() for result in results]}
        (self.output_dir / "evaluate_random.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_best_result_bundle(self.output_dir, payload["results"], payload["summary"], "random")
        return payload

    def _run_real_dataset(self) -> dict[str, Any]:
        config = self._build_real_eval_config()
        split_name = config.data.eval_split
        instances = load_dataset(
            config.data.dataset_dir,
            split_name,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            test_ratio=config.data.test_ratio,
            split_seed=config.data.split_seed,
        )
        with torch.no_grad():
            results, summary = evaluate_dataset(
                instances,
                config=config,
                model=self.model,
                device=self.device,
                cache=self.cache,
            )
        summary = dict(summary)
        summary["cache_hits"] = float(self.cache.hits)
        summary["cache_misses"] = float(self.cache.misses)
        summary["test_mode"] = "real529"
        payload = {"summary": summary, "results": [result.to_dict() for result in results]}
        (self.output_dir / f"evaluate_{split_name}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        _write_best_result_bundle(self.output_dir, payload["results"], payload["summary"], split_name)
        logger.info(
            "Real Test split={} num_instances={} avg_obj={:.4f} feas={:.3f}",
            split_name,
            int(summary["num_instances"]),
            summary["avg_objective"],
            summary["feasible_ratio"],
        )
        return payload

    def run(self) -> dict[str, Any]:
        if self.test_mode == "random":
            return self._run_random()
        if self.test_mode in {"real", "real529"}:
            return self._run_real_dataset()
        raise ValueError(f"unsupported test_mode '{self.test_mode}'")


def load_model_for_inference(config: AppConfig, device: torch.device) -> tuple[TSPModel, dict]:
    model = TSPModel(config.model).to(device)
    if not config.runtime.checkpoint_path:
        raise ValueError("checkpoint_path is required for inference")
    checkpoint = load_checkpoint(config.runtime.checkpoint_path, model, map_location=device)
    model.eval()
    return model, checkpoint


def evaluate_instance(
    instance: CVTSPInstance,
    config: AppConfig,
    model: TSPModel | None = None,
    sequences: Iterable[Iterable[int]] | None = None,
    device: torch.device | None = None,
    cache: RewardCache | None = None,
) -> EvaluationResult:
    tester = TSPTester(config=config, model=model, device=device, cache=cache)
    return tester.evaluate_instance(instance, sequences=sequences)


def evaluate_dataset(
    instances: list[CVTSPInstance],
    config: AppConfig,
    model: TSPModel | None = None,
    device: torch.device | None = None,
    cache: RewardCache | None = None,
) -> tuple[list[EvaluationResult], dict[str, float]]:
    tester = TSPTester(config=config, model=model, device=device, cache=cache)
    return tester.evaluate_dataset(instances)


def run_inference(
    config: AppConfig,
    instance_path: str | Path,
    cache: RewardCache | None = None,
) -> tuple[CVTSPInstance, EvaluationResult]:
    instance = load_instance(instance_path)
    tester = TSPTester(config=config, cache=cache)
    with torch.no_grad():
        result = tester.evaluate_instance(instance)
    return instance, result


def _compact_solution_for_export(solution: dict) -> dict:
    return {
        "status": solution.get("status"),
        "solve_time": solution.get("solve_time"),
        "takeoff_points": solution.get("takeoff_points"),
        "landing_points": solution.get("landing_points"),
        "t1": solution.get("t1"),
        "t2": solution.get("t2"),
        "Tij": solution.get("Tij"),
    }


def _write_best_result_bundle(output_dir: Path, results: list[dict], summary: dict, split_name: str) -> None:
    bundle_dir = output_dir / "best_results"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for result in results:
        best_candidate = min(
            result["candidates"],
            key=lambda candidate: float(candidate["solution"]["objective"]),
        )
        payload = {
            "instance_id": result["instance_id"],
            "success": result["success"],
            "objective": result["best_objective"],
            "final_route": best_candidate["solution"].get("route_with_depot"),
            "solution": _compact_solution_for_export(best_candidate["solution"]),
        }
        (bundle_dir / f"{result['instance_id']}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        summary_rows.append(
            {
                "instance_id": result["instance_id"],
                "success": result["success"],
                "best_objective": result["best_objective"],
                "best_status": result["best_status"],
                "best_sequence": " ".join(map(str, result["best_sequence"])),
                "best_route_with_depot": " ".join(map(str, best_candidate["solution"].get("route_with_depot", []))),
            }
        )

    summary_path = output_dir / f"{split_name}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary_rows[0].keys()) if summary_rows else ["instance_id"])
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    (output_dir / f"{split_name}_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a split or a single instance.")
    parser.add_argument("--config")
    parser.add_argument("--split")
    parser.add_argument("--instance-path")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--decode-type")
    parser.add_argument("--num-candidates", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--log-level")
    return parser


def _apply_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    if args.split:
        config.data.eval_split = args.split
    if args.checkpoint_path:
        config.runtime.checkpoint_path = args.checkpoint_path
    if args.output_dir:
        config.runtime.output_dir = args.output_dir
    if args.decode_type:
        config.decode.decode_type = args.decode_type
    if args.num_candidates is not None:
        config.decode.num_candidates = args.num_candidates
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

    output_dir = Path(config.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = RewardCache()

    if args.instance_path:
        instance = load_instance(args.instance_path)
        device = torch.device(config.runtime.device)
        model, _ = load_model_for_inference(config, device)
        with torch.no_grad():
            result = evaluate_instance(instance, config=config, model=model, device=device, cache=cache)
        payload = result.to_dict()
        (output_dir / f"evaluate_{instance.instance_id}.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "instance_id": payload["instance_id"],
                    "success": payload["success"],
                    "best_objective": payload["best_objective"],
                    "best_sequence": payload["best_sequence"],
                    "num_candidates": payload["num_candidates"],
                    "details_path": str(output_dir / f"evaluate_{instance.instance_id}.json"),
                },
                indent=2,
            )
        )
        return

    instances = load_dataset(
        config.data.dataset_dir,
        config.data.eval_split,
        train_ratio=config.data.train_ratio,
        val_ratio=config.data.val_ratio,
        test_ratio=config.data.test_ratio,
        split_seed=config.data.split_seed,
    )
    device = torch.device(config.runtime.device)
    model, _ = load_model_for_inference(config, device)
    with torch.no_grad():
        results, summary = evaluate_dataset(instances, config=config, model=model, device=device, cache=cache)

    payload = {"summary": summary, "results": [result.to_dict() for result in results]}
    (output_dir / f"evaluate_{config.data.eval_split}.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    _write_best_result_bundle(output_dir, payload["results"], payload["summary"], config.data.eval_split)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
