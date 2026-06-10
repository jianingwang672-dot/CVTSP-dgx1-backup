from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from CVTSP_SOCP import gurobi_cvp_socp
from src.CVXPYLayerSolver import solve_fixed_sequence_numpy
from src.TSProblemDef import CVTSPInstance


@dataclass
class SolverConfig:
    solver_backend: str = "gurobi"
    gurobi_time_limit: float | None = None
    gurobi_threads: int | None = 16
    output_flag: int = 0
    cvxpylayer_solver_args: dict[str, Any] = field(default_factory=dict)
    cvxpylayer_dtype: str = "float64"


@dataclass
class SPSolution:
    objective: float
    makespan: float
    success: bool
    status: str
    solve_time: float
    sequence: list[int]
    takeoff_points: np.ndarray | None = None
    landing_points: np.ndarray | None = None
    t1: np.ndarray | None = None
    t2: np.ndarray | None = None
    tau: np.ndarray | None = None
    Tseg: np.ndarray | None = None
    raw_debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = _to_serializable(asdict(self))
        payload["target_sequence_0based"] = list(self.sequence)
        payload["target_sequence_1based"] = [node + 1 for node in self.sequence]
        payload["visit_order_1based"] = [node + 1 for node in self.sequence]
        if self.Tseg is not None:
            route = [0] + [node + 1 for node in self.sequence] + [0]
            edge_times = []
            tseg_values = self.Tseg.tolist() if isinstance(self.Tseg, np.ndarray) else list(self.Tseg)
            for idx, time_value in enumerate(tseg_values):
                edge_times.append(
                    {
                        "from": int(route[idx]),
                        "to": int(route[idx + 1]),
                        "time": float(time_value),
                    }
                )
            payload["Tij"] = edge_times
            payload["route_with_depot"] = route
            payload["full_route_depot0_targets1based"] = route
        return payload


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    return value


def _assert_valid_sequence(sequence: list[int], problem_size: int) -> None:
    seq = list(sequence)
    if len(seq) != problem_size:
        raise ValueError(f"invalid sequence: expected length {problem_size}, got {len(seq)}")
    if any(node < 0 or node >= problem_size for node in seq):
        raise ValueError(f"invalid sequence: sequence contains indices outside 0..{problem_size - 1}")
    if len(set(seq)) != problem_size:
        raise ValueError("invalid sequence: sequence contains duplicates or omissions")


def solve(instance: CVTSPInstance, sequence: list[int], config: SolverConfig, env: Any = None) -> SPSolution:
    _assert_valid_sequence(sequence, instance.J)
    start_time = time.perf_counter()
    try:
        backend = str(config.solver_backend).strip().lower()
        if backend == "gurobi":
            result = gurobi_cvp_socp(
                depot=instance.depot,
                targets=instance.targets,
                tour=sequence,
                Vc=instance.carrier_speed,
                Vv=instance.uav_speed,
                endurance_a=instance.endurance,
                threads=config.gurobi_threads,
                time_limit=config.gurobi_time_limit,
                test=True,
                output_flag=config.output_flag,
                env=env,
            )
        elif backend in {"cvxpylayer", "cvxpy_layer", "cvxpy"}:
            result = solve_fixed_sequence_numpy(
                depot=instance.depot,
                targets=instance.targets,
                sequence=sequence,
                carrier_speed=instance.carrier_speed,
                uav_speed=instance.uav_speed,
                endurance=instance.endurance,
                solver_args=config.cvxpylayer_solver_args,
                dtype=config.cvxpylayer_dtype,
            )
        else:
            raise ValueError(f"unsupported solver_backend '{config.solver_backend}'")
        solve_time = time.perf_counter() - start_time
        return SPSolution(
            objective=float(result["obj"]),
            makespan=float(result["obj"]),
            success=True,
            status=str(result.get("status", "UNKNOWN")),
            solve_time=solve_time,
            sequence=list(sequence),
            takeoff_points=np.asarray(result.get("sx")),
            landing_points=np.asarray(result.get("lx")),
            t1=np.asarray(result.get("t1")),
            t2=np.asarray(result.get("t2")),
            tau=np.asarray(result.get("tau")),
            Tseg=np.asarray(result.get("Tseg")),
            raw_debug={k: v for k, v in result.items() if k not in {"sx", "lx", "t1", "t2", "tau", "Tseg"}},
        )
    except Exception as exc:  # pragma: no cover
        solve_time = time.perf_counter() - start_time
        return SPSolution(
            objective=float("inf"),
            makespan=float("inf"),
            success=False,
            status=f"ERROR:{type(exc).__name__}",
            solve_time=solve_time,
            sequence=list(sequence),
            raw_debug={"error": repr(exc)},
        )
