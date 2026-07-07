from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn


DEFAULT_SOLVER_ARGS: dict[str, Any] = {
    "eps": 1e-5,
    "max_iters": 10000,
}


@dataclass(frozen=True)
class CVPLayerBundle:
    problem_size: int
    layer: Any


_LAYER_CACHE: dict[int, CVPLayerBundle] = {}
_OBJECTIVE_LAYER_CACHE: dict[int, CVPLayerBundle] = {}
_DLL_DIRECTORY_HANDLES: list[Any] = []
_DLL_DIRECTORY_PATHS: set[str] = set()


def _add_windows_dll_directories() -> None:
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return
    candidates = [
        Path(r"C:\Strawberry\c\bin"),
        Path(r"C:\msys64\mingw64\bin"),
        Path(r"C:\msys64\ucrt64\bin"),
    ]
    for path in candidates:
        if not path.exists():
            continue
        normalized = str(path.resolve()).lower()
        if normalized in _DLL_DIRECTORY_PATHS:
            continue
        try:
            handle = os.add_dll_directory(str(path))
        except (FileNotFoundError, OSError):
            continue
        _DLL_DIRECTORY_HANDLES.append(handle)
        _DLL_DIRECTORY_PATHS.add(normalized)


def _import_cvxpy_layer():
    try:
        _add_windows_dll_directories()
        import cvxpy as cp
        from cvxpylayers.torch import CvxpyLayer
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise ImportError(
            "cvxpylayer backend requires cvxpy, cvxpylayers, and diffcp. "
            "Install them with `python -m pip install cvxpy cvxpylayers diffcp`."
        ) from exc
    return cp, CvxpyLayer


def _build_cvp_layer(problem_size: int, *, objective_only: bool) -> CVPLayerBundle:
    problem_size = int(problem_size)
    if problem_size <= 0:
        raise ValueError("problem_size must be positive")

    cp, CvxpyLayer = _import_cvxpy_layer()

    depot = cp.Parameter(2, name="depot")
    targets = cp.Parameter((problem_size, 2), name="ordered_targets")
    carrier_speed = cp.Parameter(nonneg=True, name="carrier_speed")
    uav_speed = cp.Parameter(nonneg=True, name="uav_speed")
    endurance = cp.Parameter(nonneg=True, name="endurance")

    takeoff = cp.Variable((problem_size, 2), name="takeoff")
    landing = cp.Variable((problem_size, 2), name="landing")
    t1 = cp.Variable(problem_size, nonneg=True, name="t1")
    t2 = cp.Variable(problem_size, nonneg=True, name="t2")
    tau = cp.Variable(problem_size, nonneg=True, name="tau")
    tseg = cp.Variable(problem_size + 1, nonneg=True, name="Tseg")

    constraints = [
        tau == t1 + t2,
        tau <= endurance,
    ]
    for idx in range(problem_size):
        target = targets[idx, :]
        constraints.extend(
            [
                cp.norm(takeoff[idx, :] - target, 2) <= uav_speed * t1[idx],
                cp.norm(landing[idx, :] - target, 2) <= uav_speed * t2[idx],
                cp.norm(landing[idx, :] - takeoff[idx, :], 2) <= carrier_speed * tau[idx],
            ]
        )
        previous_landing = depot if idx == 0 else landing[idx - 1, :]
        constraints.append(cp.norm(takeoff[idx, :] - previous_landing, 2) <= carrier_speed * tseg[idx])
    constraints.append(cp.norm(depot - landing[problem_size - 1, :], 2) <= carrier_speed * tseg[problem_size])

    objective = cp.Minimize(cp.sum(tau) + cp.sum(tseg))
    problem = cp.Problem(objective, constraints)
    if not problem.is_dpp():
        raise RuntimeError("fixed-tour CVP cvxpylayer formulation is not DPP")

    output_variables = [tau, tseg] if objective_only else [takeoff, landing, t1, t2, tau, tseg]
    bundle = CVPLayerBundle(
        problem_size=problem_size,
        layer=CvxpyLayer(
            problem,
            parameters=[depot, targets, carrier_speed, uav_speed, endurance],
            variables=output_variables,
        ),
    )
    return bundle


def get_cvp_layer(problem_size: int) -> CVPLayerBundle:
    problem_size = int(problem_size)
    cached = _LAYER_CACHE.get(problem_size)
    if cached is not None:
        return cached
    bundle = _build_cvp_layer(problem_size, objective_only=False)
    _LAYER_CACHE[problem_size] = bundle
    return bundle


def get_cvp_objective_layer(problem_size: int) -> CVPLayerBundle:
    problem_size = int(problem_size)
    cached = _OBJECTIVE_LAYER_CACHE.get(problem_size)
    if cached is not None:
        return cached
    bundle = _build_cvp_layer(problem_size, objective_only=True)
    _OBJECTIVE_LAYER_CACHE[problem_size] = bundle
    return bundle


def _dtype_from_name(dtype_name: str | torch.dtype | None) -> torch.dtype:
    if isinstance(dtype_name, torch.dtype):
        return dtype_name
    normalized = str(dtype_name or "float64").lower()
    if normalized in {"float", "float32", "torch.float32"}:
        return torch.float32
    if normalized in {"double", "float64", "torch.float64"}:
        return torch.float64
    raise ValueError(f"unsupported cvxpylayer dtype '{dtype_name}'")


def _as_tensor(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _prepare_cvp_tensors(
    depot: torch.Tensor | np.ndarray,
    ordered_targets: torch.Tensor | np.ndarray,
    carrier_speed: torch.Tensor | float,
    uav_speed: torch.Tensor | float,
    endurance: torch.Tensor | float,
    dtype: str | torch.dtype = "float64",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if torch.is_tensor(ordered_targets):
        device = ordered_targets.device
    elif torch.is_tensor(depot):
        device = depot.device
    else:
        device = torch.device("cpu")
    tensor_dtype = _dtype_from_name(dtype)

    depot_t = _as_tensor(depot, device=device, dtype=tensor_dtype)
    targets_t = _as_tensor(ordered_targets, device=device, dtype=tensor_dtype)
    if targets_t.ndim not in {2, 3} or targets_t.size(-1) != 2:
        raise ValueError(f"ordered_targets must have shape (J, 2) or (B, J, 2), got {tuple(targets_t.shape)}")
    problem_size = int(targets_t.size(-2))

    carrier_speed_t = _as_tensor(carrier_speed, device=device, dtype=tensor_dtype)
    uav_speed_t = _as_tensor(uav_speed, device=device, dtype=tensor_dtype)
    endurance_t = _as_tensor(endurance, device=device, dtype=tensor_dtype)
    if targets_t.ndim == 2:
        depot_t = depot_t.reshape(2)
        carrier_speed_t = carrier_speed_t.reshape(())
        uav_speed_t = uav_speed_t.reshape(())
        endurance_t = endurance_t.reshape(())
    else:
        batch_size = int(targets_t.size(0))
        if depot_t.numel() == 2:
            depot_t = depot_t.reshape(1, 2).expand(batch_size, 2)
        else:
            depot_t = depot_t.reshape(batch_size, 2)
        if carrier_speed_t.numel() == 1:
            carrier_speed_t = carrier_speed_t.reshape(1).expand(batch_size)
        else:
            carrier_speed_t = carrier_speed_t.reshape(batch_size)
        if uav_speed_t.numel() == 1:
            uav_speed_t = uav_speed_t.reshape(1).expand(batch_size)
        else:
            uav_speed_t = uav_speed_t.reshape(batch_size)
        if endurance_t.numel() == 1:
            endurance_t = endurance_t.reshape(1).expand(batch_size)
        else:
            endurance_t = endurance_t.reshape(batch_size)
    return depot_t, targets_t, carrier_speed_t, uav_speed_t, endurance_t, problem_size


def solve_ordered_targets_torch(
    depot: torch.Tensor | np.ndarray,
    ordered_targets: torch.Tensor | np.ndarray,
    carrier_speed: torch.Tensor | float,
    uav_speed: torch.Tensor | float,
    endurance: torch.Tensor | float,
    solver_args: dict[str, Any] | None = None,
    dtype: str | torch.dtype = "float64",
) -> dict[str, torch.Tensor]:
    """Differentiable fixed-tour CVP layer.

    `ordered_targets` is the already ordered target coordinate tensor with shape (J, 2)
    or a batched tensor with shape (B, J, 2).
    Gradients can flow to tensor parameters that require gradients. A hard permutation
    index is still non-differentiable; use a soft ordered-target tensor if the upper
    policy needs direct pathwise gradients.
    """
    depot_t, targets_t, carrier_speed_t, uav_speed_t, endurance_t, problem_size = _prepare_cvp_tensors(
        depot=depot,
        ordered_targets=ordered_targets,
        carrier_speed=carrier_speed,
        uav_speed=uav_speed,
        endurance=endurance,
        dtype=dtype,
    )

    layer = get_cvp_layer(problem_size).layer
    merged_solver_args = dict(DEFAULT_SOLVER_ARGS)
    if solver_args:
        merged_solver_args.update(solver_args)

    takeoff, landing, t1, t2, tau, tseg = layer(
        depot_t,
        targets_t,
        carrier_speed_t,
        uav_speed_t,
        endurance_t,
        solver_args=merged_solver_args,
    )
    if targets_t.ndim == 2:
        objective = tau.sum() + tseg.sum()
    else:
        objective = tau.sum(dim=-1) + tseg.sum(dim=-1)
    return {
        "objective": objective,
        "makespan": objective,
        "takeoff_points": takeoff,
        "landing_points": landing,
        "t1": t1,
        "t2": t2,
        "tau": tau,
        "Tseg": tseg,
    }


def solve_ordered_targets_objective_torch(
    depot: torch.Tensor | np.ndarray,
    ordered_targets: torch.Tensor | np.ndarray,
    carrier_speed: torch.Tensor | float,
    uav_speed: torch.Tensor | float,
    endurance: torch.Tensor | float,
    solver_args: dict[str, Any] | None = None,
    dtype: str | torch.dtype = "float64",
) -> torch.Tensor:
    """Fixed-tour CVP objective-only layer for high-throughput hard rewards."""
    depot_t, targets_t, carrier_speed_t, uav_speed_t, endurance_t, problem_size = _prepare_cvp_tensors(
        depot=depot,
        ordered_targets=ordered_targets,
        carrier_speed=carrier_speed,
        uav_speed=uav_speed,
        endurance=endurance,
        dtype=dtype,
    )
    layer = get_cvp_objective_layer(problem_size).layer
    merged_solver_args = dict(DEFAULT_SOLVER_ARGS)
    if solver_args:
        merged_solver_args.update(solver_args)

    tau, tseg = layer(
        depot_t,
        targets_t,
        carrier_speed_t,
        uav_speed_t,
        endurance_t,
        solver_args=merged_solver_args,
    )
    if targets_t.ndim == 2:
        return tau.sum() + tseg.sum()
    return tau.sum(dim=-1) + tseg.sum(dim=-1)


class CVPObjectiveLayer(nn.Module):
    """Torch module wrapper for inserting the fixed-tour CVP layer into a model."""

    def __init__(
        self,
        solver_args: dict[str, Any] | None = None,
        dtype: str | torch.dtype = "float64",
    ):
        super().__init__()
        self.solver_args = dict(solver_args or DEFAULT_SOLVER_ARGS)
        self.dtype = dtype

    def forward(
        self,
        depot: torch.Tensor,
        ordered_targets: torch.Tensor,
        carrier_speed: torch.Tensor | float,
        uav_speed: torch.Tensor | float,
        endurance: torch.Tensor | float,
    ) -> torch.Tensor:
        result = solve_ordered_targets_torch(
            depot=depot,
            ordered_targets=ordered_targets,
            carrier_speed=carrier_speed,
            uav_speed=uav_speed,
            endurance=endurance,
            solver_args=self.solver_args,
            dtype=self.dtype,
        )
        return result["objective"]


def solve_fixed_sequence_numpy(
    depot: np.ndarray,
    targets: np.ndarray,
    sequence: list[int],
    carrier_speed: float,
    uav_speed: float,
    endurance: float,
    solver_args: dict[str, Any] | None = None,
    dtype: str | torch.dtype = "float64",
) -> dict[str, Any]:
    ordered_targets = np.asarray(targets, dtype=float)[list(sequence)]
    with torch.enable_grad():
        result_t = solve_ordered_targets_torch(
            depot=np.asarray(depot, dtype=float),
            ordered_targets=ordered_targets,
            carrier_speed=float(carrier_speed),
            uav_speed=float(uav_speed),
            endurance=float(endurance),
            solver_args=solver_args,
            dtype=dtype,
        )
    return {
        "obj": float(result_t["objective"].detach().cpu().item()),
        "status": "CVXPYLAYER",
        "sx": result_t["takeoff_points"].detach().cpu().numpy(),
        "lx": result_t["landing_points"].detach().cpu().numpy(),
        "t1": result_t["t1"].detach().cpu().numpy(),
        "t2": result_t["t2"].detach().cpu().numpy(),
        "tau": result_t["tau"].detach().cpu().numpy(),
        "Tseg": result_t["Tseg"].detach().cpu().numpy(),
        "solver_args": dict(DEFAULT_SOLVER_ARGS, **(solver_args or {})),
    }


def solve_fixed_sequence_objective_numpy(
    depot: np.ndarray,
    targets: np.ndarray,
    sequence: list[int],
    carrier_speed: float,
    uav_speed: float,
    endurance: float,
    solver_args: dict[str, Any] | None = None,
    dtype: str | torch.dtype = "float64",
) -> dict[str, Any]:
    ordered_targets = np.asarray(targets, dtype=float)[list(sequence)]
    with torch.no_grad():
        objective = solve_ordered_targets_objective_torch(
            depot=np.asarray(depot, dtype=float),
            ordered_targets=ordered_targets,
            carrier_speed=float(carrier_speed),
            uav_speed=float(uav_speed),
            endurance=float(endurance),
            solver_args=solver_args,
            dtype=dtype,
        )
    return {
        "obj": float(objective.detach().cpu().item()),
        "status": "CVXPYLAYER_OBJECTIVE",
        "solver_args": dict(DEFAULT_SOLVER_ARGS, **(solver_args or {})),
    }
