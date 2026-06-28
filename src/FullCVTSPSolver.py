from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.TSProblemDef import CVTSPInstance


@dataclass(frozen=True)
class FullCVTSPConfig:
    formulation: str = "enhanced"
    time_limit: float | None = 3600.0
    threads: int = 1
    mip_gap: float = 1e-6
    output_flag: int = 1
    log_file: str | None = None
    seed: int = 1234
    numeric_focus: int = 1
    write_model: str | None = None


@dataclass
class FullCVTSPResult:
    instance_id: str
    status: str
    status_code: int
    success: bool
    optimal: bool
    objective: float | None
    best_bound: float | None
    mip_gap: float | None
    build_time: float
    solve_time: float
    total_wall_time: float
    node_count: float
    solution_count: int
    model_variables: int
    model_constraints: int
    model_general_constraints: int
    route_with_depot: list[int]
    target_sequence_0based: list[int]
    target_sequence_1based: list[int]
    selected_arcs: list[tuple[int, int]]
    takeoff_points: list[list[float]]
    landing_points: list[list[float]]
    outbound_times: list[float]
    return_times: list[float]
    carrier_arc_times: list[dict[str, float | int]]
    config: dict[str, Any]
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _status_name(status: int, grb: Any) -> str:
    names = {
        grb.LOADED: "LOADED",
        grb.OPTIMAL: "OPTIMAL",
        grb.INFEASIBLE: "INFEASIBLE",
        grb.INF_OR_UNBD: "INF_OR_UNBD",
        grb.UNBOUNDED: "UNBOUNDED",
        grb.CUTOFF: "CUTOFF",
        grb.ITERATION_LIMIT: "ITERATION_LIMIT",
        grb.NODE_LIMIT: "NODE_LIMIT",
        grb.TIME_LIMIT: "TIME_LIMIT",
        grb.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        grb.INTERRUPTED: "INTERRUPTED",
        grb.NUMERIC: "NUMERIC",
        grb.SUBOPTIMAL: "SUBOPTIMAL",
        grb.INPROGRESS: "INPROGRESS",
        grb.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        grb.WORK_LIMIT: "WORK_LIMIT",
        grb.MEM_LIMIT: "MEM_LIMIT",
    }
    return names.get(status, f"STATUS_{status}")


def _add_norm(
    model: Any,
    component_x: Any,
    component_y: Any,
    name: str,
    grb: Any,
) -> Any:
    dx = model.addVar(lb=-grb.INFINITY, name=f"{name}_dx")
    dy = model.addVar(lb=-grb.INFINITY, name=f"{name}_dy")
    distance = model.addVar(lb=0.0, name=f"{name}_norm")
    model.addConstr(dx == component_x, name=f"{name}_dx_def")
    model.addConstr(dy == component_y, name=f"{name}_dy_def")
    model.addGenConstrNorm(distance, [dx, dy], 2.0, name=f"{name}_soc")
    return distance


def _route_from_arcs(selected_arcs: list[tuple[int, int]], target_count: int) -> list[int]:
    successor = {i: j for i, j in selected_arcs}
    route = [0]
    current = 0
    for _ in range(target_count + 1):
        if current not in successor:
            break
        current = successor[current]
        route.append(current)
        if current == 0:
            break
    return route


def solve_full_cvtsp_gurobi(
    instance: CVTSPInstance,
    config: FullCVTSPConfig | None = None,
) -> FullCVTSPResult:
    """Solve the complete CVTSP as the paper's MISOCP Model or Model+.

    The routing sequence and all continuous take-off/landing decisions are
    optimized jointly by Gurobi. Node 0 is the depot and targets are 1..n.
    """
    config = config or FullCVTSPConfig()
    formulation = config.formulation.strip().lower()
    if formulation not in {"basic", "enhanced", "model", "model+"}:
        raise ValueError("formulation must be 'basic' or 'enhanced'")
    enhanced = formulation in {"enhanced", "model+"}

    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise ImportError("gurobipy is required for the full CVTSP baseline") from exc

    n = int(instance.J)
    if n < 2:
        raise ValueError("the complete CVTSP model requires at least two targets")

    carrier_speed = float(instance.carrier_speed)
    vehicle_speed = float(instance.uav_speed)
    endurance = float(instance.endurance)
    if carrier_speed <= 0 or vehicle_speed <= 0 or endurance <= 0:
        raise ValueError("speeds and endurance must be positive")

    coords = np.vstack(
        [
            np.asarray(instance.depot, dtype=float).reshape(1, 2),
            np.asarray(instance.targets, dtype=float).reshape(n, 2),
        ]
    )
    nodes = list(range(n + 1))
    targets = list(range(1, n + 1))
    arcs = [(i, j) for i in nodes for j in nodes if i != j]
    target_arcs = [(i, j) for i in targets for j in targets if i != j]
    distances = {
        (i, j): float(np.linalg.norm(coords[i] - coords[j]))
        for i, j in arcs
    }

    start_wall = time.perf_counter()
    model = gp.Model(f"full_cvtsp_{instance.instance_id}")
    model.Params.OutputFlag = int(config.output_flag)
    model.Params.Threads = max(int(config.threads), 1)
    model.Params.MIPGap = max(float(config.mip_gap), 0.0)
    model.Params.Seed = int(config.seed)
    model.Params.NumericFocus = int(config.numeric_focus)
    if config.time_limit is not None:
        model.Params.TimeLimit = max(float(config.time_limit), 0.0)
    if config.log_file:
        log_path = Path(config.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        model.Params.LogFile = str(log_path)

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    carrier_time = model.addVars(arcs, lb=0.0, name="T")
    order = model.addVars(targets, lb=1.0, ub=float(n), name="u")

    max_target_offset = 0.5 * endurance * (carrier_speed + vehicle_speed)
    takeoff_x = {}
    takeoff_y = {}
    landing_x = {}
    landing_y = {}
    for j in targets:
        px, py = coords[j]
        takeoff_x[j] = model.addVar(
            lb=px - max_target_offset,
            ub=px + max_target_offset,
            name=f"sx[{j}]",
        )
        takeoff_y[j] = model.addVar(
            lb=py - max_target_offset,
            ub=py + max_target_offset,
            name=f"sy[{j}]",
        )
        landing_x[j] = model.addVar(
            lb=px - max_target_offset,
            ub=px + max_target_offset,
            name=f"lx[{j}]",
        )
        landing_y[j] = model.addVar(
            lb=py - max_target_offset,
            ub=py + max_target_offset,
            name=f"ly[{j}]",
        )

    time_upper = max_target_offset / vehicle_speed
    outbound_time = model.addVars(targets, lb=0.0, ub=time_upper, name="t1")
    return_time = model.addVars(targets, lb=0.0, ub=time_upper, name="t2")
    makespan = model.addVar(lb=0.0, name="Cmax")

    model.setObjective(makespan, GRB.MINIMIZE)

    # Equations (2)-(7): objective composition and second-order cones.
    model.addConstr(
        makespan
        >= gp.quicksum(outbound_time[j] + return_time[j] for j in targets)
        + gp.quicksum(carrier_time[i, j] for i, j in arcs),
        name="makespan_definition",
    )

    for j in targets:
        px, py = coords[j]
        outbound_distance = _add_norm(
            model,
            takeoff_x[j] - px,
            takeoff_y[j] - py,
            f"vehicle_out[{j}]",
            GRB,
        )
        model.addConstr(
            outbound_distance <= vehicle_speed * outbound_time[j],
            name=f"vehicle_out_time[{j}]",
        )

        return_distance = _add_norm(
            model,
            landing_x[j] - px,
            landing_y[j] - py,
            f"vehicle_return[{j}]",
            GRB,
        )
        model.addConstr(
            return_distance <= vehicle_speed * return_time[j],
            name=f"vehicle_return_time[{j}]",
        )

        sync_distance = _add_norm(
            model,
            landing_x[j] - takeoff_x[j],
            landing_y[j] - takeoff_y[j],
            f"sync[{j}]",
            GRB,
        )
        model.addConstr(
            sync_distance
            <= carrier_speed * (outbound_time[j] + return_time[j]),
            name=f"sync_time[{j}]",
        )
        model.addConstr(
            outbound_time[j] + return_time[j] <= endurance,
            name=f"endurance[{j}]",
        )

    for i, j in arcs:
        if i == 0:
            from_x, from_y = float(coords[0, 0]), float(coords[0, 1])
        else:
            from_x, from_y = landing_x[i], landing_y[i]
        if j == 0:
            to_x, to_y = float(coords[0, 0]), float(coords[0, 1])
        else:
            to_x, to_y = takeoff_x[j], takeoff_y[j]

        segment_distance = _add_norm(
            model,
            to_x - from_x,
            to_y - from_y,
            f"carrier_arc[{i},{j}]",
            GRB,
        )
        endurance_factor = 0.5 if i == 0 or j == 0 else 1.0
        big_m = distances[i, j] + endurance_factor * endurance * (
            carrier_speed + vehicle_speed
        )
        model.addConstr(
            segment_distance
            <= carrier_speed * carrier_time[i, j] + big_m * (1.0 - x[i, j]),
            name=f"carrier_arc_time[{i},{j}]",
        )

    # Equations (8)-(11): Hamiltonian cycle and improved MTZ constraints.
    for i in nodes:
        model.addConstr(
            gp.quicksum(x[i, j] for j in nodes if j != i) == 1,
            name=f"one_successor[{i}]",
        )
    for j in nodes:
        model.addConstr(
            gp.quicksum(x[i, j] for i in nodes if i != j) == 1,
            name=f"one_predecessor[{j}]",
        )
    for i, j in target_arcs:
        model.addConstr(
            order[i] - order[j] + n * x[i, j] + (n - 2) * x[j, i]
            <= n - 1,
            name=f"improved_mtz[{i},{j}]",
        )

    if enhanced:
        half_reach = 0.5 * endurance * (carrier_speed + vehicle_speed)
        full_reach = endurance * (carrier_speed + vehicle_speed)

        # Equations (14)-(21): arc lower/upper bounds and flight-time bounds.
        for i, j in arcs:
            predecessor_return = 0.0 if i == 0 else return_time[i]
            successor_outbound = 0.0 if j == 0 else outbound_time[j]
            model.addConstr(
                vehicle_speed * predecessor_return
                + vehicle_speed * successor_outbound
                + carrier_speed * carrier_time[i, j]
                >= distances[i, j] * x[i, j],
                name=f"arc_distance_lb[{i},{j}]",
            )

            reach = half_reach if i == 0 or j == 0 else full_reach
            lower_distance = max(0.0, distances[i, j] - reach)
            upper_distance = distances[i, j] + reach
            model.addConstr(
                carrier_speed * carrier_time[i, j]
                >= lower_distance * x[i, j],
                name=f"carrier_time_lb[{i},{j}]",
            )
            model.addConstr(
                carrier_speed * carrier_time[i, j]
                <= upper_distance * x[i, j],
                name=f"carrier_time_ub[{i},{j}]",
            )

        flight_time_bound = (
            endurance * (carrier_speed + vehicle_speed) / (2.0 * vehicle_speed)
        )
        for j in targets:
            model.addConstr(
                vehicle_speed * outbound_time[j]
                <= vehicle_speed * return_time[j]
                + carrier_speed * (outbound_time[j] + return_time[j]),
                name=f"flight_balance_out[{j}]",
            )
            model.addConstr(
                vehicle_speed * return_time[j]
                <= vehicle_speed * outbound_time[j]
                + carrier_speed * (outbound_time[j] + return_time[j]),
                name=f"flight_balance_return[{j}]",
            )
            model.addConstr(
                outbound_time[j] <= flight_time_bound,
                name=f"outbound_time_ub[{j}]",
            )
            model.addConstr(
                return_time[j] <= flight_time_bound,
                name=f"return_time_ub[{j}]",
            )

        # Equations (22)-(25): route-dependent lower bound beta.
        beta = {}
        for i, j in arcs:
            reach = half_reach if i == 0 or j == 0 else full_reach
            beta[i, j] = max(
                (distances[i, j] - reach) / carrier_speed
                + reach / vehicle_speed,
                distances[i, j] / vehicle_speed,
            )
        model.addConstr(
            makespan >= gp.quicksum(beta[i, j] * x[i, j] for i, j in arcs),
            name="beta_makespan_lb",
        )

        # Equation (27): CVP lower bound based on the selected TSP arcs.
        selected_distance = gp.quicksum(
            distances[i, j] * x[i, j] for i, j in arcs
        )
        model.addConstr(
            makespan
            >= selected_distance / carrier_speed
            - n * vehicle_speed * endurance / carrier_speed
            + n * endurance,
            name="cvp_makespan_lb",
        )

        # Equation (28): full endurance for sufficiently distant neighbors.
        for j in targets:
            phi_j = [
                i
                for i in nodes
                if i != j
                and (
                    (i == 0 and distances[0, j] >= half_reach)
                    or (i != 0 and distances[i, j] >= full_reach)
                )
            ]
            if not phi_j:
                continue
            model.addConstr(
                outbound_time[j] + return_time[j]
                >= endurance
                * (
                    gp.quicksum(x[i, j] for i in phi_j)
                    + gp.quicksum(x[j, k] for k in phi_j)
                    - 1.0
                ),
                name=f"full_endurance[{j}]",
            )

    model.update()
    if config.write_model:
        model_path = Path(config.write_model)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model.write(str(model_path))
    build_time = time.perf_counter() - start_wall

    model.optimize()
    total_wall_time = time.perf_counter() - start_wall

    status_code = int(model.Status)
    status = _status_name(status_code, GRB)
    solution_count = int(model.SolCount)
    has_solution = solution_count > 0
    optimal = status_code == GRB.OPTIMAL

    objective = float(model.ObjVal) if has_solution else None
    best_bound = (
        float(model.ObjBound)
        if status_code not in {GRB.INFEASIBLE, GRB.INF_OR_UNBD, GRB.UNBOUNDED}
        else None
    )
    mip_gap = float(model.MIPGap) if has_solution else None

    selected_arcs = (
        sorted((i, j) for i, j in arcs if x[i, j].X > 0.5)
        if has_solution
        else []
    )
    route = _route_from_arcs(selected_arcs, n) if has_solution else []
    target_sequence_1based = (
        [node for node in route[1:-1] if node != 0]
        if route and route[-1] == 0
        else [node for node in route[1:] if node != 0]
    )
    target_sequence_0based = [node - 1 for node in target_sequence_1based]

    takeoff_points = (
        [[float(takeoff_x[j].X), float(takeoff_y[j].X)] for j in targets]
        if has_solution
        else []
    )
    landing_points = (
        [[float(landing_x[j].X), float(landing_y[j].X)] for j in targets]
        if has_solution
        else []
    )
    outbound_values = (
        [float(outbound_time[j].X) for j in targets] if has_solution else []
    )
    return_values = (
        [float(return_time[j].X) for j in targets] if has_solution else []
    )
    carrier_arc_values = (
        [
            {
                "from": int(i),
                "to": int(j),
                "time": float(carrier_time[i, j].X),
            }
            for i, j in selected_arcs
        ]
        if has_solution
        else []
    )

    message = ""
    if has_solution and len(target_sequence_0based) != n:
        message = "incumbent arcs did not produce a complete depot-to-depot route"

    result = FullCVTSPResult(
        instance_id=instance.instance_id,
        status=status,
        status_code=status_code,
        success=has_solution,
        optimal=optimal,
        objective=objective,
        best_bound=best_bound,
        mip_gap=mip_gap,
        build_time=float(build_time),
        solve_time=float(model.Runtime),
        total_wall_time=float(total_wall_time),
        node_count=float(model.NodeCount),
        solution_count=solution_count,
        model_variables=int(model.NumVars),
        model_constraints=int(model.NumConstrs),
        model_general_constraints=int(model.NumGenConstrs),
        route_with_depot=route,
        target_sequence_0based=target_sequence_0based,
        target_sequence_1based=target_sequence_1based,
        selected_arcs=selected_arcs,
        takeoff_points=takeoff_points,
        landing_points=landing_points,
        outbound_times=outbound_values,
        return_times=return_values,
        carrier_arc_times=carrier_arc_values,
        config=asdict(config),
        message=message,
    )
    model.dispose()
    return result
