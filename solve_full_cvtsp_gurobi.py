from __future__ import annotations

import argparse
import json
from pathlib import Path

from CVTSP_SOCP import gurobi_cvp_socp
from src.FullCVTSPSolver import FullCVTSPConfig, solve_full_cvtsp_gurobi
from src.TSProblemDef import load_instance


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Solve the complete CVTSP jointly with Gurobi using the MISOCP "
            "formulation from Li, Zhou, and Cote (2025)."
        )
    )
    parser.add_argument("instance", help="path to one CVTSP instance text file")
    parser.add_argument(
        "--formulation",
        choices=["basic", "enhanced"],
        default="enhanced",
        help="basic: Equations (1)-(13); enhanced: Model+ Equations (1)-(28)",
    )
    parser.add_argument("--time-limit", type=float, default=3600.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--mip-gap", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--numeric-focus", type=int, default=1)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--log-file")
    parser.add_argument("--write-model", help="optional .lp or .mps output path")
    parser.add_argument(
        "--output-json",
        help="result JSON path; defaults to outputs/full_gurobi_baseline/<instance>.json",
    )
    parser.add_argument(
        "--verify-fixed-route",
        action="store_true",
        help="re-solve the incumbent route as a fixed-tour CVP after timing",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    instance = load_instance(args.instance)

    output_json = Path(
        args.output_json
        or Path("outputs")
        / "full_gurobi_baseline"
        / f"{instance.instance_id}_{args.formulation}.json"
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)

    log_file = args.log_file
    if log_file is None and not args.quiet:
        log_file = str(output_json.with_suffix(".log"))

    config = FullCVTSPConfig(
        formulation=args.formulation,
        time_limit=args.time_limit,
        threads=args.threads,
        mip_gap=args.mip_gap,
        output_flag=0 if args.quiet else 1,
        log_file=log_file,
        seed=args.seed,
        numeric_focus=args.numeric_focus,
        write_model=args.write_model,
    )

    print(
        f"Solving {instance.instance_id}: J={instance.J}, "
        f"formulation={args.formulation}, time_limit={args.time_limit}s, "
        f"threads={args.threads}"
    )
    result = solve_full_cvtsp_gurobi(instance, config)
    payload = result.to_dict()

    if args.verify_fixed_route and result.success:
        verification = gurobi_cvp_socp(
            depot=instance.depot,
            targets=instance.targets,
            tour=result.target_sequence_0based,
            Vc=instance.carrier_speed,
            Vv=instance.uav_speed,
            endurance_a=instance.endurance,
            threads=args.threads,
            test=True,
            output_flag=0,
        )
        fixed_objective = float(verification["obj"])
        payload["fixed_route_verification"] = {
            "objective": fixed_objective,
            "absolute_difference": abs(fixed_objective - float(result.objective)),
        }

    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"status={result.status}")
    print(f"objective={result.objective}")
    print(f"best_bound={result.best_bound}")
    print(f"mip_gap={result.mip_gap}")
    print(f"build_time={result.build_time:.6f}s")
    print(f"gurobi_solve_time={result.solve_time:.6f}s")
    print(f"total_wall_time={result.total_wall_time:.6f}s")
    print(f"node_count={result.node_count:.0f}")
    print(f"route={result.route_with_depot}")
    print(f"result_json={output_json.resolve()}")
    if "fixed_route_verification" in payload:
        verification = payload["fixed_route_verification"]
        print(
            "fixed_route_objective="
            f"{verification['objective']}, "
            "difference="
            f"{verification['absolute_difference']}"
        )


if __name__ == "__main__":
    main()
