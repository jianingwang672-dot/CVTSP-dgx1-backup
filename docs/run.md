# Run Guide

## Dependencies
```bash
python -m pip install -r requirements.txt
```

需要保证以下依赖可用：
- `torch`
- `gurobipy`
- `cvxpy`
- `cvxpylayers`
- `diffcp`
- `numpy`
- `pandas`
- `PyYAML`
- `loguru`

Solver backend:
- `train_n100.py` / `test_n100.py` default to `solver_backend="cvxpylayer"`.
- Set `solver_backend="gurobi"` in the parameter dict or config to switch back.

REBAR-style cvxpylayer training:
- `cvxpylayer_rebar_loss_enable=True` uses hard-route makespan as the main target.
- `C_hard` comes from the sampled POMO route reward.
- `C_soft` and `C_cond` are cvxpylayer objectives on relaxed routes.
- The loss combines a hard-route policy-gradient term and a differentiable
  cvxpylayer control-variate term, avoiding direct `loss=f(soft_route)` training.
- `cvxpylayer_rebar_eta`, `cvxpylayer_rebar_temperature`, and
  `cvxpylayer_rebar_conditional_bias` control the strength and smoothness.

## Full Gurobi CVTSP baseline

`solve_full_cvtsp_gurobi.py` jointly optimizes the target order, take-off and
landing points, UAV flight times, and carrier travel times. It implements the
MISOCP from Li, Zhou, and Cote (2025):

- `--formulation basic`: Equations (1)-(13)
- `--formulation enhanced`: Model+, Equations (1)-(28), default

Example:

```bash
python3 solve_full_cvtsp_gurobi.py instance/Data/Example_1.txt \
  --formulation enhanced \
  --time-limit 3600 \
  --threads 1 \
  --mip-gap 1e-6 \
  --output-json outputs/full_gurobi_baseline/Example_1.json
```

The JSON result records model build time, Gurobi solve time, total wall time,
incumbent objective, best bound, MIP gap, node count, and the complete route.
Use `--verify-fixed-route` to re-solve the returned route with the existing
fixed-tour CVP model after the baseline timer has stopped.

## Train
```bash
python -m src.TSPTrainer \
  --output-dir outputs/train_run
```

训练输出：
- `run.log`
- `resolved_config.yaml`
- `metrics.csv`
- `best.pt`
- `last.pt`
- `train_summary.json`

## Evaluate One Real Instance
```bash
python -m src.TSPTester \
  --checkpoint-path outputs/train_run/best.pt \
  --instance-path instance/Data/Example_1.txt \
  --output-dir outputs/eval_one
```

## Evaluate Real Test Split
```bash
python -m src.TSPTester \
  --checkpoint-path outputs/train_run/best.pt \
  --split test \
  --output-dir outputs/eval_test
```

## Infer One Real Instance
```bash
python -m src.TSPTester \
  --checkpoint-path outputs/train_run/best.pt \
  --instance-path instance/Data/Example_1.txt \
  --output-dir outputs/infer_one
```

## Notes
- 当前仓库只保留真实 `529` 个实例。
- 上层为 DRL sequence policy，下层为 exact Gurobi solver。
- `sample` 模式会生成多个 candidate，再由 exact solver 选最优。
