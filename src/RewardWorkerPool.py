from __future__ import annotations

import multiprocessing as mp
from dataclasses import dataclass
from queue import Empty
from typing import Any

from src.CVPSolver import SPSolution, SolverConfig, solve


@dataclass
class RewardTask:
    task_id: int
    instance: Any
    sequence: list[int]
    solver_config: SolverConfig


@dataclass
class RewardResult:
    task_id: int
    instance_id: str
    sequence_key: tuple[int, ...]
    solution: SPSolution


@dataclass
class WorkerFailure:
    worker_index: int
    error: str


def _worker_main(
    worker_index: int,
    request_queue: Any,
    result_queue: Any,
    env_output_flag: int,
) -> None:
    env = None
    try:
        while True:
            task = request_queue.get()
            if task is None:
                break
            try:
                if str(task.solver_config.solver_backend).strip().lower() == "gurobi" and env is None:
                    import gurobipy as gp

                    env = gp.Env(params={"OutputFlag": int(env_output_flag)})
                solution = solve(task.instance, task.sequence, task.solver_config, env=env)
                result_queue.put(
                    RewardResult(
                        task_id=int(task.task_id),
                        instance_id=str(task.instance.instance_id),
                        sequence_key=tuple(int(node) for node in task.sequence),
                        solution=solution,
                    )
                )
            except Exception as exc:  # pragma: no cover
                result_queue.put(WorkerFailure(worker_index=worker_index, error=repr(exc)))
                break
    except Exception as exc:  # pragma: no cover
        result_queue.put(WorkerFailure(worker_index=worker_index, error=repr(exc)))
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


class RewardWorkerPool:
    def __init__(
        self,
        worker_count: int,
        env_output_flag: int = 0,
        queue_size: int | None = None,
        poll_timeout: float = 5.0,
    ):
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        self.worker_count = int(worker_count)
        self.env_output_flag = int(env_output_flag)
        self.poll_timeout = float(poll_timeout)
        self._ctx = mp.get_context("spawn")
        maxsize = int(queue_size) if queue_size is not None else max(self.worker_count * 4, 1)
        self._request_queue = self._ctx.Queue(maxsize=maxsize)
        self._result_queue = self._ctx.Queue()
        self._workers: list[mp.Process] = []
        self._closed = False
        self._start_workers()

    def _start_workers(self) -> None:
        for worker_index in range(self.worker_count):
            worker = self._ctx.Process(
                target=_worker_main,
                args=(
                    worker_index,
                    self._request_queue,
                    self._result_queue,
                    self.env_output_flag,
                ),
                daemon=True,
            )
            worker.start()
            self._workers.append(worker)

    def _raise_if_worker_failed(self) -> None:
        dead_workers = [worker for worker in self._workers if not worker.is_alive()]
        if dead_workers:
            details = ", ".join(
                f"pid={worker.pid}, exitcode={worker.exitcode}"
                for worker in dead_workers
            )
            raise RuntimeError(f"persistent reward worker stopped unexpectedly: {details}")

    def map(self, tasks: list[tuple[Any, list[int], SolverConfig]]) -> list[tuple[str, tuple[int, ...], SPSolution]]:
        if self._closed:
            raise RuntimeError("reward worker pool is already closed")
        task_count = len(tasks)
        if task_count == 0:
            return []

        for task_id, (instance, sequence, solver_config) in enumerate(tasks):
            self._request_queue.put(
                RewardTask(
                    task_id=task_id,
                    instance=instance,
                    sequence=list(sequence),
                    solver_config=solver_config,
                )
            )

        results: list[tuple[str, tuple[int, ...], SPSolution] | None] = [None] * task_count
        remaining = task_count
        while remaining > 0:
            try:
                item = self._result_queue.get(timeout=self.poll_timeout)
            except Empty:
                self._raise_if_worker_failed()
                continue

            if isinstance(item, WorkerFailure):
                raise RuntimeError(f"persistent reward worker failed: {item.error}")

            if not isinstance(item, RewardResult):
                raise RuntimeError(f"unexpected reward worker message: {item!r}")

            results[item.task_id] = (item.instance_id, item.sequence_key, item.solution)
            remaining -= 1

        return [result for result in results if result is not None]

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        for _ in self._workers:
            try:
                self._request_queue.put_nowait(None)
            except Exception:
                pass

        for worker in self._workers:
            worker.join(timeout=2.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2.0)

        try:
            self._request_queue.close()
        except Exception:
            pass
        try:
            self._result_queue.close()
        except Exception:
            pass

