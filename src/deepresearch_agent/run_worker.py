from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass

from deepresearch_agent.run_control import RunController


@dataclass
class WorkerLoopSummary:
    processed_count: int
    idle_polls: int
    last_run_id: str | None
    stopped_reason: str

    def as_dict(self) -> dict[str, int | str | None]:
        return {
            "processed_count": self.processed_count,
            "idle_polls": self.idle_polls,
            "last_run_id": self.last_run_id,
            "stopped_reason": self.stopped_reason,
        }


async def run_worker_loop(
    *,
    controller: RunController | None = None,
    poll_interval_seconds: float = 1.0,
    max_runs: int | None = None,
    idle_exit: bool = False,
) -> WorkerLoopSummary:
    selected_controller = controller or RunController()
    processed_count = 0
    idle_polls = 0
    last_run_id: str | None = None
    stopped_reason = "interrupted"
    if max_runs is not None and max_runs <= 0:
        return WorkerLoopSummary(
            processed_count=0,
            idle_polls=0,
            last_run_id=None,
            stopped_reason="max_runs",
        )

    while True:
        run = await selected_controller.process_next_queued()
        if run is None:
            idle_polls += 1
            if idle_exit:
                stopped_reason = "idle"
                break
            await asyncio.sleep(poll_interval_seconds)
            continue

        processed_count += 1
        idle_polls = 0
        last_run_id = run.run_id
        if max_runs is not None and processed_count >= max_runs:
            stopped_reason = "max_runs"
            break

    return WorkerLoopSummary(
        processed_count=processed_count,
        idle_polls=idle_polls,
        last_run_id=last_run_id,
        stopped_reason=stopped_reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local queued-run worker.")
    parser.add_argument(
        "--poll-interval-seconds",
        type=float,
        default=1.0,
        help="Seconds to sleep when no queued run is available.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Stop after processing this many runs. Omit to run until interrupted.",
    )
    parser.add_argument(
        "--idle-exit",
        action="store_true",
        help="Exit after one idle poll instead of waiting forever.",
    )
    parser.add_argument("--json", action="store_true", help="Print a JSON summary.")
    args = parser.parse_args()

    try:
        summary = asyncio.run(
            run_worker_loop(
                poll_interval_seconds=max(args.poll_interval_seconds, 0.0),
                max_runs=args.max_runs,
                idle_exit=args.idle_exit,
            )
        )
    except KeyboardInterrupt:
        summary = WorkerLoopSummary(
            processed_count=0,
            idle_polls=0,
            last_run_id=None,
            stopped_reason="interrupted",
        )

    if args.json:
        print(json.dumps(summary.as_dict(), ensure_ascii=False, indent=2))
    else:
        print(
            "Worker stopped: "
            f"{summary.stopped_reason}; processed={summary.processed_count}; "
            f"idle_polls={summary.idle_polls}; last_run_id={summary.last_run_id}"
        )


if __name__ == "__main__":
    main()
