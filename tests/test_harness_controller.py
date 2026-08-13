from __future__ import annotations

from threading import Event
import time
import unittest
from typing import Any

from teaching_skill_miner.harness import (
    CancellationToken,
    FollowUpQueue,
    HarnessLimits,
    HarnessModelResponse,
    HarnessRunHandle,
    RetryPolicy,
    SteeringQueue,
    ToolRegistry,
    run_agent_harness,
)


class _BlockingThenFinalModel:
    def __init__(self) -> None:
        self.entered = Event()
        self.release = Event()
        self.calls = 0
        self.requests: list[Any] = []

    def plan(
        self,
        request: Any,
        *,
        cancellation_token: CancellationToken,
        deadline_monotonic: float,
    ) -> HarnessModelResponse:
        del deadline_monotonic
        self.calls += 1
        self.requests.append(request)
        if self.calls == 1:
            self.entered.set()
            while not self.release.wait(0.01):
                cancellation_token.raise_if_cancelled()
            return HarnessModelResponse(
                kind="final", output={"message": "stale answer"}
            )
        return HarnessModelResponse(
            kind="final", output={"message": "steered answer"}
        )


def _run_kwargs() -> dict[str, Any]:
    return {
        "limits": HarnessLimits(deadline_seconds=5.0),
        "retry_policy": RetryPolicy(max_attempts=1),
        "allowed_permissions": {"tool.read"},
    }


class HarnessControllerTests(unittest.TestCase):
    def test_late_steer_supersedes_uncommitted_final_output(self) -> None:
        model = _BlockingThenFinalModel()
        steering = SteeringQueue()
        handle = HarnessRunHandle(
            run_id="run_steer",
            turn_id="turn_steer",
            steering_queue=steering,
        )

        def target() -> dict[str, Any]:
            cancel_handle = handle.cancellation_token
            return run_agent_harness(
                model,
                ToolRegistry(),
                {"mode": "chat"},
                run_id=handle.run_id,
                turn_id=handle.turn_id,
                cancellation_token=cancel_handle,
                event_sink=handle.event_sink,
                steering_source=steering.drain,
                **_run_kwargs(),
            )

        handle.start(target)
        self.assertTrue(model.entered.wait(1.0))
        handle.steer("请改为先给一个例子", input_id="steer_1")
        model.release.set()
        result = handle.wait(2.0)

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["output"]["message"], "steered answer")
        self.assertEqual(model.calls, 2)
        observations = model.requests[1].observations
        self.assertTrue(
            any(
                item.get("kind") == "steer"
                and item.get("content") == "请改为先给一个例子"
                for item in observations
            )
        )
        event_types = [item["type"] for item in result["events"]]
        self.assertIn("input.steered", event_types)
        self.assertIn("action.superseded", event_types)
        self.assertEqual(event_types.count("action.completed"), 1)

    def test_run_handle_cancel_returns_before_blocking_model_finishes(self) -> None:
        model = _BlockingThenFinalModel()
        handle = HarnessRunHandle(run_id="run_cancel", turn_id="turn_cancel")
        cancel_handle = handle.cancellation_token

        handle.start(
            lambda: run_agent_harness(
                model,
                ToolRegistry(),
                {"mode": "chat"},
                run_id=handle.run_id,
                turn_id=handle.turn_id,
                cancellation_token=cancel_handle,
                event_sink=handle.event_sink,
                **_run_kwargs(),
            )
        )
        self.assertTrue(model.entered.wait(1.0))
        started = time.monotonic()
        self.assertTrue(handle.cancel("user_requested"))
        result = handle.wait(1.0)
        elapsed = time.monotonic() - started
        model.release.set()

        self.assertLess(elapsed, 0.5)
        self.assertEqual(result["status"], "cancelled")
        self.assertEqual(result["events"][-1]["type"], "run.cancelled")
        self.assertFalse(
            any(item["type"] == "action.completed" for item in result["events"])
        )

    def test_event_cursor_replays_only_new_events(self) -> None:
        class Model:
            def plan(
                self,
                _request: Any,
                *,
                cancellation_token: CancellationToken,
                deadline_monotonic: float,
            ) -> HarnessModelResponse:
                del cancellation_token, deadline_monotonic
                return HarnessModelResponse(kind="final", output={"message": "ok"})

        handle = HarnessRunHandle(run_id="run_events", turn_id="turn_events")
        cancel_handle = handle.cancellation_token
        handle.start(
            lambda: run_agent_harness(
                Model(),
                ToolRegistry(),
                {},
                run_id=handle.run_id,
                turn_id=handle.turn_id,
                event_sink=handle.event_sink,
                cancellation_token=cancel_handle,
                **_run_kwargs(),
            )
        )
        result = handle.wait(1.0)
        all_events = handle.events_after(0)
        tail = handle.events_after(2)

        self.assertEqual(all_events, result["events"])
        self.assertTrue(tail)
        self.assertTrue(all(item["sequence"] > 2 for item in tail))
        streamed = [item for item in handle.iter_events(after_sequence=2) if item]
        self.assertEqual(streamed, tail)

    def test_follow_up_queue_is_fifo_and_individually_cancelable(self) -> None:
        queue = FollowUpQueue(max_items=3)
        first = queue.submit({"message": "first"}, input_id="follow_1")
        second = queue.submit({"message": "second"}, input_id="follow_2")

        self.assertEqual((first, second), ("follow_1", "follow_2"))
        self.assertTrue(queue.remove("follow_1"))
        item = queue.pop()
        self.assertIsNotNone(item)
        assert item is not None
        self.assertEqual(item.input_id, "follow_2")
        self.assertEqual(item.payload, {"message": "second"})
        self.assertIsNone(queue.pop())


if __name__ == "__main__":
    unittest.main()
