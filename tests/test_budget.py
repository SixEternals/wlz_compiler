import io
import json
import math
import unittest
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

from wlz_optimizer.budget import BudgetController, BudgetLimits
from wlz_optimizer.stdlib_llm import StdlibOpenAIClient


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class BudgetControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.controller = BudgetController(
            BudgetLimits(100, 10, reserved_tokens=20, reserved_seconds=2),
            clock=self.clock,
        )

    def test_search_protects_reserve_at_exact_boundary(self) -> None:
        self.assertTrue(self.controller.check_start(80, 8).allowed)
        self.assertEqual(self.controller.check_start(81, 0).reason, "token_reserve")
        self.assertEqual(self.controller.check_start(0, 8.1).reason, "wall_time_reserve")

    def test_finalization_can_use_reserve_but_not_cross_hard_limit(self) -> None:
        self.assertTrue(self.controller.check_start(100, 10, use_reserve=True).allowed)
        denied = self.controller.check_start(101, 0, use_reserve=True)
        self.assertFalse(denied.allowed)
        self.assertEqual(denied.reason, "token_limit")
        self.assertIsNone(denied.snapshot.stop_reason)

    def test_reservation_holds_capacity_and_commit_releases_unused_amount(self) -> None:
        decision = self.controller.reserve(30, 20, 5, 3)
        self.assertTrue(decision.allowed)
        self.assertIsNotNone(decision.reservation)
        self.assertEqual(decision.snapshot.reserved_tokens, 55)
        self.assertEqual(decision.snapshot.reserved_seconds, 3)
        self.assertEqual(decision.snapshot.in_flight_calls, 1)
        self.assertEqual(self.controller.reserve(1, 25, 0, 0).reason, "token_reserve")

        settled = self.controller.commit(decision.reservation, 40)
        self.assertEqual(settled.used_tokens, 40)
        self.assertEqual(settled.reserved_tokens, 0)
        self.assertEqual(settled.remaining_tokens, 60)
        self.assertEqual(settled.in_flight_calls, 0)

    def test_release_is_unbilled_and_reservation_cannot_be_reused(self) -> None:
        reservation = self.controller.reserve(10, 10, 0, 1).reservation
        released = self.controller.release(reservation)
        self.assertEqual(released.used_tokens, 0)
        self.assertEqual(released.reserved_tokens, 0)
        with self.assertRaisesRegex(ValueError, "not active"):
            self.controller.release(reservation)

    def test_commit_requires_usage_or_explicit_fallback(self) -> None:
        reservation = self.controller.reserve(10, 10, 0, 1).reservation
        with self.assertRaisesRegex(ValueError, "fallback_tokens"):
            self.controller.commit(reservation, None)
        self.assertEqual(self.controller.snapshot().in_flight_calls, 1)
        settled = self.controller.commit(reservation, None, fallback_tokens=20)
        self.assertEqual(settled.used_tokens, 20)

    def test_uncertain_call_charges_full_reservation_and_stops(self) -> None:
        reservation = self.controller.reserve(10, 20, 5, 1).reservation
        stopped = self.controller.mark_uncertain(reservation)
        self.assertEqual(stopped.used_tokens, 35)
        self.assertEqual(stopped.reserved_tokens, 0)
        self.assertEqual(stopped.stop_reason, "unknown_in_flight_call")
        self.assertFalse(self.controller.reserve(1, 1, 0, 0).allowed)

    def test_monotonic_deadline_is_enforced(self) -> None:
        self.clock.now = -10
        controller = BudgetController(BudgetLimits(1, 10), clock=self.clock)
        self.clock.advance(9)
        self.assertEqual(
            controller.check_start(0, 2, use_reserve=True).reason,
            "wall_time_limit",
        )
        self.assertTrue(controller.check_start(0, 1, use_reserve=True).allowed)
        self.clock.advance(1)
        self.assertEqual(controller.snapshot().stop_reason, "wall_time_limit")

    def test_first_hard_stop_reason_is_latched(self) -> None:
        controller = BudgetController(
            BudgetLimits(token_limit=10, wall_time_seconds=5),
            clock=self.clock,
        )
        reservation = controller.reserve(5, 5, 0, 0).reservation
        self.assertEqual(controller.commit(reservation, 10).stop_reason, "token_limit")
        self.clock.advance(6)
        self.assertEqual(controller.snapshot().stop_reason, "token_limit")

    def test_invalid_limits_and_inputs_are_rejected(self) -> None:
        invalid_limits = [
            dict(token_limit=True, wall_time_seconds=1),
            dict(token_limit=1, wall_time_seconds=math.nan),
            dict(token_limit=1, wall_time_seconds=1, reserved_tokens=2),
            dict(token_limit=1, wall_time_seconds=1, reserved_seconds=2),
        ]
        for kwargs in invalid_limits:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                BudgetLimits(**kwargs)

        invalid_checks = [(-1, 0), (0, -1), (0, math.inf)]
        for tokens, seconds in invalid_checks:
            with self.subTest(tokens=tokens, seconds=seconds), self.assertRaises(ValueError):
                self.controller.check_start(tokens, seconds)
        with self.assertRaises(ValueError):
            self.controller.check_start(0, 0, use_reserve=1)
        with self.assertRaises(ValueError):
            self.controller.reserve(0, 0, 0, 0)


class _FakeResponse:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False


class StdlibClientBudgetTests(unittest.TestCase):
    def _config(self, budget: BudgetController | None) -> SimpleNamespace:
        return SimpleNamespace(
            llm_models=["deepseek-v4-pro"],
            api_url="https://example.invalid",
            api_key="bounded-test-key",
            llm_temperature=0.2,
            max_llm_tokens=128,
            budget_controller=budget,
            llm_expected_seconds=1,
        )

    def test_http_error_releases_reservation_and_budget_stays_usable(self) -> None:
        budget = BudgetController(BudgetLimits(10_000, 300))
        client = StdlibOpenAIClient(self._config(budget))
        error = urllib.error.HTTPError(
            "https://example.invalid", 429, "Too Many Requests", None,
            io.BytesIO(b"rate limited"),
        )
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(RuntimeError, "LLM HTTP 429"):
                client.generate("mutate this")
        snapshot = budget.snapshot()
        self.assertEqual(snapshot.used_tokens, 0)
        self.assertEqual(snapshot.in_flight_calls, 0)
        self.assertIsNone(snapshot.stop_reason)
        record = client.call_history[-1]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_type"], "http_error")
        self.assertEqual(record["http_status"], 429)
        self.assertTrue(budget.reserve(10, 10, 0, 1).allowed)

    def test_empty_content_is_recorded_as_failed_call(self) -> None:
        client = StdlibOpenAIClient(self._config(None))
        response = _FakeResponse({
            "choices": [{"message": {"content": "   "}, "finish_reason": "length"}],
            "usage": {"total_tokens": 7},
        })
        with patch("urllib.request.urlopen", return_value=response):
            with self.assertRaisesRegex(RuntimeError, "empty content"):
                client.generate("mutate this")
        self.assertEqual(len(client.call_history), 1)
        record = client.call_history[0]
        self.assertEqual(record["status"], "failed")
        self.assertEqual(record["error_type"], "empty_content")


class BudgetCheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = FakeClock()
        self.limits = BudgetLimits(
            100,
            10,
            reserved_tokens=20,
            reserved_seconds=2,
        )

    def test_snapshot_restore_roundtrip(self) -> None:
        controller = BudgetController(self.limits, clock=self.clock)
        reservation = controller.reserve(10, 20, 0, 1).reservation
        controller.commit(reservation, 25)
        self.clock.advance(3)
        state = json.loads(json.dumps(controller.snapshot()))

        restored = BudgetController(self.limits, clock=self.clock)
        restored_snapshot = restored.restore(state)

        self.assertEqual(restored_snapshot.used_tokens, state["used_tokens"])
        self.assertEqual(restored_snapshot.elapsed_seconds, state["elapsed_seconds"])
        self.assertEqual(restored_snapshot.stop_reason, state["stop_reason"])
        self.assertFalse(restored.check_start(76, 0, use_reserve=True).allowed)

    def test_restore_rejects_tampered_version(self) -> None:
        state = BudgetController(self.limits, clock=self.clock).snapshot()
        state["version"] = "tampered"

        with self.assertRaisesRegex(ValueError, "version"):
            BudgetController(self.limits, clock=self.clock).restore(state)

    def test_restore_rejects_mismatched_limits(self) -> None:
        state = BudgetController(self.limits, clock=self.clock).snapshot()
        different_limits = BudgetLimits(
            101,
            10,
            reserved_tokens=20,
            reserved_seconds=2,
        )

        with self.assertRaisesRegex(ValueError, "limits"):
            BudgetController(different_limits, clock=self.clock).restore(state)

    def test_restore_with_inflight_fails_closed(self) -> None:
        controller = BudgetController(self.limits, clock=self.clock)
        controller.reserve(10, 20, 0, 1)

        restored = BudgetController(self.limits, clock=self.clock)
        snapshot = restored.restore(controller.snapshot())

        self.assertEqual(snapshot.stop_reason, "unknown_in_flight_call")
        self.assertEqual(snapshot.in_flight_calls, 0)
        self.assertFalse(restored.reserve(1, 1, 0, 0).allowed)

    def test_restore_then_reject_new_calls_when_over_limit(self) -> None:
        controller = BudgetController(self.limits, clock=self.clock)
        reservation = controller.reserve(40, 50, 0, 1, use_reserve=True).reservation
        controller.commit(reservation, 99)

        restored = BudgetController(self.limits, clock=self.clock)
        restored.restore(controller.snapshot())

        denied = restored.reserve(1, 1, 0, 0, use_reserve=True)
        self.assertFalse(denied.allowed)
        self.assertEqual(denied.reason, "token_limit")
