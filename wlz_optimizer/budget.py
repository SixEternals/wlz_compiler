"""Shared token and wall-clock budget accounting for one search run."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Callable, Optional


BUDGET_CHECKPOINT_VERSION = "budget-checkpoint-v1"


def _token_count(name: str, value: int, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    if value < (1 if positive else 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return value


def _seconds(
    name: str, value: float, *, positive: bool = False, allow_negative: bool = False
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if (not allow_negative and result < 0) or (positive and result == 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


@dataclass(frozen=True)
class BudgetLimits:
    """Hard per-operator limits and the portion protected for finalization."""

    token_limit: int
    wall_time_seconds: float
    reserved_tokens: int = 0
    reserved_seconds: float = 0.0

    def __post_init__(self) -> None:
        _token_count("token_limit", self.token_limit, positive=True)
        _seconds("wall_time_seconds", self.wall_time_seconds, positive=True)
        _token_count("reserved_tokens", self.reserved_tokens)
        _seconds("reserved_seconds", self.reserved_seconds)
        if self.reserved_tokens > self.token_limit:
            raise ValueError("reserved_tokens must not exceed token_limit")
        if self.reserved_seconds > self.wall_time_seconds:
            raise ValueError("reserved_seconds must not exceed wall_time_seconds")


class BudgetSnapshot(dict):
    """JSON-safe checkpoint state with attribute access for existing callers."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


@dataclass(frozen=True)
class BudgetReservation:
    reservation_id: int
    token_upper_bound: int
    expected_seconds: float
    use_tail_reserve: bool


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: Optional[str]
    snapshot: BudgetSnapshot
    reservation: Optional[BudgetReservation] = None


class BudgetController:
    """Reserve and settle token/time capacity for one bounded search run."""

    def __init__(
        self, limits: BudgetLimits, *, clock: Callable[[], float] = time.monotonic
    ) -> None:
        if not isinstance(limits, BudgetLimits):
            raise TypeError("limits must be a BudgetLimits instance")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self.limits = limits
        self._clock = clock
        self._started_at = _seconds("clock value", self._clock(), allow_negative=True)
        self._used_tokens = 0
        self._reserved_tokens = 0
        self._reserved_seconds = 0.0
        self._next_reservation_id = 1
        self._active_reservations: dict[int, BudgetReservation] = {}
        self._stop_reason: Optional[str] = None

    def snapshot(self) -> BudgetSnapshot:
        now = _seconds("clock value", self._clock(), allow_negative=True)
        if now < self._started_at:
            raise RuntimeError("monotonic clock moved backwards")
        elapsed = now - self._started_at
        remaining_tokens = max(
            0, self.limits.token_limit - self._used_tokens - self._reserved_tokens
        )
        remaining_seconds = max(
            0.0,
            self.limits.wall_time_seconds - elapsed - self._reserved_seconds,
        )
        if self._stop_reason is None:
            if self._used_tokens >= self.limits.token_limit:
                self._stop_reason = "token_limit"
            elif elapsed >= self.limits.wall_time_seconds:
                self._stop_reason = "wall_time_limit"
        return BudgetSnapshot(
            version=BUDGET_CHECKPOINT_VERSION,
            limits={
                "token_limit": self.limits.token_limit,
                "wall_time_seconds": self.limits.wall_time_seconds,
                "reserved_tokens": self.limits.reserved_tokens,
                "reserved_seconds": self.limits.reserved_seconds,
            },
            used_tokens=self._used_tokens,
            remaining_tokens=remaining_tokens,
            elapsed_seconds=elapsed,
            remaining_seconds=remaining_seconds,
            stop_reason=self._stop_reason,
            reserved_tokens=self._reserved_tokens,
            reserved_seconds=self._reserved_seconds,
            in_flight_calls=len(self._active_reservations),
            next_reservation_id=self._next_reservation_id,
        )

    def restore(self, state: dict) -> BudgetSnapshot:
        """Restore one checkpoint, failing closed for unknown in-flight calls."""

        if not isinstance(state, dict):
            raise ValueError("budget checkpoint must be a dictionary")
        if state.get("version") != BUDGET_CHECKPOINT_VERSION:
            raise ValueError("unsupported budget checkpoint version")
        expected_limits = {
            "token_limit": self.limits.token_limit,
            "wall_time_seconds": self.limits.wall_time_seconds,
            "reserved_tokens": self.limits.reserved_tokens,
            "reserved_seconds": self.limits.reserved_seconds,
        }
        if state.get("limits") != expected_limits:
            raise ValueError("budget checkpoint limits do not match controller")

        used_tokens = _token_count("checkpoint used_tokens", state.get("used_tokens"))
        elapsed_seconds = _seconds(
            "checkpoint elapsed_seconds", state.get("elapsed_seconds")
        )
        reserved_tokens = _token_count(
            "checkpoint reserved_tokens", state.get("reserved_tokens")
        )
        reserved_seconds = _seconds(
            "checkpoint reserved_seconds", state.get("reserved_seconds")
        )
        in_flight_calls = _token_count(
            "checkpoint in_flight_calls", state.get("in_flight_calls")
        )
        next_reservation_id = _token_count(
            "checkpoint next_reservation_id",
            state.get("next_reservation_id"),
            positive=True,
        )
        stop_reason = state.get("stop_reason")
        if stop_reason not in {
            None,
            "token_limit",
            "wall_time_limit",
            "unknown_in_flight_call",
        }:
            raise ValueError("invalid budget checkpoint stop_reason")
        if in_flight_calls == 0 and (reserved_tokens or reserved_seconds):
            raise ValueError("budget checkpoint has reserved capacity without a call")
        if in_flight_calls and reserved_tokens == 0:
            raise ValueError("budget checkpoint in-flight call has no token reservation")

        now = _seconds("clock value", self._clock(), allow_negative=True)
        self._started_at = now - elapsed_seconds
        self._used_tokens = used_tokens
        self._reserved_tokens = 0
        self._reserved_seconds = 0.0
        self._active_reservations.clear()
        self._next_reservation_id = next_reservation_id
        self._stop_reason = (
            "unknown_in_flight_call" if in_flight_calls else stop_reason
        )
        return self.snapshot()

    def check_start(
        self,
        estimated_total_tokens: int,
        expected_seconds: float,
        *,
        use_reserve: bool = False,
    ) -> BudgetDecision:
        _token_count("estimated_total_tokens", estimated_total_tokens)
        _seconds("expected_seconds", expected_seconds)
        if not isinstance(use_reserve, bool):
            raise ValueError("use_reserve must be a boolean")

        snapshot = self.snapshot()
        reason = snapshot.stop_reason
        if reason is None and estimated_total_tokens > snapshot.remaining_tokens:
            reason = "token_limit"
        if reason is None and expected_seconds > snapshot.remaining_seconds:
            reason = "wall_time_limit"
        if reason is None and not use_reserve:
            usable_tokens = max(0, snapshot.remaining_tokens - self.limits.reserved_tokens)
            usable_seconds = max(0.0, snapshot.remaining_seconds - self.limits.reserved_seconds)
            if estimated_total_tokens > usable_tokens:
                reason = "token_reserve"
            elif expected_seconds > usable_seconds:
                reason = "wall_time_reserve"
        return BudgetDecision(allowed=reason is None, reason=reason, snapshot=snapshot)

    def reserve(
        self,
        estimated_input_tokens: int,
        max_completion_tokens: int,
        safety_margin_tokens: int,
        expected_seconds: float,
        *,
        use_reserve: bool = False,
    ) -> BudgetDecision:
        """Atomically reserve one call's conservative token and time upper bounds."""

        _token_count("estimated_input_tokens", estimated_input_tokens)
        _token_count("max_completion_tokens", max_completion_tokens)
        _token_count("safety_margin_tokens", safety_margin_tokens)
        total_tokens = (
            estimated_input_tokens + max_completion_tokens + safety_margin_tokens
        )
        if total_tokens == 0:
            raise ValueError("reservation token upper bound must be positive")
        _seconds("expected_seconds", expected_seconds)
        decision = self.check_start(
            total_tokens,
            expected_seconds,
            use_reserve=use_reserve,
        )
        if not decision.allowed:
            return decision
        reservation = BudgetReservation(
            reservation_id=self._next_reservation_id,
            token_upper_bound=total_tokens,
            expected_seconds=float(expected_seconds),
            use_tail_reserve=use_reserve,
        )
        self._next_reservation_id += 1
        self._active_reservations[reservation.reservation_id] = reservation
        self._reserved_tokens += reservation.token_upper_bound
        self._reserved_seconds += reservation.expected_seconds
        return BudgetDecision(
            allowed=True,
            reason=None,
            snapshot=self.snapshot(),
            reservation=reservation,
        )

    def commit(
        self,
        reservation: BudgetReservation,
        total_tokens: Optional[int],
        *,
        fallback_tokens: Optional[int] = None,
    ) -> BudgetSnapshot:
        """Settle a completed call and release its unused reservation."""

        if total_tokens is not None:
            _token_count("total_tokens", total_tokens)
        if fallback_tokens is not None:
            _token_count("fallback_tokens", fallback_tokens, positive=True)
        if total_tokens is None:
            if fallback_tokens is None:
                raise ValueError("fallback_tokens is required when API usage is missing")
            total_tokens = fallback_tokens
        active = self._take_reservation(reservation)
        self._used_tokens += total_tokens
        return self.snapshot()

    def release(self, reservation: BudgetReservation) -> BudgetSnapshot:
        """Release capacity for a call that was never started."""

        self._take_reservation(reservation)
        return self.snapshot()

    def mark_uncertain(self, reservation: BudgetReservation) -> BudgetSnapshot:
        """Fail closed when a started call has no trustworthy usage result."""

        active = self._take_reservation(reservation)
        self._used_tokens += active.token_upper_bound
        if self._stop_reason is None:
            self._stop_reason = "unknown_in_flight_call"
        return self.snapshot()

    def _take_reservation(self, reservation: BudgetReservation) -> BudgetReservation:
        if not isinstance(reservation, BudgetReservation):
            raise TypeError("reservation must be a BudgetReservation")
        active = self._active_reservations.get(reservation.reservation_id)
        if active is not reservation:
            raise ValueError("reservation is not active for this controller")
        del self._active_reservations[reservation.reservation_id]
        self._reserved_tokens -= active.token_upper_bound
        self._reserved_seconds -= active.expected_seconds
        return active
