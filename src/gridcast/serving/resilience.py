"""Small, dependency-free resilience primitives for the serving layer."""

import asyncio
import random
import time
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager


class OverloadedError(RuntimeError):
    """Raised when a request arrives while all serving slots are occupied."""


class CircuitBreaker:
    """Circuit breaker with a single half-open recovery trial."""

    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout_s: float = 30.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Initialize a closed breaker with validated failure settings."""
        if failure_threshold < 1 or reset_timeout_s <= 0:
            raise ValueError("failure_threshold and reset_timeout_s must be positive")
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._half_open_trial = False

    @property
    def state(self) -> str:
        """Return the current closed, open, or half_open state."""
        if self._opened_at is None:
            return "closed"
        if self._clock() - self._opened_at < self.reset_timeout_s:
            return "open"
        return "half_open"

    def allow(self) -> bool:
        """Allow normal traffic or the one eligible half-open recovery trial."""
        if self.state == "closed":
            return True
        if self.state == "open" or self._half_open_trial:
            return False
        self._half_open_trial = True
        return True

    def record_success(self) -> None:
        """Close the circuit after a successful primary prediction."""
        self._failures = 0
        self._opened_at = None
        self._half_open_trial = False

    def record_failure(self) -> None:
        """Count a primary failure and open immediately for a failed trial."""
        self._failures += 1
        if self._half_open_trial or self._failures >= self.failure_threshold:
            self._opened_at = self._clock()
            self._half_open_trial = False


class ConcurrencyLimiter(AbstractAsyncContextManager[None]):
    """Non-queuing async admission control for forecast requests."""

    def __init__(self, max_in_flight: int) -> None:
        """Initialize an empty limiter with a positive capacity."""
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be positive")
        self.max_in_flight = max_in_flight
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        """Return the number of currently admitted requests."""
        return self._in_flight

    async def __aenter__(self) -> None:
        """Acquire a slot without waiting for an occupied slot."""
        self.acquire()

    def acquire(self) -> None:
        """Acquire a slot without waiting for an occupied slot."""
        if self._in_flight >= self.max_in_flight:
            raise OverloadedError("forecast service is overloaded")
        self._in_flight += 1

    async def __aexit__(self, *_: object) -> None:
        """Release the admitted serving slot."""
        self.release()

    def release(self) -> None:
        """Release one previously admitted serving slot."""
        self._in_flight -= 1


class FaultInjector:
    """Test-only latency and error injector for non-production deployments."""

    def __init__(
        self,
        error_rate: float,
        latency_ms: int,
        rng: random.Random | None = None,
    ) -> None:
        """Initialize injectable faults with deterministic random support."""
        if not 0.0 <= error_rate <= 1.0 or latency_ms < 0:
            raise ValueError(
                "error_rate must be within [0, 1] and latency_ms non-negative"
            )
        self.error_rate = error_rate
        self.latency_ms = latency_ms
        self.rng = rng or random.Random()

    async def inject(self) -> None:
        """Apply configured latency then probabilistically raise an injected error."""
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        if self.error_rate and self.rng.random() < self.error_rate:
            raise RuntimeError("injected serving fault")
