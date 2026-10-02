import asyncio
import random

import pytest

from gridcast.serving.resilience import (
    CircuitBreaker,
    ConcurrencyLimiter,
    FaultInjector,
    OverloadedError,
)


def test_circuit_breaker_full_state_machine() -> None:
    now = [0.0]
    breaker = CircuitBreaker(2, 5.0, lambda: now[0])
    assert breaker.allow()
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "open"
    assert not breaker.allow()
    now[0] = 5.0
    assert breaker.state == "half_open"
    assert breaker.allow()
    assert not breaker.allow()
    breaker.record_failure()
    assert breaker.state == "open"
    now[0] = 10.0
    assert breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed"


def test_limiter_and_fault_injector_paths() -> None:
    async def exercise() -> None:
        limiter = ConcurrencyLimiter(1)
        async with limiter:
            assert limiter.in_flight == 1
            with pytest.raises(OverloadedError):
                async with limiter:
                    pass
        assert limiter.in_flight == 0
        await FaultInjector(0.0, 0).inject()
        await FaultInjector(0.0, 1).inject()
        with pytest.raises(RuntimeError, match="injected"):
            await FaultInjector(1.0, 0, random.Random(1)).inject()

    asyncio.run(exercise())
    with pytest.raises(ValueError):
        ConcurrencyLimiter(0)
    with pytest.raises(ValueError):
        CircuitBreaker(0, 1)
