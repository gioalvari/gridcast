"""Validated environment configuration for GridCast production serving."""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServingSettings:
    """Immutable serving settings resolved from ``GRIDCAST_*`` variables."""

    model_uri: str
    model_cache_dir: Path
    environment: str
    request_timeout_s: float
    max_in_flight: int
    max_admitted: int
    breaker_failures: int
    breaker_reset_s: float
    max_body_bytes: int
    fault_error_rate: float
    fault_latency_ms: int
    v1_sunset: str

    @classmethod
    def from_env(cls, environ: Mapping[str, str] = os.environ) -> "ServingSettings":
        """Read and validate all serving settings from an environment mapping."""
        model_uri = environ.get("GRIDCAST_MODEL_URI", "")
        if not model_uri:
            raise ValueError("GRIDCAST_MODEL_URI is required for serving")
        max_in_flight = _integer(environ, "GRIDCAST_MAX_IN_FLIGHT", 32)
        settings = cls(
            model_uri=model_uri,
            model_cache_dir=Path(
                environ.get("GRIDCAST_MODEL_CACHE_DIR", "/tmp/gridcast-models")
            ),
            environment=environ.get("GRIDCAST_ENV", "local"),
            request_timeout_s=_float(environ, "GRIDCAST_REQUEST_TIMEOUT_S", 0.5),
            max_in_flight=max_in_flight,
            max_admitted=_integer(environ, "GRIDCAST_MAX_ADMITTED", 2 * max_in_flight),
            breaker_failures=_integer(environ, "GRIDCAST_BREAKER_FAILURES", 5),
            breaker_reset_s=_float(environ, "GRIDCAST_BREAKER_RESET_S", 30.0),
            max_body_bytes=_integer(environ, "GRIDCAST_MAX_BODY_BYTES", 1_048_576),
            fault_error_rate=_float(environ, "GRIDCAST_FAULT_ERROR_RATE", 0.0),
            fault_latency_ms=_integer(environ, "GRIDCAST_FAULT_LATENCY_MS", 0),
            v1_sunset=environ.get("GRIDCAST_V1_SUNSET", "2027-06-30"),
        )
        if (
            settings.request_timeout_s <= 0
            or settings.max_in_flight < 1
            or settings.max_admitted < settings.max_in_flight
            or settings.breaker_failures < 1
            or settings.breaker_reset_s <= 0
            or settings.max_body_bytes < 1
            or not 0 <= settings.fault_error_rate <= 1
            or settings.fault_latency_ms < 0
        ):
            raise ValueError("serving numeric settings are out of range")
        if settings.environment == "production" and (
            settings.fault_error_rate != 0 or settings.fault_latency_ms != 0
        ):
            raise ValueError("fault injection must be disabled in production")
        return settings


def _float(environ: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(environ.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be numeric") from error


def _integer(environ: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(environ.get(name, str(default)))
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
