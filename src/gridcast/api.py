import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from gridcast.api_models import (
    DecisionResponse,
    HealthResponse,
    LeaderboardEntry,
    MetadataResponse,
    PerformanceResponse,
    PointForecastResponse,
    ProbabilisticForecastResponse,
)
from gridcast.api_service import ForecastNotFoundError, GridCastService
from gridcast.dashboard_data import MissingArtifactsError, load_dashboard_data

LOGGER = logging.getLogger(__name__)


@lru_cache
def get_gridcast_service() -> GridCastService:
    """Load and cache the artifact-backed application service.

    Returns
    -------
    GridCastService
        Read-only forecast service.
    """
    return GridCastService(load_dashboard_data())


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Log API process lifecycle without eagerly requiring artifacts."""
    LOGGER.info("GridCast API started")
    yield
    get_gridcast_service.cache_clear()
    LOGGER.info("GridCast API stopped")


app = FastAPI(
    title="GridCast API",
    description="Read-only PJME point and probabilistic forecast artifacts.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.exception_handler(MissingArtifactsError)
async def missing_artifacts_handler(
    _: Request, error: MissingArtifactsError
) -> JSONResponse:
    """Map absent generated artifacts to service unavailable."""
    LOGGER.warning("API artifacts unavailable: %s", error)
    return JSONResponse(status_code=503, content={"detail": str(error)})


@app.exception_handler(ForecastNotFoundError)
async def forecast_not_found_handler(
    _: Request, error: ForecastNotFoundError
) -> JSONResponse:
    """Map missing forecast selections to HTTP 404."""
    return JSONResponse(status_code=404, content={"detail": str(error)})


@app.get("/health", response_model=HealthResponse, tags=["operations"])
async def health() -> HealthResponse:
    """Return process-level health without loading forecast artifacts."""
    return HealthResponse(status="ok", service="gridcast", version="1.0.0")


@app.get("/api/v1/metadata", response_model=MetadataResponse, tags=["metadata"])
async def metadata(
    service: Annotated[GridCastService, Depends(get_gridcast_service)],
) -> MetadataResponse:
    """Return dataset, evaluation, and uncertainty metadata."""
    return await service.metadata()


@app.get(
    "/api/v1/leaderboard",
    response_model=list[LeaderboardEntry],
    tags=["forecasts"],
)
async def leaderboard(
    service: Annotated[GridCastService, Depends(get_gridcast_service)],
    split: Annotated[
        str, Query(pattern="^(validation|historical_holdout)$")
    ] = "historical_holdout",
) -> list[LeaderboardEntry]:
    """Return point-model ranking for validation or historical holdout."""
    return await service.leaderboard(split)


@app.get(
    "/api/v1/forecasts/point",
    response_model=PointForecastResponse,
    tags=["forecasts"],
)
async def point_forecasts(
    service: Annotated[GridCastService, Depends(get_gridcast_service)],
    fold: Annotated[int, Query(ge=1)],
    split: Annotated[
        str, Query(pattern="^(validation|historical_holdout)$")
    ] = "historical_holdout",
    models: Annotated[list[str] | None, Query()] = None,
) -> PointForecastResponse:
    """Return actuals and selected point models for one weekly fold."""
    return await service.point_forecasts(split, fold, models)


@app.get(
    "/api/v1/forecasts/probabilistic",
    response_model=ProbabilisticForecastResponse,
    tags=["forecasts"],
)
async def probabilistic_forecasts(
    service: Annotated[GridCastService, Depends(get_gridcast_service)],
    fold: Annotated[int, Query(ge=1)],
    split: Annotated[
        str, Query(pattern="^(validation|historical_holdout)$")
    ] = "historical_holdout",
) -> ProbabilisticForecastResponse:
    """Return raw quantiles and conformal bounds for one weekly fold."""
    return await service.probabilistic_forecasts(split, fold)


@app.get(
    "/api/v1/performance",
    response_model=PerformanceResponse,
    tags=["operations"],
)
async def performance(
    service: Annotated[GridCastService, Depends(get_gridcast_service)],
) -> PerformanceResponse:
    """Return optional local model performance measurements."""
    return await service.performance()


@app.get(
    "/api/v1/decisions",
    response_model=DecisionResponse,
    tags=["decisions"],
)
async def decisions(
    service: Annotated[GridCastService, Depends(get_gridcast_service)],
) -> DecisionResponse:
    """Return optional synthetic decision-cost sensitivity results."""
    return await service.decisions()
