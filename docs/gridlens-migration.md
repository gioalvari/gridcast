# GridLens consolidation

GridLens was consolidated into GridCast to avoid maintaining two overlapping
energy-forecasting repositories.

## Preserved in GridCast

- forecast-origin, validity, and availability contracts;
- DST-safe Italian delivery days;
- immutable ECMWF weather-run selection;
- eight-node Italian archived weather ingestion and coverage checks;
- asymmetric procurement cost and cost-optimal quantile metrics;
- ENTSO-E Italian actual-load feasibility work.

## Reused rather than duplicated

GridCast already had more complete implementations for:

- ENTSO-E authentication, retries, XML parsing, quality reports, and Parquet;
- seasonal baselines and chronological backtesting;
- quantile models and global, hourly, and rolling conformal calibration;
- point, probabilistic, and interval metrics;
- CLI packaging, strict typing, tests, CI, FastAPI, Streamlit, and Docker.

Those components remain canonical. No second `gridlens` package or duplicate
CLI framework was introduced.

The original GridLens repository is retained as an archived redirect so old
links continue to resolve, but all future development belongs in GridCast.
