# Local inference performance

This report summarizes one reproducible local run of:

```bash
make data
make weather
make performance
```

## Protocol

- Host: Apple silicon, macOS arm64, 14 logical CPU cores
- Python: 3.13.11
- Training history: latest five years, 43,800 hourly rows
- Forecast request: one 168-hour week
- LightGBM: 300 estimators
- Timing: five warmup calls and 100 measured in-process predictions

## Results

| Model | Fit time | Median prediction | P95 prediction | Serialized size | Fit RSS delta |
|---|---:|---:|---:|---:|---:|
| Weekly seasonal naive | 0.10 ms | 0.001 ms | 0.001 ms | 1.55 KiB | 0.0 MiB |
| LightGBM base | 2.34 s | 0.564 ms | 0.769 ms | 836.41 KiB | 7.69 MiB |
| LightGBM + weather + holidays | 2.32 s | 0.687 ms | 1.032 ms | 836.87 KiB | 5.63 MiB |

Both LightGBM variants serve a complete weekly horizon in under one millisecond
median warm latency on this machine. The exogenous model's 19-feature input has
negligible deployment-size overhead relative to the 11-feature base model.

The benchmark also identified unnecessary state in the seasonal-naive model.
Retaining only its final seasonal cycle reduced serialized size from roughly
1.1 MiB to 1.55 KiB without changing forecasts.

## Limitations

- These are local in-process timings, not HTTP end-to-end latency.
- Sub-millisecond measurements are sensitive to scheduler and CPU state.
- Process RSS deltas are order-sensitive and should be treated as indicative.
- Pickle size depends on Python and library versions.
- Results should be regenerated on deployment hardware before capacity planning.
