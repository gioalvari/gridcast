import argparse
import json
import logging
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from gridcast.backtesting import BacktestConfig, rolling_backtest
from gridcast.benchmark import (
    BenchmarkConfig,
    run_pjme_benchmark,
    write_benchmark_artifacts,
)
from gridcast.data import generate_synthetic_load
from gridcast.eda import create_eda_report
from gridcast.entsoe import ITALY_BIDDING_ZONE, ingest_entsoe_actual_load
from gridcast.performance import (
    PerformanceConfig,
    run_performance_benchmark,
    write_performance_artifacts,
)
from gridcast.pjm import ingest_pjme
from gridcast.probabilistic import (
    ProbabilisticConfig,
    run_probabilistic_benchmark,
    write_probabilistic_artifacts,
)
from gridcast.weather import ingest_temperature

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the GridCast command-line parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured top-level parser.
    """
    parser = argparse.ArgumentParser(
        prog="gridcast",
        description="Reproducible energy forecasting experiments.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo = subparsers.add_parser("demo", help="run the synthetic baseline demo")
    demo.add_argument("--periods", type=int, default=24 * 140)
    demo.add_argument("--initial-window", type=int, default=24 * 84)
    demo.add_argument("--horizon", type=int, default=24 * 7)
    demo.add_argument("--step", type=int, default=24 * 7)
    demo.add_argument("--seasonal-period", type=int, default=24 * 7)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    data = subparsers.add_parser("data", help="manage public energy datasets")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    download = data_subparsers.add_parser(
        "download", help="download and prepare PJME hourly load"
    )
    download.add_argument(
        "--raw-path", type=Path, default=Path("data/raw/PJME_hourly.csv")
    )
    download.add_argument(
        "--output", type=Path, default=Path("data/processed/pjme_hourly.parquet")
    )
    download.add_argument(
        "--report", type=Path, default=Path("artifacts/data/pjme_quality.json")
    )
    download.add_argument("--force", action="store_true")
    weather = data_subparsers.add_parser(
        "weather", help="download and prepare Philadelphia ERA5 temperature"
    )
    weather.add_argument(
        "--raw-path", type=Path, default=Path("data/raw/philadelphia_era5.json")
    )
    weather.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/philadelphia_temperature.parquet"),
    )
    weather.add_argument(
        "--report", type=Path, default=Path("artifacts/data/weather_quality.json")
    )
    weather.add_argument("--force", action="store_true")
    entsoe = data_subparsers.add_parser(
        "entsoe", help="download ENTSO-E actual total load"
    )
    entsoe.add_argument("--start", required=True, help="inclusive date, YYYY-MM-DD")
    entsoe.add_argument("--end", required=True, help="exclusive date, YYYY-MM-DD")
    entsoe.add_argument("--area", default=ITALY_BIDDING_ZONE)
    entsoe.add_argument("--raw-dir", type=Path, default=Path("data/raw/entsoe"))
    entsoe.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/entsoe_italy_actual_load.parquet"),
    )
    entsoe.add_argument(
        "--report", type=Path, default=Path("artifacts/data/entsoe_quality.json")
    )
    entsoe.add_argument("--force", action="store_true")
    eda = subparsers.add_parser("eda", help="create PJME exploratory analysis")
    eda.add_argument(
        "--input", type=Path, default=Path("data/processed/pjme_hourly.parquet")
    )
    eda.add_argument("--output-dir", type=Path, default=Path("artifacts/eda"))
    benchmark = subparsers.add_parser(
        "benchmark", help="benchmark baselines and LightGBM on PJME"
    )
    benchmark.add_argument(
        "--input", type=Path, default=Path("data/processed/pjme_hourly.parquet")
    )
    benchmark.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/benchmark")
    )
    benchmark.add_argument(
        "--weather",
        type=Path,
        default=Path("data/processed/philadelphia_temperature.parquet"),
    )
    benchmark.add_argument("--without-exogenous", action="store_true")
    benchmark.add_argument("--horizon", type=int, default=24 * 7)
    benchmark.add_argument("--validation-folds", type=int, default=12)
    benchmark.add_argument("--test-folds", type=int, default=52)
    benchmark.add_argument("--max-train-hours", type=int, default=24 * 365 * 5)
    benchmark.add_argument("--n-estimators", type=int, default=300)
    probabilistic = subparsers.add_parser(
        "probabilistic", help="run quantile and conformal PJME benchmark"
    )
    probabilistic.add_argument(
        "--input", type=Path, default=Path("data/processed/pjme_hourly.parquet")
    )
    probabilistic.add_argument(
        "--weather",
        type=Path,
        default=Path("data/processed/philadelphia_temperature.parquet"),
    )
    probabilistic.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/probabilistic")
    )
    probabilistic.add_argument("--horizon", type=int, default=24 * 7)
    probabilistic.add_argument("--validation-folds", type=int, default=12)
    probabilistic.add_argument("--test-folds", type=int, default=52)
    probabilistic.add_argument("--max-train-hours", type=int, default=24 * 365 * 5)
    probabilistic.add_argument("--n-estimators", type=int, default=300)
    probabilistic.add_argument("--rolling-window-folds", type=int, default=12)
    performance = subparsers.add_parser(
        "performance", help="benchmark local fit and inference performance"
    )
    performance.add_argument(
        "--input", type=Path, default=Path("data/processed/pjme_hourly.parquet")
    )
    performance.add_argument(
        "--weather",
        type=Path,
        default=Path("data/processed/philadelphia_temperature.parquet"),
    )
    performance.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/performance")
    )
    performance.add_argument("--horizon", type=int, default=24 * 7)
    performance.add_argument("--max-train-hours", type=int, default=24 * 365 * 5)
    performance.add_argument("--n-estimators", type=int, default=300)
    performance.add_argument("--warmup-runs", type=int, default=5)
    performance.add_argument("--repetitions", type=int, default=100)
    return parser


def run_demo(args: argparse.Namespace) -> dict[str, float | int]:
    """Run the synthetic baseline experiment and persist its artifacts.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed demo command arguments.

    Returns
    -------
    dict
        Aggregate backtest metrics.
    """
    config = BacktestConfig(
        initial_window=args.initial_window,
        horizon=args.horizon,
        step=args.step,
        seasonal_period=args.seasonal_period,
    )
    data = generate_synthetic_load(periods=args.periods, seed=args.seed)
    result = rolling_backtest(data, config)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data.to_csv(output_dir / "load.csv", index=False)
    result.forecasts.to_csv(output_dir / "forecasts.csv", index=False)
    result.metrics.to_csv(output_dir / "metrics.csv", index=False)
    report: dict[str, object] = {
        "model": "seasonal_naive",
        "data": {"periods": args.periods, "seed": args.seed},
        "backtest": {
            "initial_window": config.initial_window,
            "horizon": config.horizon,
            "step": config.step,
            "seasonal_period": config.seasonal_period,
        },
        "metrics": result.summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    LOGGER.info("Wrote demo artifacts to %s", output_dir.resolve())
    return result.summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run the GridCast command-line interface.

    Parameters
    ----------
    argv : sequence of str, optional
        Arguments to parse. Defaults to process arguments.

    Returns
    -------
    int
        Process exit status.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        summary = run_demo(args)
        LOGGER.info(
            "Backtest complete: %d folds, MAE %.2f MW, MASE %.3f",
            summary["folds"],
            summary["mae"],
            summary["mase"],
        )
    elif args.command == "data" and args.data_command == "download":
        report = ingest_pjme(
            args.raw_path,
            args.output,
            args.report,
            force=args.force,
        )
        LOGGER.info(
            "Prepared %d hourly observations (%d duplicates, %d imputed hours)",
            report.output_rows,
            report.duplicate_timestamps,
            report.imputed_observations,
        )
    elif args.command == "data" and args.data_command == "weather":
        weather_report = ingest_temperature(
            args.raw_path,
            args.output,
            args.report,
            force=args.force,
        )
        LOGGER.info(
            "Prepared %d hourly temperature observations from %s to %s",
            weather_report.observations,
            weather_report.start,
            weather_report.end,
        )
    elif args.command == "data" and args.data_command == "entsoe":
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=UTC)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=UTC)
        entsoe_report = ingest_entsoe_actual_load(
            args.raw_dir,
            args.output,
            args.report,
            start,
            end,
            area=args.area,
            force=args.force,
        )
        LOGGER.info(
            "Prepared %d ENTSO-E load observations at %d-minute resolution",
            entsoe_report.observations,
            entsoe_report.resolution_minutes,
        )
    elif args.command == "eda":
        import pandas as pd

        eda_summary = create_eda_report(pd.read_parquet(args.input), args.output_dir)
        LOGGER.info(
            "EDA complete: %d observations from %s to %s",
            eda_summary["observations"],
            eda_summary["start"],
            eda_summary["end"],
        )
    elif args.command == "benchmark":
        import pandas as pd

        benchmark_config = BenchmarkConfig(
            horizon=args.horizon,
            validation_folds=args.validation_folds,
            test_folds=args.test_folds,
            max_train_hours=args.max_train_hours,
            n_estimators=args.n_estimators,
        )
        weather_data = None
        if not args.without_exogenous:
            if not args.weather.exists():
                msg = f"weather data not found at {args.weather}; run `make weather`"
                raise FileNotFoundError(msg)
            weather_data = pd.read_parquet(args.weather)
        benchmark_result = run_pjme_benchmark(
            pd.read_parquet(args.input), benchmark_config, weather_data
        )
        write_benchmark_artifacts(benchmark_result, benchmark_config, args.output_dir)
        test_leaderboard = benchmark_result.leaderboard.loc[
            benchmark_result.leaderboard["split"].eq("test")
        ]
        winner = test_leaderboard.iloc[0]
        LOGGER.info(
            "Frozen test winner: %s, MAE %.2f MW, MASE %.3f",
            winner["model"],
            winner["mae"],
            winner["mase"],
        )
    elif args.command == "probabilistic":
        import pandas as pd

        probabilistic_config = ProbabilisticConfig(
            horizon=args.horizon,
            validation_folds=args.validation_folds,
            test_folds=args.test_folds,
            max_train_hours=args.max_train_hours,
            n_estimators=args.n_estimators,
            rolling_window_folds=args.rolling_window_folds,
        )
        probabilistic_result = run_probabilistic_benchmark(
            pd.read_parquet(args.input),
            pd.read_parquet(args.weather),
            probabilistic_config,
        )
        write_probabilistic_artifacts(
            probabilistic_result, probabilistic_config, args.output_dir
        )
        test_metrics = probabilistic_result.metrics.loc[
            probabilistic_result.metrics["split"].eq("test")
        ].iloc[0]
        LOGGER.info(
            "Probabilistic test: raw %.3f, global %.3f, hourly %.3f, rolling %.3f",
            test_metrics["raw_coverage"],
            test_metrics["calibrated_coverage"],
            test_metrics["hourly_calibrated_coverage"],
            test_metrics["rolling_calibrated_coverage"],
        )
    elif args.command == "performance":
        import pandas as pd

        performance_config = PerformanceConfig(
            horizon=args.horizon,
            max_train_hours=args.max_train_hours,
            n_estimators=args.n_estimators,
            warmup_runs=args.warmup_runs,
            repetitions=args.repetitions,
        )
        performance_result = run_performance_benchmark(
            pd.read_parquet(args.input),
            pd.read_parquet(args.weather),
            performance_config,
        )
        write_performance_artifacts(
            performance_result, performance_config, args.output_dir
        )
        fastest = performance_result.measurements.sort_values(
            "prediction_median_ms"
        ).iloc[0]
        LOGGER.info(
            "Fastest weekly inference: %s at %.3f ms median",
            fastest["model"],
            fastest["prediction_median_ms"],
        )
    return 0
