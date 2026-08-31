import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from gridcast.columns import Col
from gridcast.pjm import validate_hourly_load

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def create_eda_report(data: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    """Create summary tables and plots for an hourly load dataset.

    Parameters
    ----------
    data : pandas.DataFrame
        Canonical hourly load data.
    output_dir : pathlib.Path
        Directory receiving JSON, CSV, and PNG artifacts.

    Returns
    -------
    dict
        JSON-compatible descriptive summary.
    """
    validate_hourly_load(data)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamps = data[Col.TIMESTAMP]
    target = data[Col.TARGET]
    peak_position = int(target.to_numpy().argmax())
    summary: dict[str, object] = {
        "observations": len(data),
        "start": timestamps.iloc[0].isoformat(),
        "end": timestamps.iloc[-1].isoformat(),
        "years": round((timestamps.iloc[-1] - timestamps.iloc[0]).days / 365.25, 2),
        "load_mw": {
            "mean": float(target.mean()),
            "standard_deviation": float(target.std()),
            "minimum": float(target.min()),
            "p05": float(target.quantile(0.05)),
            "median": float(target.median()),
            "p95": float(target.quantile(0.95)),
            "maximum": float(target.max()),
        },
        "peak_timestamp": timestamps.iloc[peak_position].isoformat(),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    enriched = data.assign(
        **{
            Col.DATE: timestamps.dt.floor("D"),
            Col.HOUR: timestamps.dt.hour,
            Col.WEEKDAY: timestamps.dt.day_name(),
        }
    )
    daily = enriched.groupby(Col.DATE, as_index=False).agg({Col.TARGET: "mean"})
    hourly = enriched.groupby(Col.HOUR, as_index=False)[Col.TARGET].agg(["mean", "std"])
    weekday_order = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]
    weekly = (
        enriched.groupby([Col.WEEKDAY, Col.HOUR], as_index=False)[Col.TARGET]
        .mean()
        .pivot(index=Col.WEEKDAY, columns=Col.HOUR, values=Col.TARGET)
        .reindex(weekday_order)
    )
    daily.to_csv(output_dir / "daily_load.csv", index=False)
    hourly.to_csv(output_dir / "hourly_profile.csv", index=False)
    weekly.to_csv(output_dir / "weekly_profile.csv")
    _plot_history(daily, output_dir / "load_history.png")
    _plot_profiles(hourly, weekly, output_dir / "seasonal_profiles.png")
    return summary


def _plot_history(daily: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(14, 5))
    axis.plot(daily[Col.DATE], daily[Col.TARGET], color="#15616d", linewidth=0.7)
    axis.set(title="PJME daily average electricity load", ylabel="Load (MW)")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _plot_profiles(
    hourly: pd.DataFrame, weekly: pd.DataFrame, output_path: Path
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    mean = hourly["mean"].to_numpy(dtype=float)
    standard_deviation = hourly["std"].to_numpy(dtype=float)
    hours = hourly.index.to_numpy(dtype=float)
    axes[0].plot(hours, mean, color="#15616d", linewidth=2)
    axes[0].fill_between(
        hours,
        mean - standard_deviation,
        mean + standard_deviation,
        color="#15616d",
        alpha=0.15,
    )
    axes[0].set(
        title="Average intraday profile",
        xlabel="Hour",
        ylabel="Load (MW)",
    )
    axes[0].set_xticks(np.arange(0, 24, 3))
    image = axes[1].imshow(weekly.to_numpy(), aspect="auto", cmap="YlOrRd")
    axes[1].set(
        title="Average weekly load profile",
        xlabel="Hour",
        ylabel="Day",
        yticks=np.arange(len(weekly.index)),
        yticklabels=[day[:3] for day in weekly.index],
    )
    axes[1].set_xticks(np.arange(0, 24, 3))
    figure.colorbar(image, ax=axes[1], label="Load (MW)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
