from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gridcast.columns import Col
from gridcast.dashboard_data import (
    DashboardData,
    MissingArtifactsError,
    benchmark_week,
    daily_history,
    display_model,
    load_dashboard_data,
    probabilistic_week,
)

INK = "#17242B"
ORANGE = "#F05D23"
CYAN = "#42B7C8"
PALE_CYAN = "rgba(66, 183, 200, 0.22)"
PAPER = "#F4F1EA"
MUTED = "#69767C"
GRID = "rgba(23, 36, 43, 0.10)"
MODEL_COLORS = {
    "lightgbm_exogenous": "#F05D23",
    "lightgbm_weather": "#E68A3C",
    "lightgbm": "#42B7C8",
    "lightgbm_holidays": "#7D8C91",
    "seasonal_naive_24h": "#5969A6",
    "seasonal_naive_168h": "#9A6FB0",
    "persistence_1h": "#B5AAA0",
}


@st.cache_data(show_spinner=False)
def _load_data() -> DashboardData:
    return load_dashboard_data()


def _chart_layout(figure: go.Figure, height: int = 430) -> go.Figure:
    figure.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=78, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Arial, sans-serif", color=INK),
        hoverlabel=dict(bgcolor=INK, font_color="white", bordercolor=INK),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="right",
            x=1,
        ),
    )
    figure.update_xaxes(gridcolor=GRID, zeroline=False)
    figure.update_yaxes(gridcolor=GRID, zeroline=False)
    return figure


def _format_mw(value: float) -> str:
    return f"{value:,.0f} MW"


def _as_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _render_header() -> None:
    st.markdown(
        """
        <section class="hero">
          <div class="eyebrow">PJME / 2002–2018 / HOURLY</div>
          <h1>GridCast</h1>
          <p>Electricity forecasting under honest temporal validation.</p>
          <div class="hero-rule"></div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def _render_overview(data: DashboardData) -> None:
    summary = data.eda_summary
    load_summary = summary.get("load_mw", {})
    if not isinstance(load_summary, dict):
        load_summary = {}
    test_board = data.leaderboard.loc[data.leaderboard[Col.SPLIT].eq("test")]
    winner = test_board.sort_values("mae").iloc[0]
    probabilistic = data.probabilistic_metrics.loc[
        data.probabilistic_metrics[Col.SPLIT].eq("test")
    ].iloc[0]

    st.markdown("## System pulse")
    st.caption("Headline results from the frozen 52-week test period.")
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Hourly observations",
        f"{int(_as_float(summary.get('observations', 0))):,}",
    )
    winner_name = str(winner[Col.MODEL])
    winner_label = (
        "LightGBM + exog."
        if winner_name == "lightgbm_exogenous"
        else display_model(winner_name)
    )
    metric_columns[1].metric("Best model", winner_label)
    metric_columns[2].metric("Best MAE", _format_mw(float(winner["mae"])))
    metric_columns[3].metric(
        "Calibrated coverage",
        f"{float(probabilistic['calibrated_coverage']):.1%}",
        delta="target 80%",
        delta_color="off",
    )

    history = daily_history(data.history)
    history_chart = go.Figure()
    history_chart.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["mean_load_mw"],
            mode="lines",
            name="Daily mean",
            line=dict(color=CYAN, width=1.2),
        )
    )
    history_chart.add_trace(
        go.Scatter(
            x=history["date"],
            y=history["peak_load_mw"],
            mode="lines",
            name="Daily peak",
            line=dict(color=ORANGE, width=0.8),
            opacity=0.6,
        )
    )
    history_chart.update_layout(title="Sixteen years of PJM East demand")
    history_chart.update_yaxes(title="Load (MW)")
    st.plotly_chart(
        _chart_layout(history_chart, 470),
        width="stretch",
        config={"displayModeBar": False},
    )

    left, right = st.columns([1.35, 1])
    with left:
        _render_leaderboard(data.leaderboard)
    with right:
        st.markdown(
            f"""
            <div class="note-card">
              <div class="eyebrow">DATA PROFILE</div>
              <div class="big-number">{_as_float(load_summary.get("mean", 0)):,.0f}</div>
              <div class="unit">MW average hourly load</div>
              <div class="note-grid">
                <span>Median</span><strong>{_as_float(load_summary.get("median", 0)):,.0f} MW</strong>
                <span>95th percentile</span><strong>{_as_float(load_summary.get("p95", 0)):,.0f} MW</strong>
                <span>Observed maximum</span><strong>{_as_float(load_summary.get("maximum", 0)):,.0f} MW</strong>
                <span>Coverage</span><strong>{_as_float(summary.get("years", 0)):.1f} years</strong>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_leaderboard(leaderboard: pd.DataFrame) -> None:
    test = leaderboard.loc[leaderboard[Col.SPLIT].eq("test")].sort_values(
        "mae", ascending=True
    )
    labels = [display_model(str(model)) for model in test[Col.MODEL]]
    colors = [MODEL_COLORS.get(str(model), MUTED) for model in test[Col.MODEL]]
    chart = go.Figure(
        go.Bar(
            x=test["mae"],
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{value:,.0f}" for value in test["mae"]],
            textposition="outside",
            hovertemplate="%{y}<br>MAE %{x:,.0f} MW<extra></extra>",
        )
    )
    chart.update_layout(title="Frozen-test model ranking", showlegend=False)
    chart.update_xaxes(title="MAE (MW)")
    chart.update_yaxes(autorange="reversed")
    st.plotly_chart(
        _chart_layout(chart, 430),
        width="stretch",
        config={"displayModeBar": False},
    )


def _render_point_forecasts(data: DashboardData) -> None:
    st.markdown("## Point forecast laboratory")
    st.caption(
        "Inspect any frozen weekly fold and compare models against the same actual load."
    )
    test = data.benchmark_forecasts.loc[data.benchmark_forecasts[Col.SPLIT].eq("test")]
    available_models = list(dict.fromkeys(test[Col.MODEL].astype(str)))
    default_models = [
        model
        for model in [
            "lightgbm_exogenous",
            "seasonal_naive_24h",
            "seasonal_naive_168h",
        ]
        if model in available_models
    ]
    controls = st.columns([1, 2.2])
    fold = controls[0].selectbox(
        "Test week",
        options=sorted(test[Col.FOLD].unique(), reverse=True),
        format_func=lambda value: f"Week {value}",
    )
    models = controls[1].multiselect(
        "Models",
        options=available_models,
        default=default_models,
        format_func=display_model,
    )
    if not models:
        st.info("Select at least one model to draw the forecast comparison.")
        return
    week = benchmark_week(data.benchmark_forecasts, "test", int(fold), models)
    actual = week.drop_duplicates(Col.TIMESTAMP)
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=actual[Col.TIMESTAMP],
            y=actual[Col.TARGET],
            mode="lines",
            name="Actual",
            line=dict(color=INK, width=3),
        )
    )
    for model_name in models:
        model = week.loc[week[Col.MODEL].eq(model_name)]
        chart.add_trace(
            go.Scatter(
                x=model[Col.TIMESTAMP],
                y=model[Col.PREDICTION],
                mode="lines",
                name=display_model(model_name),
                line=dict(color=MODEL_COLORS.get(model_name, MUTED), width=2),
            )
        )
    chart.update_layout(title=f"Frozen test week {fold}")
    chart.update_yaxes(title="Load (MW)")
    st.plotly_chart(
        _chart_layout(chart, 520),
        width="stretch",
        config={"displayModeBar": False},
    )

    fold_metrics = data.benchmark_fold_metrics.loc[
        data.benchmark_fold_metrics[Col.SPLIT].eq("test")
        & data.benchmark_fold_metrics[Col.FOLD].eq(fold)
        & data.benchmark_fold_metrics[Col.MODEL].isin(models)
    ].copy()
    fold_metrics["Model"] = fold_metrics[Col.MODEL].map(display_model)
    fold_metrics["MAE (MW)"] = fold_metrics["mae"].round(0).astype(int)
    fold_metrics["RMSE (MW)"] = fold_metrics["rmse"].round(0).astype(int)
    fold_metrics["MASE"] = fold_metrics["mase"].round(3)
    st.dataframe(
        fold_metrics[["Model", "MAE (MW)", "RMSE (MW)", "MASE"]],
        hide_index=True,
        width="stretch",
    )


def _render_uncertainty(data: DashboardData) -> None:
    st.markdown("## Uncertainty, made accountable")
    st.caption(
        "P10–P90 intervals are calibrated on validation only, then frozen for test."
    )
    test = data.probabilistic_forecasts.loc[
        data.probabilistic_forecasts[Col.SPLIT].eq("test")
    ]
    fold = st.selectbox(
        "Test week",
        options=sorted(test[Col.FOLD].unique(), reverse=True),
        format_func=lambda value: f"Week {value}",
        key="probabilistic_fold",
    )
    week = probabilistic_week(data.probabilistic_forecasts, "test", int(fold))
    chart = go.Figure()
    chart.add_trace(
        go.Scatter(
            x=week[Col.TIMESTAMP],
            y=week[Col.P10_CALIBRATED],
            mode="lines",
            line=dict(color="rgba(66,183,200,0)"),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    chart.add_trace(
        go.Scatter(
            x=week[Col.TIMESTAMP],
            y=week[Col.P90_CALIBRATED],
            mode="lines",
            line=dict(color=CYAN, width=1),
            fill="tonexty",
            fillcolor=PALE_CYAN,
            name="Conformal P10–P90",
        )
    )
    chart.add_trace(
        go.Scatter(
            x=week[Col.TIMESTAMP],
            y=week[Col.P50],
            mode="lines",
            name="P50",
            line=dict(color=ORANGE, width=2),
        )
    )
    chart.add_trace(
        go.Scatter(
            x=week[Col.TIMESTAMP],
            y=week[Col.TARGET],
            mode="lines",
            name="Actual",
            line=dict(color=INK, width=2.5),
        )
    )
    chart.update_layout(title=f"Calibrated interval · frozen test week {fold}")
    chart.update_yaxes(title="Load (MW)")
    st.plotly_chart(
        _chart_layout(chart, 520),
        width="stretch",
        config={"displayModeBar": False},
    )

    metrics = data.probabilistic_metrics.loc[
        data.probabilistic_metrics[Col.SPLIT].eq("test")
    ].iloc[0]
    summary_test = data.probabilistic_summary.get("test", {})
    correction = _as_float(data.probabilistic_summary.get("conformal_correction_mw", 0))
    columns = st.columns(4)
    columns[0].metric("Raw coverage", f"{float(metrics['raw_coverage']):.1%}")
    columns[1].metric(
        "Calibrated coverage",
        f"{float(metrics['calibrated_coverage']):.1%}",
        delta=f"{float(metrics['calibrated_coverage'] - metrics['raw_coverage']):.1%}",
    )
    columns[2].metric(
        "Calibrated width", _format_mw(float(metrics["calibrated_mean_width_mw"]))
    )
    columns[3].metric("Correction / bound", _format_mw(correction))

    if isinstance(summary_test, dict):
        coverage = go.Figure(
            go.Bar(
                x=["Target", "Raw", "Conformal"],
                y=[
                    0.8,
                    _as_float(summary_test.get("raw_coverage", 0)),
                    _as_float(summary_test.get("calibrated_coverage", 0)),
                ],
                marker_color=[ORANGE, "#9ADBE5", CYAN],
                texttemplate="%{y:.1%}",
                textposition="outside",
            )
        )
        coverage.update_layout(title="Coverage calibration", showlegend=False)
        coverage.update_yaxes(range=[0, 1], tickformat=".0%")
        st.plotly_chart(
            _chart_layout(coverage, 390),
            width="stretch",
            config={"displayModeBar": False},
        )


def _render_methodology() -> None:
    st.markdown("## Evaluation contract")
    st.caption("The rules that make the headline numbers defensible.")
    st.markdown(
        """
        <div class="method-grid">
          <article><span>01</span><h3>Chronological only</h3><p>No random split. Every forecast is strictly later than its training cutoff.</p></article>
          <article><span>02</span><h3>Frozen final year</h3><p>Fifty-two weekly folds cover every season and remain separate from validation.</p></article>
          <article><span>03</span><h3>Weather without hindsight</h3><p>ERA5 enters through 168/336-hour lags and prior-year climatology, never realized future values.</p></article>
          <article><span>04</span><h3>Validation-only calibration</h3><p>The conformal correction is learned on 12 validation folds before final test evaluation.</p></article>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.warning(
        "Research benchmark: PJME ends in 2018 and one Philadelphia weather point "
        "does not represent the entire PJM East footprint."
    )


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700;800&display=swap');
        html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
        .stApp { background: #F4F1EA; color: #17242B; }
        [data-testid="stHeader"] { background: rgba(244, 241, 234, 0.86); }
        [data-testid="stSidebar"] { background: #17242B; }
        [data-testid="stSidebar"] * { color: #F4F1EA; }
        [data-testid="stSidebar"] [data-baseweb="radio"] label {
          padding: .42rem .2rem; font-weight: 700;
        }
        .block-container { max-width: 1320px; padding-top: 2rem; padding-bottom: 5rem; }
        .hero { padding: 2.4rem 0 1.5rem; }
        .hero h1 { font-size: clamp(4rem, 11vw, 8.8rem); line-height: .82; letter-spacing: -.08em; margin: .35rem 0 1rem; font-weight: 800; }
        .hero p { font-size: clamp(1.05rem, 2vw, 1.5rem); color: #526168; margin: 0; }
        .hero-rule { width: 100%; height: 8px; background: linear-gradient(90deg, #F05D23 0 22%, #17242B 22% 76%, #42B7C8 76%); margin-top: 2rem; }
        .eyebrow { font-family: 'DM Mono', monospace; letter-spacing: .16em; font-size: .72rem; font-weight: 500; color: #F05D23; }
        h2 { letter-spacing: -.035em !important; font-size: clamp(2rem, 4vw, 3.2rem) !important; margin-top: 1.2rem !important; }
        [data-testid="stMetric"] { background: rgba(255,255,255,.45); border-top: 3px solid #17242B; padding: 1rem; min-height: 120px; }
        [data-testid="stMetricLabel"] { font-family: 'DM Mono', monospace; text-transform: uppercase; letter-spacing: .06em; }
        [data-testid="stMetricValue"] { font-weight: 800; letter-spacing: -.045em; }
        .note-card { background: #17242B; color: #F4F1EA; padding: 2rem; min-height: 390px; border-radius: 2px; }
        .note-card .eyebrow { color: #67CFDA; }
        .big-number { font-size: clamp(3.6rem, 7vw, 6.5rem); line-height: 1; letter-spacing: -.07em; font-weight: 800; margin-top: 1.2rem; }
        .unit { color: #B9C3C7; margin-bottom: 2.5rem; }
        .note-grid { display: grid; grid-template-columns: 1fr auto; gap: .85rem 1rem; border-top: 1px solid #526168; padding-top: 1rem; }
        .note-grid span { color: #B9C3C7; }
        .note-grid strong { text-align: right; }
        .method-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 1.5rem 0 2rem; }
        .method-grid article { border-top: 4px solid #F05D23; background: rgba(255,255,255,.5); padding: 1.4rem; min-height: 225px; }
        .method-grid span { font-family: 'DM Mono', monospace; color: #F05D23; }
        .method-grid h3 { margin: 1.4rem 0 .7rem; letter-spacing: -.03em; }
        .method-grid p { color: #526168; line-height: 1.55; }
        .stPlotlyChart { background: rgba(255,255,255,.28); border: 1px solid rgba(23,36,43,.08); }
        @media (max-width: 900px) {
          .block-container { padding: 1rem 1rem 4rem; }
          .hero { padding-top: 1.2rem; }
          .method-grid { grid-template-columns: 1fr 1fr; }
          .method-grid article { min-height: auto; }
        }
        @media (max-width: 560px) {
          .hero h1 { font-size: 4rem; }
          .hero-rule { height: 5px; }
          .method-grid { grid-template-columns: 1fr; }
          [data-testid="stMetric"] { min-height: 100px; }
          .note-card { min-height: auto; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the GridCast Streamlit dashboard."""
    st.set_page_config(
        page_title="GridCast · PJME Forecasting",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="auto",
    )
    _inject_styles()
    try:
        data = _load_data()
    except (MissingArtifactsError, ValueError) as error:
        st.error(str(error))
        st.code("make data weather eda benchmark probabilistic")
        st.stop()
        return

    with st.sidebar:
        st.markdown("### GRIDCAST")
        st.caption("Forecast intelligence console")
        section = st.radio(
            "Navigate",
            ["Overview", "Point forecasts", "Uncertainty", "Methodology"],
            label_visibility="collapsed",
        )
        st.divider()
        st.caption("FROZEN TEST")
        st.markdown("**52 weekly folds**")
        st.caption("Aug 2017 → Aug 2018")
        st.divider()
        st.caption("Built from public PJM + ERA5 data")

    _render_header()
    renderers: dict[str, Any] = {
        "Overview": lambda: _render_overview(data),
        "Point forecasts": lambda: _render_point_forecasts(data),
        "Uncertainty": lambda: _render_uncertainty(data),
        "Methodology": _render_methodology,
    }
    renderers[section]()


if __name__ == "__main__":
    main()
