"""Visual report generation module.

Generates self-contained, interactive HTML reports using Plotly:
- Per-exchange ranking tables sorted by mentions and sentiment score
- Interactive Sentiment vs. Price charts per ticker with Support & Resistance lines
- Out-of-sample Predicted vs. Actual return plots and validation metric tables
- Top news articles with sentiment badges
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.utils import (
    load_parquet,
    load_settings_config,
    setup_logging,
)

logger = logging.getLogger("sentiment_price_agent.report")


def create_price_sentiment_chart(
    df_features: pd.DataFrame,
    summary_df: pd.DataFrame,
    ticker: str,
) -> str:
    """Create interactive dual-axis chart for price, sentiment, and SR levels.

    Args:
        df_features: Hourly features dataframe.
        summary_df: Per-ticker summary dataframe.
        ticker: Ticker symbol.

    Returns:
        HTML div string from Plotly.
    """
    tkr_feat = df_features[df_features["ticker"] == ticker].sort_values("ts_utc")
    tkr_sum = summary_df[summary_df["ticker"] == ticker]

    if tkr_feat.empty:
        return f"<div class='empty-chart'>No price data for {ticker}</div>"

    support_val = float(tkr_sum["support"].iloc[0]) if not tkr_sum.empty else 0.0
    resistance_val = float(tkr_sum["resistance"].iloc[0]) if not tkr_sum.empty else 0.0
    pivot_val = float(tkr_sum["pivot_point"].iloc[0]) if not tkr_sum.empty else 0.0

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=(f"{ticker} Hourly Close Price & Key Levels", "Engagement-Weighted Sentiment"),
        row_heights=[0.7, 0.3],
    )

    # 1. Price Line
    fig.add_trace(
        go.Scatter(
            x=tkr_feat["ts_utc"],
            y=tkr_feat["close"],
            name="Close Price",
            mode="lines+markers",
            line=dict(color="#2979ff", width=2),
            marker=dict(size=4),
        ),
        row=1,
        col=1,
    )

    # Add Support & Resistance horizontal lines
    if support_val > 0:
        fig.add_hline(
            y=support_val,
            line_dash="dash",
            line_color="#00e676",
            line_width=1.5,
            annotation_text=f"Support ({support_val:.2f})",
            annotation_position="top right",
            row=1,
            col=1,
        )

    if resistance_val > 0:
        fig.add_hline(
            y=resistance_val,
            line_dash="dash",
            line_color="#ff1744",
            line_width=1.5,
            annotation_text=f"Resistance ({resistance_val:.2f})",
            annotation_position="bottom right",
            row=1,
            col=1,
        )

    if pivot_val > 0:
        fig.add_hline(
            y=pivot_val,
            line_dash="dot",
            line_color="#ffa726",
            line_width=1.2,
            annotation_text=f"Pivot ({pivot_val:.2f})",
            annotation_position="bottom left",
            row=1,
            col=1,
        )

    # 2. Sentiment Bars
    sent_colors = [
        "#00e676" if s > 0.05 else ("#ff1744" if s < -0.05 else "#78909c")
        for s in tkr_feat["engagement_weighted_sentiment"]
    ]
    fig.add_trace(
        go.Bar(
            x=tkr_feat["ts_utc"],
            y=tkr_feat["engagement_weighted_sentiment"],
            name="Hourly Sentiment",
            marker_color=sent_colors,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#1e222d",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=480,
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment (-1 to 1)", row=2, col=1, range=[-1.05, 1.05])

    return fig.to_html(full_html=False, include_plotlyjs=False)


def create_predictions_vs_actual_chart(df_preds: pd.DataFrame) -> str:
    """Create interactive plot comparing predicted next-bar returns to actual returns."""
    if df_preds.empty:
        return "<div class='empty-chart'>No validation predictions available</div>"

    fig = go.Figure()

    # Actual returns
    fig.add_trace(
        go.Scatter(
            x=df_preds["ts_utc"],
            y=df_preds["actual_return"],
            name="Actual Return",
            mode="lines",
            line=dict(color="#90a4ae", width=1.5),
            opacity=0.7,
        )
    )

    # LightGBM Predictions
    fig.add_trace(
        go.Scatter(
            x=df_preds["ts_utc"],
            y=df_preds["lgbm_pred"],
            name="LightGBM Predicted",
            mode="lines",
            line=dict(color="#00e5ff", width=2),
        )
    )

    # Ridge Predictions
    fig.add_trace(
        go.Scatter(
            x=df_preds["ts_utc"],
            y=df_preds["ridge_pred"],
            name="Ridge Predicted",
            mode="lines",
            line=dict(color="#ab47bc", width=1.5, dash="dot"),
        )
    )

    fig.update_layout(
        title="Walk-Forward Out-of-Sample Predictions vs. Actual (Next-Bar Return)",
        template="plotly_dark",
        paper_bgcolor="#131722",
        plot_bgcolor="#1e222d",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=380,
        yaxis=dict(title="Next-Bar Return (n+1)"),
    )

    return fig.to_html(full_html=False, include_plotlyjs=False)


def render_html_table(df_subset: pd.DataFrame) -> str:
    """Render a styled HTML ranking table for an exchange."""
    html_parts = [
        "<div class='table-responsive'>",
        "<table class='data-table'>",
        "<thead><tr>",
        "<th>Ticker</th><th>Asset Name</th><th>Price</th><th>Sentiment</th><th>Direction</th><th>Position</th><th>Support</th><th>Resistance</th><th>Pivot</th><th>Mentions</th>",
        "</tr></thead><tbody>",
    ]

    for _, r in df_subset.iterrows():
        dir_val = str(r["direction"]).lower()
        if dir_val == "bullish":
            badge_cls = "badge-bullish"
            icon = "&#9650; BULLISH"
        elif dir_val == "bearish":
            badge_cls = "badge-bearish"
            icon = "&#9660; BEARISH"
        else:
            badge_cls = "badge-neutral"
            icon = "&#9644; NEUTRAL"

        sent_score = float(r["sentiment"])
        sent_color = "#00e676" if sent_score > 0.05 else ("#ff1744" if sent_score < -0.05 else "#b0bec5")

        html_parts.append(
            f"<tr>"
            f"<td><strong>{r['ticker']}</strong></td>"
            f"<td>{r['asset_name']}</td>"
            f"<td class='num-cell'>{float(r['current_price']):,.2f}</td>"
            f"<td class='num-cell' style='color:{sent_color}; font-weight:600;'>{sent_score:+.3f}</td>"
            f"<td><span class='badge {badge_cls}'>{icon}</span></td>"
            f"<td><span class='badge badge-pos'>{r['position'].upper()}</span></td>"
            f"<td class='num-cell support-lvl'>{float(r['support']):,.2f}</td>"
            f"<td class='num-cell resist-lvl'>{float(r['resistance']):,.2f}</td>"
            f"<td class='num-cell pivot-lvl'>{float(r['pivot_point']):,.2f}</td>"
            f"<td class='num-cell'><strong>{int(r['mention_count'])}</strong></td>"
            f"</tr>"
        )

    html_parts.append("</tbody></table></div>")
    return "\n".join(html_parts)


def generate_html_report(
    summary_path: Union[str, Path] = "data/processed/ticker_summary.parquet",
    features_path: Union[str, Path] = "data/processed/features_hourly.parquet",
    predictions_path: Union[str, Path] = "data/processed/predictions.parquet",
    news_path: Union[str, Path] = "data/processed/news.parquet",
    metrics_json_path: Union[str, Path] = "reports/metrics.json",
    output_path: Union[str, Path] = "reports/report.html",
) -> Path:
    """Generate self-contained interactive Plotly HTML report."""
    summary_df = load_parquet(summary_path)
    features_df = load_parquet(features_path)
    predictions_df = load_parquet(predictions_path)
    news_df = load_parquet(news_path)

    metrics_dict: dict[str, Any] = {}
    m_path = Path(metrics_json_path)
    if m_path.is_file():
        with m_path.open("r", encoding="utf-8") as f:
            metrics_dict = json.load(f)

    # 1. Exchange Tables
    hkex_df = summary_df[summary_df["exchange"] == "HKEX"].sort_values(
        by=["mention_count", "sentiment"], ascending=[False, False]
    )
    krx_df = summary_df[summary_df["exchange"] == "KRX"].sort_values(
        by=["mention_count", "sentiment"], ascending=[False, False]
    )

    hkex_table_html = render_html_table(hkex_df)
    krx_table_html = render_html_table(krx_df)

    # 2. Charts for Top Tickers
    hkex_top_ticker = hkex_df["ticker"].iloc[0] if not hkex_df.empty else "0700.HK"
    krx_top_ticker = krx_df["ticker"].iloc[0] if not krx_df.empty else "005930.KS"

    chart_hkex = create_price_sentiment_chart(features_df, summary_df, hkex_top_ticker)
    chart_krx = create_price_sentiment_chart(features_df, summary_df, krx_top_ticker)

    # 3. Model Predictions Chart
    preds_chart_html = create_predictions_vs_actual_chart(predictions_df)

    # 4. News Section
    news_cards = []
    if not news_df.empty:
        for ticker, grp in news_df.groupby("ticker"):
            items_html = []
            for _, item in grp.head(2).iterrows():
                items_html.append(
                    f"<div class='news-item'>"
                    f"<div class='news-title'><a href='{item['url']}' target='_blank' rel='noopener'>{item['title']}</a></div>"
                    f"<div class='news-date'>{item['date']}</div>"
                    f"<div class='news-snippet'>{str(item['snippet'])[:200]}...</div>"
                    f"</div>"
                )
            news_cards.append(
                f"<div class='news-card'>"
                f"<h4>Ticker: {ticker}</h4>"
                f"{''.join(items_html)}"
                f"</div>"
            )
    news_section_html = "<div class='news-grid'>" + "".join(news_cards) + "</div>" if news_cards else "<p>No recent news articles ingested.</p>"

    # 5. Metrics Cards
    metrics_cards_html = []
    for scope_name, s_data in metrics_dict.items():
        n_m = s_data.get("overall_naive", {})
        r_m = s_data.get("overall_ridge", {})
        l_m = s_data.get("overall_lgbm", {})
        l_vs = s_data.get("lgbm_vs_baseline", {})

        badge_txt = "BEATS BASELINE" if l_vs.get("beats_baseline_mae") else "UNDERPERFORMS"
        badge_cls = "badge-bullish" if l_vs.get("beats_baseline_mae") else "badge-bearish"

        metrics_cards_html.append(
            f"""
            <div class='metric-card'>
              <div class='metric-card-header'>
                <h3>{scope_name} (n={s_data.get('total_test_samples', 0)})</h3>
                <span class='badge {badge_cls}'>{badge_txt}</span>
              </div>
              <table class='metric-table'>
                <tr><th>Model</th><th>MAE</th><th>RMSE</th><th>Dir. Accuracy</th></tr>
                <tr><td>Naive Baseline</td><td>{n_m.get('mae', 0):.6f}</td><td>{n_m.get('rmse', 0):.6f}</td><td>{n_m.get('directional_accuracy', 0)*100:.1f}%</td></tr>
                <tr><td>Ridge</td><td>{r_m.get('mae', 0):.6f}</td><td>{r_m.get('rmse', 0):.6f}</td><td>{r_m.get('directional_accuracy', 0)*100:.1f}%</td></tr>
                <tr class='highlight-row'><td>LightGBM</td><td>{l_m.get('mae', 0):.6f}</td><td>{l_m.get('rmse', 0):.6f}</td><td>{l_m.get('directional_accuracy', 0)*100:.1f}%</td></tr>
              </table>
              <div class='metric-footer'>LightGBM MAE Imp: <strong>{l_vs.get('mae_improvement_pct', 0):+.2f}%</strong> | Dir. Acc Diff: <strong>{l_vs.get('directional_accuracy_diff_pct', 0):+.1f}%</strong></div>
            </div>
            """
        )

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Sentiment-Price-Agent Dashboard | HKEX & KRX</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --bg: #0b0e14;
      --card-bg: #151922;
      --border: #262c38;
      --text: #e0e6ed;
      --text-muted: #8b949e;
      --bullish: #00e676;
      --bearish: #ff1744;
      --neutral: #78909c;
      --accent: #2979ff;
    }}
    body {{
      margin: 0;
      padding: 24px;
      background: var(--bg);
      color: var(--text);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--border);
      margin-bottom: 24px;
    }}
    .header h1 {{
      margin: 0;
      font-size: 26px;
      font-weight: 700;
      letter-spacing: -0.5px;
    }}
    .subtitle {{
      color: var(--text-muted);
      font-size: 14px;
      margin-top: 4px;
    }}
    .status-pill {{
      background: rgba(41, 121, 255, 0.15);
      border: 1px solid var(--accent);
      color: #82b1ff;
      padding: 6px 14px;
      border-radius: 20px;
      font-size: 13px;
      font-weight: 600;
    }}
    .section-title {{
      font-size: 18px;
      font-weight: 600;
      margin: 28px 0 14px 0;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 18px;
      margin-bottom: 24px;
    }}
    .table-responsive {{
      overflow-x: auto;
    }}
    .data-table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 13px;
    }}
    .data-table th {{
      background: #1c212c;
      padding: 10px 12px;
      color: var(--text-muted);
      border-bottom: 1px solid var(--border);
      font-weight: 600;
    }}
    .data-table td {{
      padding: 11px 12px;
      border-bottom: 1px solid var(--border);
    }}
    .data-table tr:hover {{
      background: rgba(255, 255, 255, 0.02);
    }}
    .num-cell {{
      text-align: right;
      font-variant-numeric: tabular-nums;
    }}
    .support-lvl {{ color: #69f0ae; }}
    .resist-lvl {{ color: #ff5252; }}
    .pivot-lvl {{ color: #ffd740; }}
    .badge {{
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.3px;
    }}
    .badge-bullish {{ background: rgba(0, 230, 118, 0.15); color: #00e676; }}
    .badge-bearish {{ background: rgba(255, 23, 68, 0.15); color: #ff1744; }}
    .badge-neutral {{ background: rgba(120, 144, 156, 0.15); color: #b0bec5; }}
    .badge-pos {{ background: rgba(41, 121, 255, 0.15); color: #82b1ff; }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 20px;
    }}
    @media (max-width: 1024px) {{
      .grid-2 {{ grid-template-columns: 1fr; }}
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }}
    .metric-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 16px;
    }}
    .metric-card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 12px;
    }}
    .metric-card-header h3 {{ margin: 0; font-size: 15px; }}
    .metric-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .metric-table th, .metric-table td {{
      padding: 6px 8px;
      text-align: right;
    }}
    .metric-table th:first-child, .metric-table td:first-child {{
      text-align: left;
    }}
    .highlight-row {{
      background: rgba(41, 121, 255, 0.1);
      font-weight: 600;
    }}
    .metric-footer {{
      margin-top: 10px;
      padding-top: 8px;
      border-top: 1px dashed var(--border);
      font-size: 11px;
      color: var(--text-muted);
    }}
    .news-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
      gap: 16px;
    }}
    .news-card {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
    }}
    .news-card h4 {{ margin: 0 0 10px 0; font-size: 14px; color: var(--accent); }}
    .news-item {{
      margin-bottom: 12px;
      padding-bottom: 8px;
      border-bottom: 1px solid rgba(255, 255, 255, 0.05);
    }}
    .news-title a {{
      color: var(--text);
      text-decoration: none;
      font-size: 13px;
      font-weight: 500;
    }}
    .news-title a:hover {{ text-decoration: underline; color: #82b1ff; }}
    .news-date {{ font-size: 11px; color: var(--text-muted); margin: 3px 0; }}
    .news-snippet {{ font-size: 12px; color: #90a4ae; line-height: 1.4; }}
    .disclaimer {{
      margin-top: 40px;
      padding: 16px;
      background: #11141b;
      border-left: 3px solid #ff9100;
      font-size: 12px;
      color: var(--text-muted);
      border-radius: 0 6px 6px 0;
    }}
  </style>
</head>
<body>

  <div class="header">
    <div>
      <h1>Sentiment-Price-Agent Dashboard</h1>
      <div class="subtitle">Multi-Exchange Algorithmic Sentiment & Next-Bar Price Prediction Pipeline</div>
    </div>
    <div class="status-pill">&#9679; UTC Synchronized &bull; Walk-Forward Validated</div>
  </div>

  <!-- Validation Metrics Section -->
  <div class="section-title">&#128202; Walk-Forward Model Validation Metrics (Honest Out-of-Sample)</div>
  <div class="metric-grid">
    {''.join(metrics_cards_html)}
  </div>

  <!-- Out-of-Sample Predictions Chart -->
  <div class="card">
    {preds_chart_html}
  </div>

  <!-- Exchange Ranking Tables -->
  <div class="section-title">&#127469;&#127472; HKEX Ticker Rankings (Sorted by Mentions & Sentiment)</div>
  <div class="card">
    {hkex_table_html}
  </div>

  <div class="section-title">&#127472;&#127479; KRX Ticker Rankings (Sorted by Mentions & Sentiment)</div>
  <div class="card">
    {krx_table_html}
  </div>

  <!-- Interactive Price & Sentiment Charts -->
  <div class="section-title">&#128200; Interactive Price, Support/Resistance & Sentiment Charts</div>
  <div class="grid-2">
    <div class="card">{chart_hkex}</div>
    <div class="card">{chart_krx}</div>
  </div>

  <!-- Top News Section -->
  <div class="section-title">&#128240; Recent Financial News per Ticker (Exa API)</div>
  <div class="card">
    {news_section_html}
  </div>

  <div class="disclaimer">
    <strong>Empirical Research Disclaimer:</strong> This automated report is generated for academic and quantitative research purposes only. Financial markets possess substantial noise; out-of-sample past performance does not guarantee future results. None of the contents constitute financial, investment, or trading advice.
  </div>

</body>
</html>
"""

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        f.write(full_html)

    logger.info("Generated self-contained HTML report at %s (%d KB)", out_file, out_file.stat().st_size // 1024)
    return out_file


if __name__ == "__main__":
    setup_logging()
    rep = generate_html_report()
    print(f"HTML report successfully generated at {rep}")
