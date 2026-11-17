"""Main pipeline orchestration CLI for sentiment-price-agent.

Usage:
    python -m src.run --all        # Run the complete end-to-end pipeline
    python -m src.run --ingest     # Ingest OHLCV prices, news, and social posts
    python -m src.run --score      # Score sentiment using configured LLM provider
    python -m src.run --features   # Build hourly feature matrix and key levels
    python -m src.run --model      # Run walk-forward model validation
    python -m src.run --report     # Generate interactive visual HTML report
    python -m src.run --smoke      # Run quick smoke test on 1 ticker per exchange
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from src.features import run_stage3_pipeline
from src.ingest_apify import ingest_social_pipeline
from src.ingest_exa import ingest_news_pipeline
from src.model import run_modeling_pipeline
from src.prices import ingest_prices_pipeline
from src.report import generate_html_report
from src.sentiment import run_sentiment_pipeline
from src.smoke_test_stage1 import run_stage1_smoke_test
from src.utils import setup_logging

logger = logging.getLogger("sentiment_price_agent.cli")


def run_full_pipeline() -> None:
    """Execute the complete production pipeline from raw ingestion to HTML report."""
    start_time = time.time()
    print("\n" + "=" * 80)
    print("STARTING FULL PIPELINE EXECUTION (sentiment-price-agent)")
    print("=" * 80)

    # 1. Ingestion
    print("\n>>> STEP 1/5: Multi-Source Data Ingestion (Prices, News, Social)")
    p_h, p_d = ingest_prices_pipeline()
    print(f"  - Ingested {len(p_h)} hourly bars and {len(p_d)} daily bars.")
    news_df = ingest_news_pipeline()
    print(f"  - Ingested {len(news_df)} financial news articles via Exa.")
    posts_df = ingest_social_pipeline()
    print(f"  - Ingested {len(posts_df)} normalized social posts via Apify.")

    # 2. Sentiment Scoring
    print("\n>>> STEP 2/5: LLM Sentiment Scoring & Trading Context Extraction")
    scored_df = run_sentiment_pipeline()
    print(f"  - Scored {len(scored_df)} items with strict JSON schema.")

    # 3. Feature Engineering & Levels
    print("\n>>> STEP 3/5: Hourly Feature Engineering & Key Levels Detection")
    features_df, summary_df = run_stage3_pipeline()
    print(f"  - Built feature matrix: {features_df.shape[0]} rows x {features_df.shape[1]} columns.")
    print(f"  - Generated per-ticker summary for {len(summary_df)} equities.")

    # 4. Walk-Forward Prediction & Validation
    print("\n>>> STEP 4/5: Next-Bar (n+1) Prediction & Walk-Forward Validation")
    preds_df, metrics_dict = run_modeling_pipeline()
    print(f"  - Generated {len(preds_df)} out-of-sample predictions across walk-forward folds.")

    # 5. Visual Report Generation
    print("\n>>> STEP 5/5: Generating Interactive Visual HTML Report")
    report_file = generate_html_report()
    print(f"  - Visual report generated successfully: {report_file}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"FULL PIPELINE EXECUTION FINISHED IN {elapsed:.1f} SECONDS")
    print(f"Interactive Dashboard: {Path(report_file).resolve()}")
    print("=" * 80 + "\n")


def serve_ui(port: int = 8080) -> None:
    """Serve reports/report.html locally and open in the default web browser."""
    import http.server
    import socketserver
    import webbrowser

    report_path = Path("reports/report.html").resolve()
    if not report_path.is_file():
        print("Report not found. Generating now...")
        generate_html_report()

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(report_path.parent), **kwargs)

        def log_message(self, format, *args):
            pass  # Suppress noise

    url = f"http://127.0.0.1:{port}/report.html"
    print(f"\nServing interactive UI dashboard at {url}")
    print("Opening browser... (Press Ctrl+C to stop the server)\n")
    webbrowser.open(url)

    socketserver.TCPServer.allow_reuse_address = True
    try:
        with socketserver.TCPServer(("127.0.0.1", port), QuietHandler) as httpd:
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nUI Server stopped.")


def main() -> None:
    """CLI entrypoint."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Sentiment-Price-Agent: Algorithmic sentiment & price prediction pipeline for HKEX and KRX."
    )
    parser.add_argument("--all", action="store_true", help="Run the complete end-to-end pipeline")
    parser.add_argument("--ui", "--serve", dest="ui", action="store_true", help="Launch visual web dashboard in browser")
    parser.add_argument("--ingest", action="store_true", help="Ingest prices, news, and social posts")
    parser.add_argument("--score", action="store_true", help="Score sentiment with LLM provider")
    parser.add_argument("--features", action="store_true", help="Build hourly features and support/resistance levels")
    parser.add_argument("--model", action="store_true", help="Run walk-forward prediction and validation")
    parser.add_argument("--report", action="store_true", help="Generate interactive HTML report")
    parser.add_argument("--smoke", action="store_true", help="Run 1-ticker smoke test")

    args = parser.parse_args()

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    if args.ui:
        serve_ui()
    elif args.smoke:
        run_stage1_smoke_test()
    elif args.all:
        run_full_pipeline()
    else:
        if args.ingest:
            print("Running data ingestion...")
            ingest_prices_pipeline()
            ingest_news_pipeline()
            ingest_social_pipeline()
        if args.score:
            print("Running sentiment scoring...")
            run_sentiment_pipeline()
        if args.features:
            print("Building features and levels...")
            run_stage3_pipeline()
        if args.model:
            print("Running walk-forward model validation...")
            run_modeling_pipeline()
        if args.report:
            print("Generating visual HTML report...")
            generate_html_report()


if __name__ == "__main__":
    main()
