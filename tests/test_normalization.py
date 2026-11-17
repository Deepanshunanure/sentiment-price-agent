"""Unit tests for timestamp normalization and ticker matching."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from src.ingest_apify import match_post_to_tickers
from src.utils import normalize_to_utc


def test_normalize_to_utc_iso_string():
    """Verify ISO string with explicit offset converts cleanly to UTC."""
    raw = "2026-11-17T15:30:00+08:00"
    res = normalize_to_utc(raw)
    assert res.tzinfo is not None
    assert res.tz_convert("UTC").hour == 7
    assert res.year == 2026


def test_normalize_to_utc_unix_timestamp():
    """Verify Unix seconds and milliseconds convert correctly to UTC."""
    # 1700000000 in seconds: 2023-11-14 22:13:20 UTC
    res_s = normalize_to_utc(1700000000)
    assert res_s.year == 2023
    assert res_s.month == 11
    assert res_s.tzinfo is not None

    # Milliseconds
    res_ms = normalize_to_utc(1700000000000)
    assert res_ms.year == 2023
    assert res_ms.month == 11


def test_normalize_to_utc_naive_with_default_tz():
    """Verify naive datetime string localized to specified timezone then converted to UTC."""
    raw = "2026-11-17 15:30:00"
    res = normalize_to_utc(raw, default_tz="Asia/Hong_Kong")
    assert res.hour == 7  # 15:30 - 8 hours = 07:30 UTC


def test_match_post_to_tickers_hkex():
    """Verify matching of HKEX tickers via cashtag, symbol, and Chinese keywords."""
    ticker_defs = [
        {
            "ticker": "0700.HK",
            "keywords": ["騰訊控股", "騰訊"],
            "cashtags": ["#0700.HK", "#700"],
        },
        {
            "ticker": "9988.HK",
            "keywords": ["阿里巴巴", "阿里"],
            "cashtags": ["#9988.HK"],
        },
    ]

    # Keyword match
    text_kw = "今日騰訊業績好過預期，值得加注"
    matched = match_post_to_tickers(text_kw, ticker_defs)
    assert "0700.HK" in matched

    # Cashtag match
    text_tag = "Breaking news on #0700.HK and #9988.HK"
    matched_tags = match_post_to_tickers(text_tag, ticker_defs)
    assert "0700.HK" in matched_tags
    assert "9988.HK" in matched_tags


def test_match_post_to_tickers_krx():
    """Verify matching of KRX tickers via Korean keywords and codes."""
    ticker_defs = [
        {
            "ticker": "005930.KS",
            "keywords": ["삼성전자", "삼전"],
            "cashtags": ["#005930"],
        },
        {
            "ticker": "000660.KS",
            "keywords": ["SK하이닉스", "하이닉스"],
            "cashtags": ["#000660"],
        },
    ]

    text_kr = "오늘 삼전 주가 상승세 대단하네요. 하이닉스도 같이 오릅니다."
    matched = match_post_to_tickers(text_kr, ticker_defs)
    assert "005930.KS" in matched
    assert "000660.KS" in matched
