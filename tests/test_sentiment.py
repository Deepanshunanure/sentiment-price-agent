"""Unit tests for sentiment scoring and JSON repair."""

import pytest
from src.sentiment import (
    RuleBasedFallbackProvider,
    SentimentRecord,
    repair_and_parse_json,
)


def test_sentiment_record_valid():
    """Verify valid SentimentRecord creation and bounds enforcement."""
    rec = SentimentRecord(
        id="test_1",
        sentiment=0.75,
        direction="bullish",
        confidence=0.9,
        position="long",
        support_resistance_levels=[380.0, 400.0],
        reasoning="Strong earnings momentum",
    )
    assert rec.sentiment == 0.75
    assert rec.direction == "bullish"
    assert rec.position == "long"
    assert len(rec.support_resistance_levels) == 2


def test_sentiment_record_bounds():
    """Verify validation errors when sentiment is out of bounds."""
    with pytest.raises(Exception):
        SentimentRecord(
            id="bad_1",
            sentiment=1.5,  # Exceeds max 1.0
            direction="bullish",
            confidence=0.8,
        )


def test_repair_and_parse_json_markdown():
    """Verify repair of markdown code fences."""
    raw = """```json
    [
      {
        "id": "1",
        "sentiment": 0.5,
        "direction": "bullish",
        "confidence": 0.8,
        "position": "long",
        "support_resistance_levels": [100.5]
      }
    ]
    ```"""
    parsed = repair_and_parse_json(raw)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "1"
    assert parsed[0]["sentiment"] == 0.5


def test_repair_and_parse_json_wrapped_dict():
    """Verify repair when LLM wraps list in a dict under 'results'."""
    raw = '{"results": [{"id": "2", "sentiment": -0.4, "direction": "bearish", "confidence": 0.7, "position": "short"}]}'
    parsed = repair_and_parse_json(raw)
    assert len(parsed) == 1
    assert parsed[0]["id"] == "2"
    assert parsed[0]["direction"] == "bearish"


def test_rule_based_provider_cantonese_and_korean():
    """Verify multilingual handling of Cantonese slang and Korean keywords."""
    provider = RuleBasedFallbackProvider()
    batch = [
        {
            "id": "cant_bull",
            "ticker": "0700.HK",
            "exchange": "HKEX",
            "text": "騰訊爆升，睇好破頂，今日入貨加注！目標 420",
        },
        {
            "id": "kr_bear",
            "ticker": "005930.KS",
            "exchange": "KRX",
            "text": "삼전 떡락해서 손절하고 숏 친다. 지지선 190000 붕괴됨",
        },
    ]
    results = provider.call_batch(batch)
    assert len(results) == 2

    # Cantonese bullish
    res_cant = results[0]
    assert res_cant["direction"] == "bullish"
    assert res_cant["sentiment"] > 0
    assert 420.0 in res_cant["support_resistance_levels"]

    # Korean bearish
    res_kr = results[1]
    assert res_kr["direction"] == "bearish"
    assert res_kr["sentiment"] < 0
    assert 190000.0 in res_kr["support_resistance_levels"]
