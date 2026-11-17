"""Sentiment scoring module.

Provides a provider-agnostic sentiment scorer supporting Gemini and Anthropic
for financial posts and news in English, Traditional Chinese (Cantonese slang), and Korean.
(Full implementation in Stage 2).
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger("sentiment_price_agent.sentiment")


class SentimentScorer:
    """Scorer interface for LLM sentiment analysis."""

    def __init__(self, provider: Optional[str] = None) -> None:
        self.provider = provider or "gemini"

    def score_batch(self, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Score a batch of posts or news articles."""
        raise NotImplementedError("Implemented in Stage 2")
