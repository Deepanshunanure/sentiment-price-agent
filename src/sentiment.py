"""Sentiment scoring module.

Provides a provider-agnostic scorer supporting Gemini and Anthropic with strict JSON schema,
batch processing, cache-first retrieval, robust JSON repair, and native support
for Traditional Chinese (including Cantonese financial slang) and Korean.
"""

from __future__ import annotations

import json
import logging
import os
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional, Union

import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.utils import (
    get_cache_path,
    get_env_var,
    load_parquet,
    load_settings_config,
    load_tickers_config,
    normalize_to_utc,
    read_cache,
    save_parquet,
    setup_logging,
    write_cache,
)

logger = logging.getLogger("sentiment_price_agent.sentiment")


class SentimentRecord(BaseModel):
    """Pydantic model validating the strict sentiment output schema."""

    id: str
    sentiment: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score from -1.0 to 1.0")
    direction: Literal["bullish", "bearish", "neutral"] = Field(..., description="Directional bias")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    position: Literal["long", "short", "none"] = Field(default="none", description="Author trading position")
    support_resistance_levels: list[float] = Field(
        default_factory=list, description="Price levels stated by author"
    )
    reasoning: Optional[str] = Field(default=None, description="Brief rationale")


SYSTEM_PROMPT = """You are a senior financial analyst and quantitative sentiment researcher specializing in Asia-Pacific equity markets (HKEX and KRX).
Analyze the provided batch of retail investor posts and news headlines for stock market sentiment, directional conviction, declared positions, and price levels.

Linguistic and domain context:
- Traditional Chinese / Cantonese Slang (Hong Kong / HKEX):
  * Bullish cues: 升 (surge), 睇好 (bullish), 牛證/牛 (bull certificate), 爆升 (skyrocket), 破頂 (breakout), 入貨/加注 (buy/add), 坐定等數錢 (sit and count profits)
  * Bearish cues: 跌 (fall), 睇淡 (bearish), 熊證/熊 (bear certificate), 沽/沽空 (short), 穿底/破位 (breakdown), 蟹/大閘蟹 (bagholder), 溝上溝落 (trapped averaging down), 玩完 (finished)
- Korean (KRX / Korean retail investors):
  * Bullish cues: 떡상 (massive moon/surge), 롱 (long), 매수 (buy), 추매 (additional buy), 존버 (hodl to the end), 우상향 (upward trajectory)
  * Bearish cues: 떡락 (crash/dump), 숏 (short), 매도/손절 (sell/cut loss), 물렸다 (bagholding/trapped), 폭락 (plunge), 개미 털기 (retail shakeout)

Strict Output Instructions:
For each input item, produce an object conforming strictly to this JSON schema:
[
  {
    "id": "<matching input id>",
    "sentiment": <float between -1.0 and 1.0, where -1.0 is extreme panic/bearish, 0.0 is neutral, +1.0 is extreme euphoria/bullish>,
    "direction": "bullish" | "bearish" | "neutral",
    "confidence": <float between 0.0 and 1.0>,
    "position": "long" | "short" | "none",
    "support_resistance_levels": [<numbers mentioned as target, support, or resistance price levels, in stock price unit>],
    "reasoning": "<concise 1-sentence explanation>"
  }
]
Return ONLY a valid JSON array. Do not include markdown code block syntax or extra text outside the JSON.
"""


def repair_and_parse_json(raw_text: str) -> list[dict[str, Any]]:
    """Clean and repair malformed JSON strings from LLM outputs.

    Handles markdown fences, unescaped newlines, trailing commas, and partial responses.
    """
    cleaned = raw_text.strip()
    # Strip markdown fences if present
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = cleaned.strip()

    # Direct parse attempt
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Sometimes model wraps array in {"results": [...]}
            for key in ["results", "items", "data", "scored"]:
                if key in data and isinstance(data[key], list):
                    return data[key]
            return [data]
    except Exception as exc:
        logger.debug("Initial JSON parse failed: %s. Attempting regex extraction.", exc)

    # Regex extraction of JSON array
    array_match = re.search(r"\[\s*\{.*?\}\s*\]", cleaned, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except Exception:
            pass

    # Extract individual JSON objects
    object_matches = re.findall(r"\{[^{}]*\}", cleaned, re.DOTALL)
    results = []
    for m in object_matches:
        try:
            obj = json.loads(m)
            results.append(obj)
        except Exception:
            continue

    if results:
        return results

    raise ValueError(f"Unable to parse or repair JSON from LLM output: {raw_text[:200]}...")


class BaseSentimentProvider(ABC):
    """Abstract interface for LLM sentiment providers."""

    @abstractmethod
    def call_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Call LLM with a batch of text items and return raw parsed dictionaries."""
        pass


class GeminiSentimentProvider(BaseSentimentProvider):
    """Google Gemini sentiment provider using google.genai."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or get_env_var("GEMINI_API_KEY", required=True)
        from google import genai

        self.client = genai.Client(api_key=self.api_key)
        self.preferred_models = [
            "gemini-3.1-flash-lite",
            "gemini-3.5-flash-lite",
            "gemini-3.8-flash",
        ]

    def _call_model_with_fallback(self, prompt: str) -> str:
        """Attempt call across available Gemini model candidates."""
        from google.genai import types

        last_err = None
        for model in self.preferred_models:
            try:
                response = self.client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        system_instruction=SYSTEM_PROMPT,
                        temperature=0.1,
                    ),
                )
                if response and response.text:
                    return response.text
            except Exception as e:
                last_err = e
                logger.warning("Gemini model %s call failed: %s. Trying fallback model...", model, e)
                continue

        raise RuntimeError(f"All Gemini model candidates failed. Last error: {last_err}")

    def call_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        batch_prompt = "Analyze the sentiment for the following items:\n" + json.dumps(
            [{"id": b["id"], "ticker": b["ticker"], "exchange": b["exchange"], "text": b["text"]} for b in batch],
            ensure_ascii=False,
            indent=2,
        )
        raw_text = self._call_model_with_fallback(batch_prompt)
        return repair_and_parse_json(raw_text)


class AnthropicSentimentProvider(BaseSentimentProvider):
    """Anthropic Claude sentiment provider."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or get_env_var("ANTHROPIC_API_KEY", required=True)
        import anthropic

        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = "claude-3-5-haiku-20241022"

    def call_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        batch_prompt = f"{SYSTEM_PROMPT}\n\nAnalyze the sentiment for these items and return a JSON array:\n" + json.dumps(
            [{"id": b["id"], "ticker": b["ticker"], "exchange": b["exchange"], "text": b["text"]} for b in batch],
            ensure_ascii=False,
            indent=2,
        )
        msg = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            temperature=0.1,
            messages=[{"role": "user", "content": batch_prompt}],
        )
        raw_text = msg.content[0].text
        return repair_and_parse_json(raw_text)


class RuleBasedFallbackProvider(BaseSentimentProvider):
    """Heuristic fallback provider when external LLM APIs are unreachable or unconfigured."""

    BULLISH_CANT_TERMS = ["升", "睇好", "牛", "爆升", "破頂", "加注", "入貨", "好消息", "業績好", "買入"]
    BEARISH_CANT_TERMS = ["跌", "睇淡", "熊", "沽", "穿底", "破位", "蟹", "減持", "業績差", "沽空"]

    BULLISH_KR_TERMS = ["떡상", "롱", "매수", "추매", "존버", "우상향", "상승", "호재"]
    BEARISH_KR_TERMS = ["떡락", "숏", "매도", "손절", "물렸다", "폭락", "하락", "악재"]

    BULLISH_EN_TERMS = ["bullish", "rally", "surge", "gain", "buy", "outperform", "upgrade", "growth", "high"]
    BEARISH_EN_TERMS = ["bearish", "plunge", "drop", "sell", "underperform", "downgrade", "decline", "low", "risk"]

    def call_batch(self, batch: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results = []
        for b in batch:
            text = b.get("text", "")
            score = 0.0

            # Count cues
            for term in self.BULLISH_CANT_TERMS + self.BULLISH_KR_TERMS + self.BULLISH_EN_TERMS:
                if term in text:
                    score += 0.3
            for term in self.BEARISH_CANT_TERMS + self.BEARISH_KR_TERMS + self.BEARISH_EN_TERMS:
                if term in text:
                    score -= 0.3

            score = max(-1.0, min(1.0, score))
            if score > 0.15:
                direction = "bullish"
            elif score < -0.15:
                direction = "bearish"
            else:
                direction = "neutral"

            # Detect price levels (digits followed by . or pure numbers)
            levels = []
            numbers = re.findall(r"\b\d{2,6}(?:\.\d+)?\b", text)
            for n in numbers[:3]:
                try:
                    val = float(n)
                    if 0.5 <= val <= 2000000:
                        levels.append(val)
                except ValueError:
                    pass

            results.append(
                {
                    "id": b["id"],
                    "sentiment": round(score, 2),
                    "direction": direction,
                    "confidence": 0.6 if direction != "neutral" else 0.5,
                    "position": "long" if direction == "bullish" else ("short" if direction == "bearish" else "none"),
                    "support_resistance_levels": levels,
                    "reasoning": f"Heuristic lexicon classification based on detected keywords (score={score:.2f})",
                }
            )
        return results


class SentimentScorer:
    """Unified, provider-agnostic sentiment scoring manager with disk caching."""

    def __init__(
        self,
        provider: Optional[str] = None,
        cache_dir: str = "data/cache",
        use_cache: bool = True,
    ) -> None:
        self.provider_name = (provider or os.getenv("SCORER_PROVIDER", "gemini")).lower()
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.backend = self._init_backend()

    def _init_backend(self) -> BaseSentimentProvider:
        """Instantiate the configured provider backend with fallback."""
        if self.provider_name == "gemini":
            try:
                return GeminiSentimentProvider()
            except Exception as e:
                logger.warning("Failed to initialize Gemini provider (%s). Falling back to rule-based.", e)
                return RuleBasedFallbackProvider()
        elif self.provider_name == "anthropic":
            try:
                return AnthropicSentimentProvider()
            except Exception as e:
                logger.warning("Failed to initialize Anthropic provider (%s). Falling back to rule-based.", e)
                return RuleBasedFallbackProvider()
        elif self.provider_name == "rule_based":
            return RuleBasedFallbackProvider()
        else:
            logger.warning("Unknown SCORER_PROVIDER '%s'. Defaulting to rule-based.", self.provider_name)
            return RuleBasedFallbackProvider()

    def score_items(self, items: list[dict[str, Any]], batch_size: int = 10) -> list[SentimentRecord]:
        """Score a list of input items with batching and cache lookups.

        Args:
            items: List of dictionaries with keys: id, text, ticker, exchange.
            batch_size: Number of items per LLM call.

        Returns:
            List of validated SentimentRecord objects.
        """
        results_map: dict[str, SentimentRecord] = {}
        uncached_items: list[dict[str, Any]] = []

        # 1. Cache lookup
        for item in items:
            item_id = item["id"]
            cache_key = f"sentiment_{self.provider_name}_{item_id}"
            if self.use_cache:
                cached = read_cache("sentiment", cache_key, self.cache_dir)
                if cached:
                    try:
                        results_map[item_id] = SentimentRecord(**cached)
                        continue
                    except Exception:
                        pass
            uncached_items.append(item)

        logger.info(
            "Sentiment scoring: %d items total (%d cached, %d to process)",
            len(items),
            len(results_map),
            len(uncached_items),
        )

        # 2. Process uncached in batches
        for i in range(0, len(uncached_items), batch_size):
            batch = uncached_items[i : i + batch_size]
            logger.info("Scoring batch %d/%d (%d items)...", i // batch_size + 1, (len(uncached_items) - 1) // batch_size + 1, len(batch))

            try:
                scored_batch = self.backend.call_batch(batch)
            except Exception as exc:
                logger.error("LLM scoring batch failed (%s). Using fallback for this batch.", exc)
                scored_batch = RuleBasedFallbackProvider().call_batch(batch)

            scored_by_id = {str(s.get("id")): s for s in scored_batch if "id" in s}

            for orig in batch:
                orig_id = str(orig["id"])
                raw_rec = scored_by_id.get(orig_id)

                if raw_rec is None:
                    # Single-item fallback if missing from LLM response
                    fb = RuleBasedFallbackProvider().call_batch([orig])[0]
                    raw_rec = fb

                # Sanitize and validate through Pydantic
                try:
                    # Clean sentiment bounds
                    val_sent = float(raw_rec.get("sentiment", 0.0))
                    val_sent = max(-1.0, min(1.0, val_sent))
                    val_conf = float(raw_rec.get("confidence", 0.5))
                    val_conf = max(0.0, min(1.0, val_conf))
                    val_dir = str(raw_rec.get("direction", "neutral")).lower()
                    if val_dir not in ("bullish", "bearish", "neutral"):
                        val_dir = "bullish" if val_sent > 0.1 else ("bearish" if val_sent < -0.1 else "neutral")
                    val_pos = str(raw_rec.get("position", "none")).lower()
                    if val_pos not in ("long", "short", "none"):
                        val_pos = "none"

                    levels = []
                    for l in raw_rec.get("support_resistance_levels") or []:
                        try:
                            levels.append(float(l))
                        except (ValueError, TypeError):
                            pass

                    record = SentimentRecord(
                        id=orig_id,
                        sentiment=round(val_sent, 3),
                        direction=val_dir,
                        confidence=round(val_conf, 3),
                        position=val_pos,
                        support_resistance_levels=levels,
                        reasoning=str(raw_rec.get("reasoning", "")),
                    )
                except ValidationError as ve:
                    logger.warning("Validation error on item %s: %s; applying fallback.", orig_id, ve)
                    record = SentimentRecord(
                        id=orig_id,
                        sentiment=0.0,
                        direction="neutral",
                        confidence=0.5,
                        position="none",
                        support_resistance_levels=[],
                        reasoning="Fallback default after validation error",
                    )

                results_map[orig_id] = record

                # Cache result
                if self.use_cache:
                    cache_key = f"sentiment_{self.provider_name}_{orig_id}"
                    write_cache("sentiment", cache_key, record.model_dump(), self.cache_dir)

        # Return ordered results
        return [results_map[str(item["id"])] for item in items]


def run_sentiment_pipeline(
    posts_path: Union[str, Path] = "data/processed/posts.parquet",
    news_path: Union[str, Path] = "data/processed/news.parquet",
    output_path: Union[str, Path] = "data/processed/scored.parquet",
    settings_config_path: Union[str, Path] = "config/settings.yaml",
) -> pd.DataFrame:
    """Execute end-to-end sentiment scoring on ingested posts and news.

    Args:
        posts_path: Path to processed posts parquet.
        news_path: Path to processed news parquet.
        output_path: Path to destination scored.parquet.
        settings_config_path: Path to settings.yaml.

    Returns:
        DataFrame with combined scored data.
    """
    settings_cfg = load_settings_config(settings_config_path)
    use_cache = settings_cfg.get("cache", {}).get("enabled", True)
    cache_dir = settings_cfg.get("cache", {}).get("cache_dir", "data/cache")

    scorer = SentimentScorer(cache_dir=cache_dir, use_cache=use_cache)

    items_to_score: list[dict[str, Any]] = []

    # 1. Load posts
    df_posts = load_parquet(posts_path)
    if not df_posts.empty:
        for idx, row in df_posts.iterrows():
            item_id = f"post_{row.get('exchange', 'HKEX')}_{row.get('ticker')}_{idx}"
            items_to_score.append(
                {
                    "id": item_id,
                    "item_type": "post",
                    "ts_utc": normalize_to_utc(row.get("ts_utc")),
                    "exchange": row.get("exchange", "HKEX"),
                    "source": row.get("source", "social"),
                    "ticker": row.get("ticker", ""),
                    "text": str(row.get("text", "")),
                    "likes": int(row.get("likes") or 0),
                    "reposts": int(row.get("reposts") or 0),
                    "views": int(row.get("views") or 0),
                    "author_followers": int(row.get("author_followers") or 0),
                }
            )

    # 2. Load news
    df_news = load_parquet(news_path)
    if not df_news.empty:
        for idx, row in df_news.iterrows():
            item_id = f"news_{row.get('ticker')}_{idx}"
            title = str(row.get("title", ""))
            snippet = str(row.get("snippet", ""))
            combined_text = f"{title}. {snippet}"[:800]
            items_to_score.append(
                {
                    "id": item_id,
                    "item_type": "news",
                    "ts_utc": normalize_to_utc(row.get("date")),
                    "exchange": "HKEX" if str(row.get("ticker", "")).endswith(".HK") else "KRX",
                    "source": "exa_news",
                    "ticker": row.get("ticker", ""),
                    "text": combined_text,
                    "likes": 0,
                    "reposts": 0,
                    "views": 0,
                    "author_followers": 0,
                }
            )

    if not items_to_score:
        logger.warning("No items found to score in %s and %s", posts_path, news_path)
        empty_df = pd.DataFrame(
            columns=[
                "id",
                "item_type",
                "ts_utc",
                "exchange",
                "source",
                "ticker",
                "text",
                "sentiment",
                "direction",
                "confidence",
                "position",
                "support_resistance_levels",
                "reasoning",
                "likes",
                "reposts",
                "views",
                "author_followers",
            ]
        )
        save_parquet(empty_df, output_path)
        return empty_df

    logger.info("Scoring %d items (%d posts, %d news)...", len(items_to_score), len(df_posts), len(df_news))
    scored_records = scorer.score_items(items_to_score, batch_size=8)

    # Assemble final DataFrame
    records_by_id = {r.id: r for r in scored_records}
    final_rows: list[dict[str, Any]] = []

    for item in items_to_score:
        rec = records_by_id.get(item["id"])
        row = dict(item)
        if rec:
            row["sentiment"] = rec.sentiment
            row["direction"] = rec.direction
            row["confidence"] = rec.confidence
            row["position"] = rec.position
            row["support_resistance_levels"] = rec.support_resistance_levels
            row["reasoning"] = rec.reasoning
        else:
            row["sentiment"] = 0.0
            row["direction"] = "neutral"
            row["confidence"] = 0.5
            row["position"] = "none"
            row["support_resistance_levels"] = []
            row["reasoning"] = "Fallback default"

        final_rows.append(row)

    df_scored = pd.DataFrame(final_rows)
    save_parquet(df_scored, output_path)
    logger.info("Successfully saved %d scored items to %s", len(df_scored), output_path)
    return df_scored


if __name__ == "__main__":
    setup_logging()
    res = run_sentiment_pipeline()
    print("\n--- Sentiment Scoring Summary ---")
    print(f"Total items scored: {len(res)}")
    if not res.empty:
        print("\nDirection breakdown:")
        print(res["direction"].value_counts())
        print("\nPosition breakdown:")
        print(res["position"].value_counts())
        print("\nAverage sentiment by ticker:")
        print(res.groupby(["exchange", "ticker"])["sentiment"].agg(["count", "mean", "std"]))
