"""News aggregation from Alpaca and Polygon."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta

from . import config
from .alpaca_client import AlpacaClient
from .utils import log, now_eastern, parse_timestamp

# ---------------------------------------------------------------------------
# Junk news filter — remove lawsuit spam & corporate housekeeping
# ---------------------------------------------------------------------------
_JUNK_TITLE_PATTERNS = [
    # Lawsuit/legal spam
    "securities fraud",
    "class action",
    "loss recovery",
    "reminds investors",
    "encourages.*investors",
    "investor rights",
    "deadline reminder",
    "deadline tuesday",
    "deadline alert",
    "recover your losses",
    "leading law firm",
    "have suffered losses",
    "securities class action",
    # Corporate housekeeping
    "inducement grant",
    "listing rule 5635",
    "repurchase of own shares",
    "omien osakkeiden",
]

_JUNK_RE = re.compile("|".join(_JUNK_TITLE_PATTERNS), re.IGNORECASE)


@dataclass(frozen=True)
class NewsItem:
    headline: str
    summary: str
    source: str
    symbols: list[str]
    published_at: datetime
    url: str

    def to_context_str(self) -> str:
        symbols_str = ", ".join(self.symbols) if self.symbols else "general"
        ts = self.published_at.strftime("%Y-%m-%d %H:%M ET")
        lines = [
            f"[{ts}] ({self.source}) {self.headline}",
            f"  Symbols: {symbols_str}",
        ]
        if self.summary:
            lines.append(f"  Summary: {self.summary}")
        return "\n".join(lines)


def _parse_articles(raw: list[dict], seen: set[str]) -> list[NewsItem]:
    """Parse raw Alpaca articles into NewsItems, filtering junk and dupes."""
    items: list[NewsItem] = []
    for article in raw:
        headline = article.get("headline") or ""
        if not headline or headline in seen:
            continue
        if _JUNK_RE.search(headline):
            continue
        seen.add(headline)

        summary = article.get("summary") or article.get("content") or ""
        source = article.get("source") or "unknown"
        syms = article.get("symbols") or []
        published = parse_timestamp(
            article.get("created_at") or article.get("updated_at")
        )
        if published is None:
            published = now_eastern()
        url = article.get("url") or ""

        items.append(
            NewsItem(
                headline=headline,
                summary=summary[:1000],
                source=source,
                symbols=syms if isinstance(syms, list) else [],
                published_at=published,
                url=url,
            )
        )
    return items


def fetch_news(
    alpaca: AlpacaClient,
    symbols: list[str] | None = None,
    lookback_hours: int | None = None,
) -> list[NewsItem]:
    """Fetch recent news from Alpaca news API."""
    lookback = lookback_hours or config.NEWS_LOOKBACK_HOURS
    start_time = now_eastern() - timedelta(hours=lookback)
    start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    try:
        raw = alpaca.get_news(symbols=symbols, limit=50, start=start_str)
    except Exception as exc:
        log(f"news fetch error: {exc}")
        return []

    seen: set[str] = set()
    items = _parse_articles(raw, seen)
    items.sort(key=lambda n: n.published_at, reverse=True)
    log(f"fetched {len(items)} news articles")
    return items


def fetch_targeted_news(
    alpaca: AlpacaClient,
    focus_symbols: list[str],
    lookback_hours: int | None = None,
) -> list[NewsItem]:
    """Fetch news in two tiers: targeted for focus symbols, then general.

    Returns targeted news first so the model sees what matters before the noise.
    """
    lookback = lookback_hours or config.NEWS_LOOKBACK_HOURS
    start_time = now_eastern() - timedelta(hours=lookback)
    start_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    seen: set[str] = set()

    # Tier 1: news for tickers the model actually cares about
    targeted: list[NewsItem] = []
    if focus_symbols:
        # Alpaca caps symbols param; chunk if needed
        unique_syms = list(dict.fromkeys(focus_symbols))[:20]
        try:
            raw = alpaca.get_news(symbols=unique_syms, limit=50, start=start_str)
            targeted = _parse_articles(raw, seen)
        except Exception as exc:
            log(f"targeted news fetch error: {exc}")

    # Tier 2: general market headlines (fewer, to fill context with signal not noise)
    general: list[NewsItem] = []
    try:
        raw = alpaca.get_news(symbols=None, limit=30, start=start_str)
        general = _parse_articles(raw, seen)  # seen set dedupes against tier 1
    except Exception as exc:
        log(f"general news fetch error: {exc}")

    targeted.sort(key=lambda n: n.published_at, reverse=True)
    general.sort(key=lambda n: n.published_at, reverse=True)

    combined = targeted + general
    log(f"fetched {len(targeted)} targeted + {len(general)} general news articles")
    return combined


def format_news_for_llm(
    items: list[NewsItem],
    max_items: int = 40,
    focus_symbols: list[str] | None = None,
) -> str:
    if not items:
        return "No recent news available."

    if not focus_symbols:
        # No split — just show everything
        lines = []
        for item in items[:max_items]:
            lines.append(item.to_context_str())
        return "\n\n".join(lines)

    focus_set = {s.upper() for s in focus_symbols}
    targeted: list[NewsItem] = []
    general: list[NewsItem] = []
    for item in items:
        if any(s.upper() in focus_set for s in item.symbols):
            targeted.append(item)
        else:
            general.append(item)

    sections: list[str] = []

    # Targeted: up to 15 articles about tickers the model is watching
    if targeted:
        sections.append("--- News for your active tickers ---")
        for item in targeted[:15]:
            sections.append(item.to_context_str())

    # General: up to 25 articles for new thesis discovery
    if general:
        sections.append("\n--- Broader market news (scan for new opportunities) ---")
        for item in general[:25]:
            sections.append(item.to_context_str())

    return "\n\n".join(sections) if sections else "No recent news available."
