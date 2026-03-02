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

    items: list[NewsItem] = []
    seen_headlines: set[str] = set()
    for article in raw:
        headline = article.get("headline") or ""
        if not headline or headline in seen_headlines:
            continue
        if _JUNK_RE.search(headline):
            continue
        seen_headlines.add(headline)

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
                summary=summary[:500],
                source=source,
                symbols=syms if isinstance(syms, list) else [],
                published_at=published,
                url=url,
            )
        )

    items.sort(key=lambda n: n.published_at, reverse=True)
    log(f"fetched {len(items)} news articles")
    return items


def format_news_for_llm(items: list[NewsItem], max_items: int = 30) -> str:
    if not items:
        return "No recent news available."
    lines = []
    for item in items[:max_items]:
        lines.append(item.to_context_str())
    return "\n\n".join(lines)
