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


@dataclass(frozen=True)
class NewsEvent:
    headline: str
    summary: str
    source_count: int
    article_count: int
    symbols: list[str]
    event_type: str
    freshness: str
    first_seen: datetime
    last_seen: datetime
    supporting_sources: list[str]
    supporting_headlines: list[str]
    age_minutes: int = 0

    def to_context_str(self) -> str:
        symbols_str = ", ".join(self.symbols) if self.symbols else "general"
        lines = [
            (
                f"[{self.freshness.upper()} | {self.event_type} | "
                f"{self.source_count} sources | {self.age_minutes}m ago] {symbols_str}"
            ),
            f"  Event: {self.headline}",
        ]
        if self.summary:
            lines.append(f"  Summary: {self.summary}")
        if self.supporting_sources:
            lines.append(f"  Sources: {', '.join(self.supporting_sources[:4])}")
        if self.article_count > 1:
            extras = self.supporting_headlines[1:3]
            if extras:
                lines.append(f"  Related headlines: {' | '.join(extras)}")
        return "\n".join(lines)


_TITLE_PREFIX_RE = re.compile(
    r"^(update\s*\d*:|breaking:|watch:|live:|exclusive:|analysis:)\s*",
    re.IGNORECASE,
)
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "by", "for", "from", "in", "into", "of",
    "on", "or", "the", "to", "with", "after", "before", "amid", "over", "under",
}
_EVENT_PATTERNS: list[tuple[str, tuple[str, ...]]] = [
    ("earnings", ("earnings", "revenue", "eps", "guidance", "quarter", "q1", "q2", "q3", "q4")),
    ("guidance", ("guidance", "outlook", "forecast", "raises", "cuts", "sees")),
    ("merger", ("acquire", "acquisition", "merge", "merger", "takeover", "buyout")),
    ("fda", ("fda", "phase 3", "trial", "approval", "clinical", "drug")),
    ("partnership", ("partnership", "partner", "deal", "contract", "agreement", "collaboration")),
    ("product", ("launch", "product", "chip", "platform", "release", "model")),
    ("analyst", ("upgrades", "downgrade", "price target", "analyst", "rating")),
    ("management", ("ceo", "cfo", "chairman", "executive", "board", "resigns")),
    ("financing", ("offering", "debt", "financing", "convertible", "share sale", "repurchase")),
    ("legal", ("lawsuit", "court", "settlement", "probe", "investigation", "regulator")),
    ("macro", ("fed", "inflation", "payrolls", "cpi", "ppi", "jobs", "treasury", "tariff")),
]
_EVENT_WEIGHTS = {
    "earnings": 3.0,
    "guidance": 3.0,
    "merger": 3.0,
    "fda": 3.0,
    "partnership": 2.2,
    "product": 2.0,
    "macro": 2.0,
    "management": 1.5,
    "financing": 1.5,
    "legal": 1.2,
    "analyst": 1.0,
    "general": 1.0,
}
_FRESHNESS_WEIGHTS = {
    "breaking": 3.0,
    "fresh": 2.0,
    "developing": 1.2,
    "stale": 0.5,
}


def _normalize_headline(text: str) -> str:
    lowered = _TITLE_PREFIX_RE.sub("", text.strip().lower())
    lowered = _NON_ALNUM_RE.sub(" ", lowered)
    return " ".join(lowered.split())


def _headline_signature(text: str) -> str:
    normalized = _normalize_headline(text)
    tokens = [tok for tok in normalized.split() if tok not in _STOPWORDS]
    if not tokens:
        return normalized[:80]
    return " ".join(tokens[:8])


def _classify_event_type(headline: str, summary: str) -> str:
    haystack = f"{headline} {summary}".lower()
    for event_type, keywords in _EVENT_PATTERNS:
        if any(keyword in haystack for keyword in keywords):
            return event_type
    return "general"


def _freshness_label(last_seen: datetime, reference_time: datetime | None = None) -> str:
    reference = reference_time or now_eastern()
    age_hours = max((reference - last_seen.astimezone(reference.tzinfo)).total_seconds() / 3600, 0.0)
    if age_hours <= 0.5:
        return "breaking"
    if age_hours <= 3:
        return "fresh"
    if age_hours <= 12:
        return "developing"
    return "stale"


def build_news_events(
    items: list[NewsItem],
    reference_time: datetime | None = None,
    max_events: int = 18,
) -> list[NewsEvent]:
    """Aggregate raw articles into deduped event packets for the model."""
    if not items:
        return []

    groups: dict[tuple[tuple[str, ...], str], list[NewsItem]] = {}
    for item in items:
        symbols = tuple(sorted({s.upper() for s in item.symbols if s}))
        key = (symbols, _headline_signature(item.headline))
        groups.setdefault(key, []).append(item)

    events: list[NewsEvent] = []
    for grouped_items in groups.values():
        grouped_items.sort(key=lambda item: item.published_at, reverse=True)
        latest = grouped_items[0]
        first_seen = min(item.published_at for item in grouped_items)
        last_seen = max(item.published_at for item in grouped_items)
        reference = reference_time or now_eastern()
        age_minutes = max(int((reference - last_seen.astimezone(reference.tzinfo)).total_seconds() / 60), 0)
        supporting_sources = list(dict.fromkeys(item.source for item in grouped_items if item.source))
        event = NewsEvent(
            headline=latest.headline,
            summary=latest.summary[:500],
            source_count=len(supporting_sources),
            article_count=len(grouped_items),
            symbols=list(dict.fromkeys(sym.upper() for item in grouped_items for sym in item.symbols if sym)),
            event_type=_classify_event_type(latest.headline, latest.summary),
            freshness=_freshness_label(last_seen, reference_time),
            first_seen=first_seen,
            last_seen=last_seen,
            supporting_sources=supporting_sources,
            supporting_headlines=[item.headline for item in grouped_items],
            age_minutes=age_minutes,
        )
        events.append(event)

    events.sort(
        key=lambda event: (
            event.last_seen,
            event.source_count,
            event.article_count,
        ),
        reverse=True,
    )
    return events[:max_events]


def rank_symbols_from_events(
    events: list[NewsEvent],
    focus_symbols: list[str] | None = None,
    max_symbols: int = 20,
) -> list[str]:
    """Rank symbols by corroborated, fresh event flow plus existing focus."""
    scores: dict[str, float] = {}
    focus_set = {sym.upper() for sym in (focus_symbols or [])}
    for event in events:
        event_weight = _EVENT_WEIGHTS.get(event.event_type, 1.0)
        freshness_weight = _FRESHNESS_WEIGHTS.get(event.freshness, 1.0)
        corroboration = 1.0 + 0.35 * max(event.source_count - 1, 0)
        article_boost = 1.0 + 0.15 * max(event.article_count - 1, 0)
        for symbol in event.symbols:
            if not symbol:
                continue
            focus_boost = 1.25 if symbol in focus_set else 1.0
            scores[symbol] = scores.get(symbol, 0.0) + event_weight * freshness_weight * corroboration * article_boost * focus_boost

    for symbol in focus_set:
        scores.setdefault(symbol, 0.0)
        scores[symbol] += 0.75

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [symbol for symbol, _ in ranked[:max_symbols]]


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

    reference_time = max(item.published_at for item in items)
    events = build_news_events(items, reference_time=reference_time)
    focus_set = {s.upper() for s in (focus_symbols or [])}
    ranked_symbols = rank_symbols_from_events(events, focus_symbols=focus_symbols, max_symbols=8)

    def _split_items(news_items: list[NewsItem]) -> tuple[list[NewsItem], list[NewsItem]]:
        targeted: list[NewsItem] = []
        general: list[NewsItem] = []
        for item in news_items:
            if focus_set and any(s.upper() in focus_set for s in item.symbols):
                targeted.append(item)
            elif not focus_set and item.symbols:
                targeted.append(item)
            else:
                general.append(item)
        return targeted, general

    def _split_events(news_events: list[NewsEvent]) -> tuple[list[NewsEvent], list[NewsEvent]]:
        targeted: list[NewsEvent] = []
        general: list[NewsEvent] = []
        for event in news_events:
            if focus_set and any(s.upper() in focus_set for s in event.symbols):
                targeted.append(event)
            elif not focus_set and event.symbols:
                targeted.append(event)
            else:
                general.append(event)
        return targeted, general

    targeted_items, general_items = _split_items(items[:max_items])
    targeted_events, general_events = _split_events(events)

    sections: list[str] = []
    if ranked_symbols:
        sections.append(f"--- Suggested focus symbols ---\n{', '.join(ranked_symbols)}")

    if targeted_events:
        sections.append("--- Structured event map for your active / high-priority tickers ---")
        for event in targeted_events[:8]:
            sections.append(event.to_context_str())

    if targeted_items:
        sections.append("--- Raw headlines for those tickers ---")
        for item in targeted_items[:12]:
            sections.append(item.to_context_str())

    if general_events:
        sections.append("--- Structured broader market event map ---")
        for event in general_events[:6]:
            sections.append(event.to_context_str())

    if general_items:
        sections.append("--- Raw broader market headlines ---")
        for item in general_items[:12]:
            sections.append(item.to_context_str())

    if not sections:
        for item in items[:max_items]:
            sections.append(item.to_context_str())

    return "\n\n".join(sections) if sections else "No recent news available."
