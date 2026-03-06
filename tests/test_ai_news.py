from datetime import datetime, timedelta

from ai_trader.news import (
    NewsEvent,
    NewsItem,
    build_news_events,
    format_news_for_llm,
    rank_symbols_from_events,
)
from ai_trader.utils import EASTERN_TZ


def _news_item(
    headline: str,
    source: str,
    symbols: list[str],
    published_at: datetime,
    summary: str = "",
) -> NewsItem:
    return NewsItem(
        headline=headline,
        summary=summary or headline,
        source=source,
        symbols=symbols,
        published_at=published_at,
        url="",
    )


def test_build_news_events_groups_duplicate_headlines():
    now = datetime(2025, 1, 10, 10, 0, tzinfo=EASTERN_TZ)
    items = [
        _news_item(
            "Breaking: NVDA tops earnings estimates, raises outlook",
            "Reuters",
            ["NVDA"],
            now - timedelta(minutes=10),
        ),
        _news_item(
            "NVDA tops earnings estimates and raises outlook",
            "Bloomberg",
            ["NVDA"],
            now - timedelta(minutes=7),
        ),
    ]

    events = build_news_events(items, reference_time=now)
    assert len(events) == 1
    event = events[0]
    assert event.article_count == 2
    assert event.source_count == 2
    assert event.event_type == "earnings"
    assert event.freshness == "breaking"


def test_rank_symbols_from_events_prefers_fresh_corroborated_events():
    now = datetime(2025, 1, 10, 10, 0, tzinfo=EASTERN_TZ)
    events = [
        NewsEvent(
            headline="NVDA raises AI guidance",
            summary="Strong multi-source catalyst",
            source_count=3,
            article_count=3,
            symbols=["NVDA"],
            event_type="guidance",
            freshness="breaking",
            first_seen=now - timedelta(minutes=20),
            last_seen=now - timedelta(minutes=5),
            supporting_sources=["Reuters", "Bloomberg", "CNBC"],
            supporting_headlines=["NVDA raises AI guidance"],
        ),
        NewsEvent(
            headline="TSLA analyst cuts target",
            summary="Single-source downgrade",
            source_count=1,
            article_count=1,
            symbols=["TSLA"],
            event_type="analyst",
            freshness="stale",
            first_seen=now - timedelta(hours=18),
            last_seen=now - timedelta(hours=18),
            supporting_sources=["Benzinga"],
            supporting_headlines=["TSLA analyst cuts target"],
        ),
    ]

    ranked = rank_symbols_from_events(events)
    assert ranked[0] == "NVDA"
    assert "TSLA" in ranked


def test_format_news_for_llm_outputs_structured_and_raw_sections(monkeypatch):
    now = datetime(2025, 1, 10, 10, 0, tzinfo=EASTERN_TZ)
    monkeypatch.setattr("ai_trader.news.now_eastern", lambda: now)
    items = [
        _news_item(
            "AAPL signs major enterprise AI partnership",
            "Reuters",
            ["AAPL", "MSFT"],
            now - timedelta(minutes=25),
            summary="Apple expands enterprise AI distribution.",
        ),
        _news_item(
            "AAPL signs major enterprise AI partnership",
            "Bloomberg",
            ["AAPL", "MSFT"],
            now - timedelta(minutes=20),
            summary="Deal deepens enterprise channel access.",
        ),
        _news_item(
            "Fed officials signal patience on rate cuts",
            "WSJ",
            ["SPY"],
            now - timedelta(hours=2),
            summary="Macro backdrop remains mixed.",
        ),
    ]

    text = format_news_for_llm(items, focus_symbols=["AAPL"])
    assert "Suggested focus symbols" in text
    assert "Structured event map" in text
    assert "Raw headlines for those tickers" in text
    assert "3 sources" not in text
    assert "2 sources" in text
    assert "AAPL signs major enterprise AI partnership" in text
