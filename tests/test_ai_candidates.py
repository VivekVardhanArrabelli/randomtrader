"""Tests for broad candidate triage helpers."""

from datetime import datetime, timedelta

from ai_trader.candidates import (
    CandidateIdea,
    build_candidate_ideas,
    format_candidate_table,
    select_candidate_finalists,
)
from ai_trader.news import NewsEvent
from ai_trader.utils import EASTERN_TZ


def test_build_candidate_ideas_prefers_stronger_direct_catalyst():
    now = datetime(2025, 1, 10, 10, 0, tzinfo=EASTERN_TZ)
    source_tags = {
        "AAPL": {"direct"},
        "XLK": {"macro"},
    }
    metrics = {
        "AAPL": {
            "intraday_chg": 1.5,
            "five_d_chg": 4.0,
            "trend": "up",
            "reaction": "early_move",
        },
        "XLK": {
            "intraday_chg": 0.4,
            "five_d_chg": 1.1,
            "trend": "up",
            "reaction": "",
        },
    }
    aapl_event = NewsEvent(
        headline="AAPL signs major enterprise AI partnership",
        summary="Strong corroborated catalyst",
        source_count=3,
        article_count=3,
        symbols=["AAPL"],
        event_type="partnership",
        freshness="breaking",
        first_seen=now - timedelta(minutes=25),
        last_seen=now - timedelta(minutes=5),
        supporting_sources=["Reuters", "Bloomberg", "CNBC"],
        supporting_headlines=["AAPL signs major enterprise AI partnership"],
        age_minutes=5,
    )
    candidates = build_candidate_ideas(
        source_tags,
        metrics,
        {"AAPL": aapl_event},
        {"AAPL": 7.5},
    )
    assert candidates[0].symbol == "AAPL"
    table = format_candidate_table(candidates, max_rows=2)
    assert "Candidate Table" in table
    assert "AAPL" in table
    assert "event=partnership/breaking/3src/5m" in table


def test_select_candidate_finalists_uses_bucket_coverage_before_backfill():
    candidates = [
        CandidateIdea("POS", "position", ("position",), 9.0, 0.5, 1.0, "up", "active_move"),
        CandidateIdea("THS", "thesis", ("thesis",), 8.5, 0.4, 1.0, "up", "developing_move"),
        CandidateIdea("DIR", "direct", ("direct",), 8.0, 1.2, 3.0, "up", "early_move"),
        CandidateIdea("SPL", "spillover", ("spillover",), 7.0, 0.8, 2.0, "up", "active_move"),
        CandidateIdea("MOV", "mover", ("mover",), 6.5, 2.5, 5.0, "up", ""),
        CandidateIdea("MAC", "macro", ("macro",), 5.0, 0.3, 1.0, "up", ""),
    ]
    finalists = select_candidate_finalists(candidates, max_symbols=5)
    assert finalists == ["POS", "THS", "DIR", "SPL", "MOV"]
