"""Tests for the thesis journal module."""

import sqlite3
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from ai_trader.journal import (
    ThesisEntry,
    ThesisJournal,
    ThesisUpdate,
    parse_thesis_updates,
)


def _make_journal(use_db: bool = False) -> ThesisJournal:
    """Create a journal, optionally with SQLite persistence."""
    if use_db:
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        return ThesisJournal(db_path=Path(tmp.name))
    return ThesisJournal()


def test_create_new_thesis():
    j = _make_journal()
    updates = [
        ThesisUpdate(
            id=None,
            underlying="AAPL",
            direction="bullish",
            thesis="Strong earnings expected",
            conviction=0.5,
            status="developing",
            new_observation="Beat revenue estimates last quarter",
        )
    ]
    j.apply_updates(updates)
    assert len(j.entries) == 1
    entry = list(j.entries.values())[0]
    assert entry.underlying == "AAPL"
    assert entry.direction == "bullish"
    assert entry.status == "developing"
    assert entry.conviction == 0.5
    assert len(entry.key_observations) == 1
    assert entry.cycles_observed == 1


def test_update_existing_thesis():
    j = _make_journal()
    # Create
    j.apply_updates([
        ThesisUpdate(
            id=None, underlying="TSLA", direction="bearish",
            thesis="Deliveries declining", conviction=0.4,
            status="developing", new_observation="Q4 miss",
        )
    ])
    thesis_id = list(j.entries.keys())[0]

    # Update
    j.apply_updates([
        ThesisUpdate(
            id=thesis_id, underlying="TSLA", direction="bearish",
            thesis="Deliveries declining", conviction=0.7,
            status="ready", new_observation="Q1 guidance weak too",
        )
    ])
    entry = j.entries[thesis_id]
    assert entry.conviction == 0.7
    assert entry.status == "ready"
    assert entry.cycles_observed == 2
    assert len(entry.key_observations) == 2


def test_invalidate_thesis_removes_from_entries():
    j = _make_journal()
    j.apply_updates([
        ThesisUpdate(
            id=None, underlying="MSFT", direction="bullish",
            thesis="Cloud growth", conviction=0.6,
            status="developing", new_observation="Azure expanding",
        )
    ])
    thesis_id = list(j.entries.keys())[0]

    j.apply_updates([
        ThesisUpdate(
            id=thesis_id, underlying="MSFT", direction="bullish",
            thesis="Cloud growth", conviction=0.2,
            status="invalidated", new_observation="Growth slowing",
        )
    ])
    # Invalidated entries get removed from in-memory dict
    assert thesis_id not in j.entries


def test_active_entries_filter():
    j = _make_journal()
    j.apply_updates([
        ThesisUpdate(
            id=None, underlying="AAPL", direction="bullish",
            thesis="iPhone cycle", conviction=0.5,
            status="developing", new_observation="",
        ),
        ThesisUpdate(
            id=None, underlying="GOOG", direction="bearish",
            thesis="Ad revenue drop", conviction=0.7,
            status="ready", new_observation="",
        ),
    ])
    active = j.active_entries()
    assert len(active) == 2
    underlyings = {e.underlying for e in active}
    assert "AAPL" in underlyings
    assert "GOOG" in underlyings


def test_to_context_str_no_entries():
    j = _make_journal()
    ctx = j.to_context_str()
    assert "No active theses" in ctx


def test_to_context_str_with_entries():
    j = _make_journal()
    j.apply_updates([
        ThesisUpdate(
            id=None, underlying="NVDA", direction="bullish",
            thesis="AI demand surge", conviction=0.8,
            status="ready", new_observation="Data center revenue up",
        )
    ])
    ctx = j.to_context_str()
    assert "NVDA" in ctx
    assert "BULLISH" in ctx
    assert "AI demand surge" in ctx


def test_sqlite_persistence():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = Path(tmp.name)
    tmp.close()

    # Create journal with persistence
    j1 = ThesisJournal(db_path=db_path)
    j1.apply_updates([
        ThesisUpdate(
            id=None, underlying="AMD", direction="bullish",
            thesis="Market share gains", conviction=0.6,
            status="developing", new_observation="New chip launch",
        )
    ])
    assert len(j1.entries) == 1

    # Load a new journal from the same DB
    j2 = ThesisJournal(db_path=db_path)
    assert len(j2.entries) == 1
    entry = list(j2.entries.values())[0]
    assert entry.underlying == "AMD"
    assert entry.conviction == 0.6


def test_sqlite_does_not_load_invalidated():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = Path(tmp.name)
    tmp.close()

    j1 = ThesisJournal(db_path=db_path)
    j1.apply_updates([
        ThesisUpdate(
            id=None, underlying="META", direction="bearish",
            thesis="Ad slowdown", conviction=0.5,
            status="developing", new_observation="",
        )
    ])
    tid = list(j1.entries.keys())[0]
    j1.apply_updates([
        ThesisUpdate(
            id=tid, underlying="META", direction="bearish",
            thesis="Ad slowdown", conviction=0.1,
            status="invalidated", new_observation="Actually ads rebounded",
        )
    ])

    j2 = ThesisJournal(db_path=db_path)
    assert len(j2.entries) == 0


def test_parse_thesis_updates():
    raw = [
        {
            "underlying": "aapl",
            "direction": "bullish",
            "thesis": "iPhone super cycle",
            "conviction": 0.7,
            "status": "developing",
            "new_observation": "Preorder numbers strong",
        },
        {
            "id": "thesis-1",
            "underlying": "GOOG",
            "direction": "bearish",
            "thesis": "Search decline",
            "conviction": 0.3,
            "status": "invalidated",
        },
    ]
    updates = parse_thesis_updates(raw)
    assert len(updates) == 2
    assert updates[0].id is None
    assert updates[0].underlying == "AAPL"  # uppercased
    assert updates[1].id == "thesis-1"
    assert updates[1].new_observation == ""  # missing key defaults to ""


def test_parse_thesis_updates_accepts_plural_observations():
    raw = [
        {
            "thesis_id": "thesis-4",
            "underlying": "INTC",
            "direction": "bullish",
            "thesis": "",
            "conviction": 0.55,
            "status": "developing",
            "observations": "INTC is extended; await pullback.",
        }
    ]
    updates = parse_thesis_updates(raw)
    assert len(updates) == 1
    assert updates[0].id == "thesis-4"
    assert updates[0].thesis == ""
    assert updates[0].new_observation == "INTC is extended; await pullback."


def test_new_thesis_uses_observation_when_thesis_blank():
    j = _make_journal()
    j.apply_updates([
        ThesisUpdate(
            id=None,
            underlying="INTC",
            direction="bullish",
            thesis="",
            conviction=0.5,
            status="developing",
            new_observation="Government stake narrative is strong but entry is extended.",
        )
    ])
    entry = list(j.entries.values())[0]
    assert entry.thesis == "Government stake narrative is strong but entry is extended."
    assert entry.key_observations == [
        "Government stake narrative is strong but entry is extended."
    ]


def test_multiple_theses_id_increment():
    j = _make_journal()
    for i in range(3):
        j.apply_updates([
            ThesisUpdate(
                id=None, underlying=f"SYM{i}", direction="bullish",
                thesis=f"Thesis {i}", conviction=0.5,
                status="developing", new_observation="",
            )
        ])
    ids = sorted(j.entries.keys())
    assert ids == ["thesis-1", "thesis-2", "thesis-3"]


def test_thesis_entry_context_str():
    entry = ThesisEntry(
        id="thesis-42",
        underlying="SPY",
        direction="bearish",
        thesis="Recession incoming",
        conviction=0.75,
        status="ready",
        key_observations=["Yield curve inverted", "PMI declining"],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        cycles_observed=3,
    )
    ctx = entry.to_context_str()
    assert "thesis-42" in ctx
    assert "SPY" in ctx
    assert "BEARISH" in ctx
    assert "Recession incoming" in ctx
    assert "cycles=3" in ctx


# ---------------------------------------------------------------------------
# New tests: journal count limits & prioritization
# ---------------------------------------------------------------------------

def _make_update(underlying="AAPL", direction="bullish", thesis="Test",
                 conviction=0.5, status="developing", observation="obs"):
    """Helper to create a ThesisUpdate with defaults."""
    return ThesisUpdate(
        id=None, underlying=underlying, direction=direction,
        thesis=thesis, conviction=conviction, status=status,
        new_observation=observation,
    )


def test_acted_on_removed_from_entries():
    """acted_on entries are removed from memory after apply_updates."""
    j = ThesisJournal()
    j.apply_updates([_make_update(status="developing")])
    tid = list(j.entries.keys())[0]
    # Mark acted_on
    j.apply_updates([
        ThesisUpdate(id=tid, underlying="AAPL", direction="bullish",
                     thesis="Test", conviction=0.8, status="acted_on",
                     new_observation="traded"),
    ])
    assert tid not in j.entries


def test_stale_developing_auto_pruned():
    """Low-conviction developing theses are pruned after stale_cycles."""
    j = ThesisJournal(stale_cycles=3, stale_conviction=0.4)
    j.apply_updates([_make_update(conviction=0.3)])
    tid = list(j.entries.keys())[0]
    # Simulate cycles by updating without raising conviction
    for _ in range(3):
        j.apply_updates([
            ThesisUpdate(id=tid, underlying="AAPL", direction="bullish",
                         thesis="Test", conviction=0.3, status="developing",
                         new_observation="no change"),
        ])
    # After 4 total cycles (1 create + 3 updates), cycles_observed=4 >= 3
    assert tid not in j.entries


def test_stale_pruning_spares_high_conviction():
    """Developing with conviction >= threshold survives stale pruning."""
    j = ThesisJournal(stale_cycles=3, stale_conviction=0.4)
    j.apply_updates([_make_update(conviction=0.5)])
    tid = list(j.entries.keys())[0]
    for _ in range(4):
        j.apply_updates([
            ThesisUpdate(id=tid, underlying="AAPL", direction="bullish",
                         thesis="Test", conviction=0.5, status="developing",
                         new_observation="still watching"),
        ])
    assert tid in j.entries


def test_stale_pruning_spares_ready():
    """ready theses are never auto-pruned regardless of cycles."""
    j = ThesisJournal(stale_cycles=3, stale_conviction=0.4)
    j.apply_updates([_make_update(conviction=0.3, status="ready")])
    tid = list(j.entries.keys())[0]
    for _ in range(5):
        j.apply_updates([
            ThesisUpdate(id=tid, underlying="AAPL", direction="bullish",
                         thesis="Test", conviction=0.3, status="ready",
                         new_observation="holding"),
        ])
    assert tid in j.entries


def test_overflow_prunes_lowest_conviction():
    """When over max_active, lowest-conviction developing theses go first."""
    j = ThesisJournal(max_active=3)
    # Create 5 developing theses with different convictions
    for i, conv in enumerate([0.3, 0.7, 0.5, 0.9, 0.2]):
        j.apply_updates([_make_update(
            underlying=f"SYM{i}", conviction=conv,
        )])
    # Should keep only top 3 by conviction: 0.9, 0.7, 0.5
    assert len(j.entries) == 3
    convictions = sorted(e.conviction for e in j.entries.values())
    assert convictions == [0.5, 0.7, 0.9]


def test_overflow_spares_ready_over_developing():
    """ready theses are protected; developing pruned first in overflow."""
    j = ThesisJournal(max_active=2)
    # Create a ready thesis with low conviction
    j.apply_updates([_make_update(underlying="READY", conviction=0.3, status="ready")])
    # Create two developing theses with higher conviction
    j.apply_updates([_make_update(underlying="DEV1", conviction=0.6)])
    j.apply_updates([_make_update(underlying="DEV2", conviction=0.8)])
    # 3 total, max_active=2 — should prune lowest-priority developing first
    assert len(j.entries) == 2
    underlyings = {e.underlying for e in j.entries.values()}
    assert "READY" in underlyings  # ready is protected


def test_tiered_display_shows_summary():
    """to_context_str shows full detail for top N, summary for rest."""
    j = ThesisJournal(max_full_display=2)
    for i, conv in enumerate([0.9, 0.7, 0.5, 0.3]):
        j.apply_updates([_make_update(underlying=f"SYM{i}", conviction=conv)])
    ctx = j.to_context_str()
    assert "showing top 2 in detail" in ctx
    assert "2 lower-priority theses (summary only)" in ctx
    # Top entries have full format (multi-line with "Thesis:" and "Observations:")
    assert "Thesis:" in ctx
    assert "Observations:" in ctx


def test_tiered_display_header():
    """Header shows count and limit info when max_active is set."""
    j = ThesisJournal(max_active=5, max_full_display=2)
    for i in range(3):
        j.apply_updates([_make_update(underlying=f"SYM{i}", conviction=0.5 + i * 0.1)])
    ctx = j.to_context_str()
    assert "[limit: 5]" in ctx
    assert "showing top 2 in detail" in ctx


def test_max_active_zero_unlimited():
    """max_active=0 does not prune (backward compat)."""
    j = ThesisJournal(max_active=0)
    for i in range(20):
        j.apply_updates([_make_update(underlying=f"SYM{i}", conviction=0.1)])
    assert len(j.entries) == 20


def test_summary_str_format():
    """to_summary_str is a single line with key info."""
    entry = ThesisEntry(
        id="thesis-1", underlying="AAPL", direction="bullish",
        thesis="Strong earnings", conviction=0.85, status="ready",
        key_observations=["beat estimates"],
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        cycles_observed=3,
    )
    s = entry.to_summary_str()
    assert "\n" not in s
    assert "thesis-1" in s
    assert "AAPL" in s
    assert "BULLISH" in s
    assert "conv=0.85" in s
    assert "status=ready" in s
    assert "cycles=3" in s
