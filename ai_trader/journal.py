"""Thesis journal — gives the LLM memory across trading cycles.

The LLM writes developing theses here. Each cycle it reads its prior
journal, updates entries with new observations, and only trades when
a thesis matures. Persisted to SQLite so it survives restarts.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .utils import log, now_eastern


@dataclass
class ThesisEntry:
    id: str
    underlying: str
    direction: str                  # bullish / bearish
    thesis: str                     # the core idea
    conviction: float               # 0.0 - 1.0, evolves over time
    status: str                     # developing / ready / acted_on / invalidated
    key_observations: list[str]     # evidence supporting or weakening the thesis
    created_at: datetime
    updated_at: datetime
    cycles_observed: int = 1        # how many cycles this thesis has been tracked

    def to_context_str(self) -> str:
        obs = "; ".join(self.key_observations[-5:]) if self.key_observations else "none yet"
        age = (now_eastern() - self.created_at).total_seconds() / 3600
        return (
            f"[{self.id}] {self.underlying} {self.direction.upper()} "
            f"(conviction={self.conviction:.2f}, status={self.status}, "
            f"cycles={self.cycles_observed}, age={age:.1f}h)\n"
            f"  Thesis: {self.thesis}\n"
            f"  Observations: {obs}"
        )

    def to_summary_str(self) -> str:
        return (
            f"[{self.id}] {self.underlying} {self.direction.upper()} "
            f"conv={self.conviction:.2f} status={self.status} "
            f"cycles={self.cycles_observed}"
        )


@dataclass
class ThesisUpdate:
    """Parsed from LLM output."""
    id: str | None                  # None = create new
    underlying: str
    direction: str
    thesis: str
    conviction: float
    status: str
    new_observation: str


class ThesisJournal:
    """In-memory thesis journal backed by SQLite."""

    def __init__(
        self,
        db_path: Path | None = None,
        max_active: int = 0,
        max_full_display: int = 0,
        stale_cycles: int = 8,
        stale_conviction: float = 0.4,
    ) -> None:
        self.entries: dict[str, ThesisEntry] = {}
        self.pruned_theses: list[str] = []  # descriptions of recently pruned theses
        self.db_path = db_path
        self._next_id = 1
        self._max_active = max_active
        self._max_full_display = max_full_display
        self._stale_cycles = stale_cycles
        self._stale_conviction = stale_conviction
        if db_path:
            self._ensure_schema()
            self._load_from_db()

    def _ensure_schema(self) -> None:
        if not self.db_path:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS thesis_journal (
                    id TEXT PRIMARY KEY,
                    underlying TEXT,
                    direction TEXT,
                    thesis TEXT,
                    conviction REAL,
                    status TEXT,
                    key_observations TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    cycles_observed INTEGER
                )
                """
            )

    def _load_from_db(self) -> None:
        if not self.db_path:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM thesis_journal WHERE status NOT IN ('invalidated', 'acted_on')"
            ).fetchall()
            for row in rows:
                row = dict(row)
                obs = json.loads(row["key_observations"]) if row["key_observations"] else []
                entry = ThesisEntry(
                    id=row["id"],
                    underlying=row["underlying"],
                    direction=row["direction"],
                    thesis=row["thesis"],
                    conviction=row["conviction"],
                    status=row["status"],
                    key_observations=obs,
                    created_at=datetime.fromisoformat(row["created_at"]),
                    updated_at=datetime.fromisoformat(row["updated_at"]),
                    cycles_observed=row["cycles_observed"],
                )
                self.entries[entry.id] = entry
                # Track highest ID
                try:
                    num = int(entry.id.split("-")[1])
                    if num >= self._next_id:
                        self._next_id = num + 1
                except (IndexError, ValueError):
                    pass

    def _save_entry(self, entry: ThesisEntry) -> None:
        if not self.db_path:
            return
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO thesis_journal
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.id,
                    entry.underlying,
                    entry.direction,
                    entry.thesis,
                    entry.conviction,
                    entry.status,
                    json.dumps(entry.key_observations),
                    entry.created_at.isoformat(),
                    entry.updated_at.isoformat(),
                    entry.cycles_observed,
                ),
            )

    def apply_updates(self, updates: list[ThesisUpdate]) -> None:
        """Apply LLM's journal updates."""
        self.pruned_theses = []  # reset for this cycle
        now = now_eastern()
        for u in updates:
            if u.id and u.id in self.entries:
                entry = self.entries[u.id]
                entry.conviction = u.conviction
                entry.status = u.status
                entry.thesis = u.thesis or entry.thesis
                entry.direction = u.direction or entry.direction
                if u.new_observation:
                    entry.key_observations.append(u.new_observation)
                entry.updated_at = now
                entry.cycles_observed += 1
                self._save_entry(entry)
                log(f"journal: updated [{entry.id}] {entry.underlying} -> {entry.status} ({entry.conviction:.2f})")
            else:
                new_id = f"thesis-{self._next_id}"
                self._next_id += 1
                obs = [u.new_observation] if u.new_observation else []
                entry = ThesisEntry(
                    id=new_id,
                    underlying=u.underlying,
                    direction=u.direction,
                    thesis=u.thesis,
                    conviction=u.conviction,
                    status=u.status,
                    key_observations=obs,
                    created_at=now,
                    updated_at=now,
                    cycles_observed=1,
                )
                self.entries[new_id] = entry
                self._save_entry(entry)
                log(f"journal: created [{new_id}] {entry.underlying} {entry.direction} ({entry.conviction:.2f})")

        # Clean up invalidated / acted_on entries from memory
        to_remove = [eid for eid, e in self.entries.items() if e.status in ("invalidated", "acted_on")]
        for eid in to_remove:
            del self.entries[eid]

        # Stale thesis auto-pruning: developing theses that lingered too long
        # with low conviction are auto-invalidated
        stale = [
            eid for eid, e in self.entries.items()
            if e.status == "developing"
            and e.cycles_observed >= self._stale_cycles
            and e.conviction < self._stale_conviction
        ]
        for eid in stale:
            entry = self.entries[eid]
            self.pruned_theses.append(
                f"Stale-pruned [{eid}] {entry.underlying} {entry.direction} "
                f"(conv={entry.conviction:.2f}, cycles={entry.cycles_observed}): {entry.thesis}"
            )
            entry.status = "invalidated"
            self._save_entry(entry)
            log(f"journal: auto-pruned stale [{eid}] {entry.underlying} "
                f"(cycles={entry.cycles_observed}, conv={entry.conviction:.2f})")
            del self.entries[eid]

        # Overflow pruning: if max_active is set and we're over the cap,
        # prune lowest-priority entries (developing before ready, lowest conviction first)
        if self._max_active > 0:
            active = [e for e in self.entries.values() if e.status in ("developing", "ready")]
            if len(active) > self._max_active:
                status_priority = {"developing": 0, "ready": 2}
                active.sort(key=lambda e: (status_priority.get(e.status, 0), e.conviction))
                to_prune = active[:len(active) - self._max_active]
                for entry in to_prune:
                    self.pruned_theses.append(
                        f"Overflow-pruned [{entry.id}] {entry.underlying} {entry.direction} "
                        f"(conv={entry.conviction:.2f}): {entry.thesis}"
                    )
                    entry.status = "invalidated"
                    self._save_entry(entry)
                    log(f"journal: overflow-pruned [{entry.id}] {entry.underlying} "
                        f"(conv={entry.conviction:.2f}, status was {entry.status})")
                    del self.entries[entry.id]

    def active_entries(self) -> list[ThesisEntry]:
        return [
            e for e in self.entries.values()
            if e.status in ("developing", "ready", "acted_on")
        ]

    def to_context_str(self) -> str:
        active = self.active_entries()
        if not active and not self.pruned_theses:
            return "No active theses. You can start developing new ones based on the news."

        parts: list[str] = []

        if active:
            active.sort(key=lambda e: e.conviction, reverse=True)
            cap_note = f" [limit: {self._max_active}]" if self._max_active > 0 else ""

            # Tiered display: full detail for top N, summary for the rest
            if self._max_full_display > 0 and len(active) > self._max_full_display:
                top = active[:self._max_full_display]
                rest = active[self._max_full_display:]
                header = f"Active theses ({len(active)}, showing top {len(top)} in detail):{cap_note}"
                lines = [header]
                for entry in top:
                    lines.append(entry.to_context_str())
                lines.append(f"--- {len(rest)} lower-priority theses (summary only) ---")
                for entry in rest:
                    lines.append(entry.to_summary_str())
                parts.append("\n\n".join(lines))
            else:
                lines = [f"Active theses ({len(active)}):{cap_note}"]
                for entry in active:
                    lines.append(entry.to_context_str())
                parts.append("\n\n".join(lines))
        else:
            parts.append("No active theses. You can start developing new ones based on the news.")

        if self.pruned_theses:
            parts.append(
                "\nRecently pruned theses (FYI — these were auto-removed):\n"
                + "\n".join(f"  - {p}" for p in self.pruned_theses)
            )

        return "\n".join(parts)

    def set_time(self, dt: datetime) -> None:
        """Override current time (used by backtester)."""
        self._override_time = dt

    def _now(self) -> datetime:
        return getattr(self, "_override_time", None) or now_eastern()


def parse_thesis_updates(raw: list[dict]) -> list[ThesisUpdate]:
    """Parse raw LLM output into ThesisUpdate objects."""
    def safe_float(value: object, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    updates = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        raw_action = str(item.get("action") or "").strip().lower()
        raw_id = item.get("id")
        if raw_id is None and raw_action not in {"create", "new"}:
            raw_thesis_id = item.get("thesis_id")
            if str(raw_thesis_id or "").startswith("thesis-"):
                raw_id = raw_thesis_id
        underlying = item.get("underlying") or item.get("ticker") or item.get("symbol")
        if not underlying:
            tickers = item.get("tickers")
            if isinstance(tickers, list) and tickers:
                underlying = tickers[0]
        underlying = underlying or ""
        thesis = (
            item.get("thesis")
            or item.get("reasoning")
            or item.get("reason")
            or item.get("summary")
            or item.get("new_observation")
            or ""
        )
        new_observation = (
            item.get("new_observation")
            or item.get("observation")
            or item.get("reasoning")
            or item.get("reason")
            or item.get("summary")
            or ""
        )
        raw_direction = item.get("direction")
        direction = str(raw_direction or "").strip().lower()
        if direction in {"call", "long", "bull"}:
            direction = "bullish"
        elif direction in {"put", "short", "bear"}:
            direction = "bearish"
        elif direction not in {"bullish", "bearish", "neutral"}:
            direction = "neutral" if raw_id is None else ""
        status = str(item.get("status") or "").strip().lower()
        if raw_action in {"invalidate", "invalidated"}:
            status = "invalidated"
        elif not status:
            status = "developing"
        updates.append(
            ThesisUpdate(
                id=str(raw_id) if raw_id is not None else None,
                underlying=str(underlying).upper(),
                direction=direction,
                thesis=str(thesis),
                conviction=safe_float(item.get("conviction"), 0.5),
                status=status,
                new_observation=str(new_observation),
            )
        )
    return updates
