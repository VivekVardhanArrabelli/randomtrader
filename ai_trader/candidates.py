"""Candidate ranking and context formatting for broad market triage."""

from __future__ import annotations

from dataclasses import dataclass

_BUCKET_PRIORITY = ("position", "thesis", "direct", "spillover", "mover", "macro")
_BUCKET_LABELS = {
    "position": "position",
    "thesis": "thesis",
    "direct": "direct",
    "spillover": "spillover",
    "mover": "mover",
    "macro": "macro",
}
_BUCKET_BASE_SCORES = {
    "position": 4.8,
    "thesis": 4.2,
    "direct": 3.8,
    "spillover": 3.0,
    "mover": 2.4,
    "macro": 1.5,
}
_REACTION_MULTIPLIERS = {
    "early_move": 1.15,
    "developing_move": 1.08,
    "active_move": 1.0,
    "not_moving_yet": 0.92,
    "extended_move": 0.72,
    "": 1.0,
}
_DEFAULT_BUCKET_LIMITS = {
    "position": 2,
    "thesis": 2,
    "direct": 3,
    "spillover": 2,
    "mover": 2,
    "macro": 1,
}


@dataclass(frozen=True)
class CandidateIdea:
    symbol: str
    bucket: str
    tags: tuple[str, ...]
    score: float
    intraday_chg: float
    five_d_chg: float
    trend: str
    reaction: str
    event_type: str = ""
    freshness: str = ""
    source_count: int = 0
    age_minutes: int = 0

    def to_context_str(self) -> str:
        bucket = _BUCKET_LABELS.get(self.bucket, self.bucket)
        tags = ",".join(self.tags)
        if self.event_type:
            event = (
                f"{self.event_type}/{self.freshness}/{self.source_count}src/"
                f"{self.age_minutes}m"
            )
        else:
            event = "none"
        return (
            f"{self.symbol} | {bucket} [{tags}] | score={self.score:.2f} | "
            f"event={event} | today({self.intraday_chg:+.1f}%) "
            f"5d({self.five_d_chg:+.1f}%) | reaction={self.reaction or 'none'} "
            f"trend={self.trend or 'unknown'}"
        )


def primary_bucket(tags: set[str]) -> str:
    for bucket in _BUCKET_PRIORITY:
        if bucket in tags:
            return bucket
    return "mover"


def score_candidate(
    bucket: str,
    event_strength: float,
    intraday_chg: float,
    five_d_chg: float,
    reaction: str,
    tag_count: int = 1,
) -> float:
    reaction_multiplier = _REACTION_MULTIPLIERS.get(reaction, 1.0)
    move_component = min(abs(intraday_chg), 6.0) * 0.35 + min(abs(five_d_chg), 12.0) * 0.12
    source_bonus = 0.25 * max(tag_count - 1, 0)
    return (
        _BUCKET_BASE_SCORES.get(bucket, 2.0)
        + event_strength * reaction_multiplier
        + move_component
        + source_bonus
    )


def build_candidate_ideas(
    source_tags_by_symbol: dict[str, set[str]],
    metrics_by_symbol: dict[str, dict],
    event_by_symbol: dict[str, object],
    event_score_by_symbol: dict[str, float],
) -> list[CandidateIdea]:
    ideas: list[CandidateIdea] = []
    for symbol, tags in source_tags_by_symbol.items():
        metrics = metrics_by_symbol.get(symbol)
        if not metrics:
            continue
        ordered_tags = tuple(
            sorted(
                tags,
                key=lambda tag: (
                    _BUCKET_PRIORITY.index(tag)
                    if tag in _BUCKET_PRIORITY
                    else len(_BUCKET_PRIORITY)
                ),
            )
        )
        bucket = primary_bucket(tags)
        event = event_by_symbol.get(symbol)
        score = score_candidate(
            bucket,
            event_score_by_symbol.get(symbol, 0.0),
            float(metrics.get("intraday_chg", 0.0)),
            float(metrics.get("five_d_chg", 0.0)),
            str(metrics.get("reaction", "")),
            tag_count=len(tags),
        )
        ideas.append(
            CandidateIdea(
                symbol=symbol,
                bucket=bucket,
                tags=ordered_tags,
                score=score,
                intraday_chg=float(metrics.get("intraday_chg", 0.0)),
                five_d_chg=float(metrics.get("five_d_chg", 0.0)),
                trend=str(metrics.get("trend", "")),
                reaction=str(metrics.get("reaction", "")),
                event_type=str(getattr(event, "event_type", "") or ""),
                freshness=str(getattr(event, "freshness", "") or ""),
                source_count=int(getattr(event, "source_count", 0) or 0),
                age_minutes=int(getattr(event, "age_minutes", 0) or 0),
            )
        )
    ideas.sort(key=lambda idea: (idea.score, idea.source_count, -idea.age_minutes), reverse=True)
    return ideas


def select_candidate_finalists(
    candidates: list[CandidateIdea],
    max_symbols: int = 6,
    bucket_limits: dict[str, int] | None = None,
) -> list[str]:
    limits = bucket_limits or _DEFAULT_BUCKET_LIMITS
    selected: list[str] = []
    seen: set[str] = set()
    for bucket in _BUCKET_PRIORITY:
        bucket_rows = [candidate for candidate in candidates if candidate.bucket == bucket]
        for candidate in bucket_rows[: limits.get(bucket, 0)]:
            if candidate.symbol in seen:
                continue
            selected.append(candidate.symbol)
            seen.add(candidate.symbol)
            if len(selected) >= max_symbols:
                return selected
    for candidate in candidates:
        if candidate.symbol in seen:
            continue
        selected.append(candidate.symbol)
        seen.add(candidate.symbol)
        if len(selected) >= max_symbols:
            break
    return selected


def format_candidate_table(
    candidates: list[CandidateIdea],
    max_rows: int = 12,
) -> str:
    if not candidates:
        return ""
    lines = ["Candidate Table (broad triage before deep dive):"]
    for candidate in candidates[:max_rows]:
        lines.append(f"  {candidate.to_context_str()}")
    return "\n".join(lines)
