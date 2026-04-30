"""Rolling-window experiment harness for AI trader backtests.

Run: python -m ai_trader.experiments --label current-idea
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from calendar import monthrange
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path

from dotenv import load_dotenv

from . import config
from .backtest import (
    BacktestConfig,
    BacktestResult,
    PrepareBacktestResult,
    prepare_backtest_data,
    run_backtest,
    save_backtest_result,
    save_debug_log,
)
from .utils import log, now_eastern


@dataclass(frozen=True)
class ExperimentBaseline:
    start_date: date
    end_date: date
    total_return_pct: float
    net_pnl: float
    total_trades: int
    win_rate: float | None
    sharpe_ratio: float | None
    max_drawdown: float
    llm_failure_days: int


@dataclass(frozen=True)
class ExperimentWindow:
    label: str
    start_date: date
    end_date: date
    kind: str = "rolling"


@dataclass
class ExperimentWindowResult:
    window: ExperimentWindow
    result: BacktestResult
    result_path: Path | None = None
    debug_log_path: Path | None = None


@dataclass
class BaselineComparison:
    label: str
    start_date: date
    end_date: date
    total_return_pct: float
    baseline_return_pct: float
    return_delta_pct: float
    sharpe_ratio: float | None
    baseline_sharpe_ratio: float | None
    sharpe_delta: float | None
    max_drawdown: float
    baseline_max_drawdown: float
    max_drawdown_delta: float
    llm_failure_days: int
    baseline_llm_failure_days: int
    llm_failure_day_delta: int


@dataclass
class ExperimentSummary:
    window_count: int
    positive_windows: int
    positive_window_ratio: float
    avg_return_pct: float | None
    median_return_pct: float | None
    avg_sharpe_ratio: float | None
    median_sharpe_ratio: float | None
    avg_profit_factor: float | None
    median_profit_factor: float | None
    worst_return_pct: float | None
    best_return_pct: float | None
    worst_max_drawdown: float | None
    avg_trade_count: float | None
    total_llm_failure_days: int
    total_llm_error_cycles: int
    reference_comparison: BaselineComparison | None
    promotion_status: str
    promotion_reasons: list[str]


README_BASELINE = ExperimentBaseline(
    start_date=date(2025, 11, 13),
    end_date=date(2026, 2, 13),
    total_return_pct=0.1226,
    net_pnl=12_255.80,
    total_trades=11,
    win_rate=0.455,
    sharpe_ratio=0.87,
    max_drawdown=13_930.0,
    llm_failure_days=0,
)

DEFAULT_WINDOW_MONTHS = 3
DEFAULT_STEP_MONTHS = 1
DEFAULT_WINDOW_COUNT = 6
DEFAULT_ANCHOR_END = README_BASELINE.end_date
EXPERIMENT_LOG_ROOT = Path(__file__).resolve().parent / "logs" / "experiments"
MATERIAL_DRAWDOWN_WORSENING_FRACTION = 0.15


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower()).strip("-")
    return slug or "experiment"


def _shift_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, monthrange(year, month)[1])
    return date(year, month, day)


def _window_key(start_date: date, end_date: date) -> tuple[str, str]:
    return start_date.isoformat(), end_date.isoformat()


def build_rolling_windows(
    anchor_end: date = DEFAULT_ANCHOR_END,
    *,
    count: int = DEFAULT_WINDOW_COUNT,
    window_months: int = DEFAULT_WINDOW_MONTHS,
    step_months: int = DEFAULT_STEP_MONTHS,
) -> list[ExperimentWindow]:
    if count <= 0:
        raise ValueError("count must be positive")
    if window_months <= 0:
        raise ValueError("window_months must be positive")
    if step_months <= 0:
        raise ValueError("step_months must be positive")

    windows: list[ExperimentWindow] = []
    for index in range(count):
        offset = step_months * (count - index - 1)
        end_date = _shift_months(anchor_end, -offset)
        start_date = _shift_months(end_date, -window_months)
        windows.append(
            ExperimentWindow(
                label=f"rolling_{index + 1:02d}",
                start_date=start_date,
                end_date=end_date,
                kind="rolling",
            )
        )
    return windows


def _default_output_dir(label: str, now: datetime | None = None) -> Path:
    current = now or now_eastern()
    timestamp = current.strftime("%Y%m%d-%H%M%S")
    return EXPERIMENT_LOG_ROOT / f"{timestamp}-{_slugify(label)}"


def _git_commit(repo_root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip() or None


def _fmt_pct(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.2%}"


def _fmt_float(value: float | None, decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{decimals}f}"


def _fmt_money(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"${value:,.2f}"


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def summarize_window_result(window_result: ExperimentWindowResult) -> dict:
    result = window_result.result
    return {
        "label": window_result.window.label,
        "kind": window_result.window.kind,
        "start_date": window_result.window.start_date.isoformat(),
        "end_date": window_result.window.end_date.isoformat(),
        "historical_options_provider": result.historical_options_provider or "unknown",
        "initial_equity": result.initial_equity,
        "final_equity": result.final_equity,
        "total_return_pct": result.total_return_pct,
        "net_pnl": result.net_pnl,
        "total_trades": result.total_trades,
        "wins": result.wins,
        "losses": result.losses,
        "win_rate": result.win_rate,
        "max_drawdown": result.max_drawdown,
        "sharpe_ratio": result.sharpe_ratio,
        "profit_factor": result.profit_factor,
        "avg_conviction": result.avg_conviction,
        "days_tested": result.days_tested,
        "llm_error_cycles": result.llm_error_cycles,
        "llm_failure_days": result.llm_failure_days,
        "log_db_path": result.log_db_path,
        "result_path": str(window_result.result_path) if window_result.result_path else None,
        "debug_log_path": str(window_result.debug_log_path) if window_result.debug_log_path else None,
    }


def compare_to_readme_baseline(
    window_result: ExperimentWindowResult,
    baseline: ExperimentBaseline = README_BASELINE,
) -> BaselineComparison:
    result = window_result.result
    sharpe_delta: float | None = None
    if result.sharpe_ratio is not None and baseline.sharpe_ratio is not None:
        sharpe_delta = result.sharpe_ratio - baseline.sharpe_ratio
    return BaselineComparison(
        label=window_result.window.label,
        start_date=window_result.window.start_date,
        end_date=window_result.window.end_date,
        total_return_pct=result.total_return_pct,
        baseline_return_pct=baseline.total_return_pct,
        return_delta_pct=result.total_return_pct - baseline.total_return_pct,
        sharpe_ratio=result.sharpe_ratio,
        baseline_sharpe_ratio=baseline.sharpe_ratio,
        sharpe_delta=sharpe_delta,
        max_drawdown=result.max_drawdown,
        baseline_max_drawdown=baseline.max_drawdown,
        max_drawdown_delta=result.max_drawdown - baseline.max_drawdown,
        llm_failure_days=result.llm_failure_days,
        baseline_llm_failure_days=baseline.llm_failure_days,
        llm_failure_day_delta=result.llm_failure_days - baseline.llm_failure_days,
    )


def _recommend_promotion(
    *,
    positive_window_ratio: float,
    median_return_pct: float | None,
    avg_return_pct: float | None,
    worst_max_drawdown: float | None,
    total_llm_failure_days: int,
    total_llm_error_cycles: int,
    reference_comparison: BaselineComparison | None,
    baseline: ExperimentBaseline,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if reference_comparison is None:
        reasons.append("reference baseline window was not part of the evaluated suite")
        return "investigate", reasons

    drawdown_tolerance = max(
        1_000.0,
        baseline.max_drawdown * MATERIAL_DRAWDOWN_WORSENING_FRACTION,
    )
    anchor_drawdown_ok = reference_comparison.max_drawdown_delta <= drawdown_tolerance
    suite_drawdown_ok = (
        worst_max_drawdown is None
        or worst_max_drawdown <= baseline.max_drawdown + drawdown_tolerance
    )

    if reference_comparison.return_delta_pct >= 0:
        reasons.append(
            f"anchor window beat baseline return by {_fmt_pct(reference_comparison.return_delta_pct)}"
        )
    else:
        reasons.append(
            "anchor window trailed baseline return by "
            f"{abs(reference_comparison.return_delta_pct):.2%}"
        )

    if median_return_pct is not None and median_return_pct > 0:
        reasons.append(
            f"rolling-window median return stayed positive at {_fmt_pct(median_return_pct)}"
        )
    else:
        reasons.append("rolling-window median return was not positive")

    if avg_return_pct is not None and avg_return_pct > 0:
        reasons.append(
            f"rolling-window average return stayed positive at {_fmt_pct(avg_return_pct)}"
        )
    else:
        reasons.append("rolling-window average return was not positive")

    reasons.append(f"{positive_window_ratio:.0%} of rolling windows finished positive")

    if suite_drawdown_ok:
        reasons.append(
            "suite drawdown stayed within the material-worsening tolerance "
            f"({_fmt_money(baseline.max_drawdown + drawdown_tolerance)} ceiling)"
        )
    else:
        reasons.append(
            "suite drawdown exceeded the material-worsening tolerance "
            f"({_fmt_money(worst_max_drawdown)} vs {_fmt_money(baseline.max_drawdown)})"
        )

    if anchor_drawdown_ok:
        reasons.append("anchor-window drawdown stayed within the material-worsening tolerance")
    else:
        reasons.append("anchor-window drawdown worsened materially versus the README baseline")

    if total_llm_failure_days == 0 and total_llm_error_cycles == 0:
        reasons.append("LLM reliability held at zero failure days and zero error cycles across the suite")
    else:
        reasons.append(
            "LLM reliability regressed with "
            f"{total_llm_failure_days} failure day(s) and {total_llm_error_cycles} error cycle(s)"
        )

    robust_positive_suite = (
        positive_window_ratio >= 0.60
        and (median_return_pct or 0.0) > 0
        and (avg_return_pct or 0.0) > 0
    )
    reliability_clean = total_llm_failure_days == 0 and total_llm_error_cycles == 0
    keep = (
        robust_positive_suite
        and suite_drawdown_ok
        and anchor_drawdown_ok
        and reliability_clean
    )
    discard = (
        positive_window_ratio < 0.50
        and (median_return_pct is None or median_return_pct <= 0)
        and (avg_return_pct is None or avg_return_pct <= 0)
        and not suite_drawdown_ok
    ) or (
        not reliability_clean
        and (
            positive_window_ratio < 0.50
            or (median_return_pct is None or median_return_pct <= 0)
        )
    )
    if keep:
        return "keep", reasons
    if discard:
        return "discard", reasons
    if reference_comparison.return_delta_pct < 0 and positive_window_ratio >= 0.60:
        reasons.append("anchor window lagged, but that is treated as a review signal rather than a hard promotion veto")
    return "investigate", reasons


def _suite_failure_day_count(window_results: list[ExperimentWindowResult]) -> int:
    failure_dates: set[str] = set()
    approximate_count = 0
    for window_result in window_results:
        explicit_dates = {
            str(entry.get("date"))
            for entry in window_result.result.decision_log
            if entry.get("llm_error") and entry.get("date")
        }
        if explicit_dates:
            failure_dates.update(explicit_dates)
            continue
        approximate_count += window_result.result.llm_failure_days
    return len(failure_dates) + approximate_count


def _suite_llm_error_cycle_count(window_results: list[ExperimentWindowResult]) -> int:
    failure_cycles: set[str] = set()
    approximate_count = 0
    for window_result in window_results:
        explicit_cycles = {
            str(entry.get("decision_time"))
            for entry in window_result.result.decision_log
            if entry.get("llm_error") and entry.get("decision_time")
        }
        if explicit_cycles:
            failure_cycles.update(explicit_cycles)
            continue
        approximate_count += window_result.result.llm_error_cycles
    return len(failure_cycles) + approximate_count


def summarize_experiment_suite(
    window_results: list[ExperimentWindowResult],
    *,
    baseline: ExperimentBaseline = README_BASELINE,
) -> ExperimentSummary:
    rolling_results = [result for result in window_results if result.window.kind == "rolling"]
    metrics_source = rolling_results or window_results

    returns = [result.result.total_return_pct for result in metrics_source]
    sharpes = [result.result.sharpe_ratio for result in metrics_source if result.result.sharpe_ratio is not None]
    profit_factors = [
        result.result.profit_factor
        for result in metrics_source
        if result.result.profit_factor is not None
    ]
    max_drawdowns = [result.result.max_drawdown for result in metrics_source]
    trade_counts = [result.result.total_trades for result in metrics_source]
    total_llm_failure_days = _suite_failure_day_count(metrics_source)
    total_llm_error_cycles = _suite_llm_error_cycle_count(metrics_source)
    positive_windows = sum(1 for value in returns if value > 0)
    positive_window_ratio = positive_windows / len(metrics_source) if metrics_source else 0.0

    reference_comparison = None
    baseline_key = _window_key(baseline.start_date, baseline.end_date)
    for window_result in window_results:
        if _window_key(window_result.window.start_date, window_result.window.end_date) == baseline_key:
            reference_comparison = compare_to_readme_baseline(window_result, baseline=baseline)
            break

    promotion_status, promotion_reasons = _recommend_promotion(
        positive_window_ratio=positive_window_ratio,
        median_return_pct=_median(returns),
        avg_return_pct=_mean(returns),
        worst_max_drawdown=max(max_drawdowns) if max_drawdowns else None,
        total_llm_failure_days=total_llm_failure_days,
        total_llm_error_cycles=total_llm_error_cycles,
        reference_comparison=reference_comparison,
        baseline=baseline,
    )

    return ExperimentSummary(
        window_count=len(metrics_source),
        positive_windows=positive_windows,
        positive_window_ratio=positive_window_ratio,
        avg_return_pct=_mean(returns),
        median_return_pct=_median(returns),
        avg_sharpe_ratio=_mean(sharpes),
        median_sharpe_ratio=_median(sharpes),
        avg_profit_factor=_mean(profit_factors),
        median_profit_factor=_median(profit_factors),
        worst_return_pct=min(returns) if returns else None,
        best_return_pct=max(returns) if returns else None,
        worst_max_drawdown=max(max_drawdowns) if max_drawdowns else None,
        avg_trade_count=_mean(trade_counts),
        total_llm_failure_days=total_llm_failure_days,
        total_llm_error_cycles=total_llm_error_cycles,
        reference_comparison=reference_comparison,
        promotion_status=promotion_status,
        promotion_reasons=promotion_reasons,
    )


def _prepare_result_to_dict(result: PrepareBacktestResult | None) -> dict | None:
    if result is None:
        return None
    return {
        "start_date": result.start_date.isoformat(),
        "end_date": result.end_date.isoformat(),
        "days_prepared": result.days_prepared,
        "decision_points": result.decision_points,
        "cache_db_path": str(result.cache_db_path),
        "cache_entries": result.cache_entries,
    }


def _baseline_to_dict(baseline: ExperimentBaseline) -> dict:
    return {
        "start_date": baseline.start_date.isoformat(),
        "end_date": baseline.end_date.isoformat(),
        "total_return_pct": baseline.total_return_pct,
        "net_pnl": baseline.net_pnl,
        "total_trades": baseline.total_trades,
        "win_rate": baseline.win_rate,
        "sharpe_ratio": baseline.sharpe_ratio,
        "max_drawdown": baseline.max_drawdown,
        "llm_failure_days": baseline.llm_failure_days,
    }


def _comparison_to_dict(comparison: BaselineComparison | None) -> dict | None:
    if comparison is None:
        return None
    return {
        "label": comparison.label,
        "start_date": comparison.start_date.isoformat(),
        "end_date": comparison.end_date.isoformat(),
        "total_return_pct": comparison.total_return_pct,
        "baseline_return_pct": comparison.baseline_return_pct,
        "return_delta_pct": comparison.return_delta_pct,
        "sharpe_ratio": comparison.sharpe_ratio,
        "baseline_sharpe_ratio": comparison.baseline_sharpe_ratio,
        "sharpe_delta": comparison.sharpe_delta,
        "max_drawdown": comparison.max_drawdown,
        "baseline_max_drawdown": comparison.baseline_max_drawdown,
        "max_drawdown_delta": comparison.max_drawdown_delta,
        "llm_failure_days": comparison.llm_failure_days,
        "baseline_llm_failure_days": comparison.baseline_llm_failure_days,
        "llm_failure_day_delta": comparison.llm_failure_day_delta,
    }


def _summary_to_dict(summary: ExperimentSummary) -> dict:
    return {
        "window_count": summary.window_count,
        "positive_windows": summary.positive_windows,
        "positive_window_ratio": summary.positive_window_ratio,
        "avg_return_pct": summary.avg_return_pct,
        "median_return_pct": summary.median_return_pct,
        "avg_sharpe_ratio": summary.avg_sharpe_ratio,
        "median_sharpe_ratio": summary.median_sharpe_ratio,
        "avg_profit_factor": summary.avg_profit_factor,
        "median_profit_factor": summary.median_profit_factor,
        "worst_return_pct": summary.worst_return_pct,
        "best_return_pct": summary.best_return_pct,
        "worst_max_drawdown": summary.worst_max_drawdown,
        "avg_trade_count": summary.avg_trade_count,
        "total_llm_failure_days": summary.total_llm_failure_days,
        "total_llm_error_cycles": summary.total_llm_error_cycles,
        "reference_comparison": _comparison_to_dict(summary.reference_comparison),
        "promotion_status": summary.promotion_status,
        "promotion_reasons": summary.promotion_reasons,
    }


def _resolve_backtest_result_path(path: Path) -> Path:
    if path.is_file():
        return path
    summary_path = path / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        windows = summary.get("windows") or []
        if len(windows) != 1:
            raise ValueError(
                f"expected exactly one window in {summary_path}, found {len(windows)}"
            )
        result_path = windows[0].get("result_path")
        if not result_path:
            raise ValueError(f"window result path missing in {summary_path}")
        return Path(result_path)
    window_files = sorted(path.glob("rolling_*.json"))
    if len(window_files) == 1:
        return window_files[0]
    raise ValueError(f"could not resolve a single backtest result from {path}")


def _load_backtest_artifact(path: Path) -> dict:
    result_path = _resolve_backtest_result_path(path)
    payload = json.loads(result_path.read_text())
    payload["_result_path"] = str(result_path)
    payload["_source_path"] = str(path)
    return payload


def _jaccard_similarity(left: set[str], right: set[str]) -> float | None:
    if not left and not right:
        return None
    union = left | right
    if not union:
        return None
    return len(left & right) / len(union)


def _decision_symbol_map(entries: list[dict], field: str) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for entry in entries:
        decision_time = str(entry.get("decision_time") or "")
        if not decision_time:
            continue
        symbols = {
            str(symbol)
            for symbol in entry.get(field, [])
            if isinstance(symbol, str) and symbol
        }
        mapping[decision_time] = symbols
    return mapping


def _proposal_symbol_map(entries: list[dict]) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = {}
    for entry in entries:
        decision_time = str(entry.get("decision_time") or "")
        if not decision_time:
            continue
        symbols = {
            str(trade.get("underlying"))
            for trade in entry.get("trades_proposed", [])
            if isinstance(trade, dict) and trade.get("underlying")
        }
        mapping[decision_time] = symbols
    return mapping


def _trade_signature(trade: dict) -> tuple[str, str, str, str]:
    return (
        str(trade.get("entry_date") or trade.get("timestamp") or ""),
        str(trade.get("underlying") or ""),
        str(trade.get("option_type") or trade.get("action") or ""),
        str(trade.get("expression_profile") or ""),
    )


def compare_experiment_runs(left_path: Path, right_path: Path) -> dict:
    left = _load_backtest_artifact(left_path)
    right = _load_backtest_artifact(right_path)

    left_provider = str(left.get("historical_options_provider") or "unknown")
    right_provider = str(right.get("historical_options_provider") or "unknown")

    left_decisions = list(left.get("decision_log") or [])
    right_decisions = list(right.get("decision_log") or [])
    left_finalists = _decision_symbol_map(left_decisions, "finalists")
    right_finalists = _decision_symbol_map(right_decisions, "finalists")
    left_proposals = _proposal_symbol_map(left_decisions)
    right_proposals = _proposal_symbol_map(right_decisions)

    shared_decision_times = sorted(set(left_finalists) & set(right_finalists))
    finalist_overlaps = [
        value
        for decision_time in shared_decision_times
        if (value := _jaccard_similarity(left_finalists[decision_time], right_finalists[decision_time])) is not None
    ]
    shared_proposal_times = sorted(set(left_proposals) & set(right_proposals))
    proposal_overlaps = [
        value
        for decision_time in shared_proposal_times
        if (value := _jaccard_similarity(left_proposals[decision_time], right_proposals[decision_time])) is not None
    ]

    left_trades = list(left.get("trades") or [])
    right_trades = list(right.get("trades") or [])
    left_trade_map = {_trade_signature(trade): trade for trade in left_trades}
    right_trade_map = {_trade_signature(trade): trade for trade in right_trades}
    shared_trade_signatures = sorted(set(left_trade_map) & set(right_trade_map))
    entry_price_deltas = [
        abs(float(left_trade_map[key].get("entry_premium") or 0.0) - float(right_trade_map[key].get("entry_premium") or 0.0))
        for key in shared_trade_signatures
    ]
    exit_price_deltas = [
        abs(float(left_trade_map[key].get("exit_premium") or 0.0) - float(right_trade_map[key].get("exit_premium") or 0.0))
        for key in shared_trade_signatures
    ]

    left_underlyings = {str(trade.get("underlying")) for trade in left_trades if trade.get("underlying")}
    right_underlyings = {str(trade.get("underlying")) for trade in right_trades if trade.get("underlying")}

    return {
        "generated_at": now_eastern().isoformat(),
        "left": {
            "source_path": str(left_path),
            "result_path": left["_result_path"],
            "historical_options_provider": left_provider,
            "total_return_pct": left.get("total_return_pct"),
            "net_pnl": left.get("net_pnl"),
            "total_trades": left.get("total_trades"),
            "max_drawdown": left.get("max_drawdown"),
            "sharpe_ratio": left.get("sharpe_ratio"),
        },
        "right": {
            "source_path": str(right_path),
            "result_path": right["_result_path"],
            "historical_options_provider": right_provider,
            "total_return_pct": right.get("total_return_pct"),
            "net_pnl": right.get("net_pnl"),
            "total_trades": right.get("total_trades"),
            "max_drawdown": right.get("max_drawdown"),
            "sharpe_ratio": right.get("sharpe_ratio"),
        },
        "metrics": {
            "return_delta_pct": (left.get("total_return_pct") or 0.0) - (right.get("total_return_pct") or 0.0),
            "net_pnl_delta": (left.get("net_pnl") or 0.0) - (right.get("net_pnl") or 0.0),
            "trade_count_delta": int(left.get("total_trades") or 0) - int(right.get("total_trades") or 0),
            "max_drawdown_delta": (left.get("max_drawdown") or 0.0) - (right.get("max_drawdown") or 0.0),
            "sharpe_delta": (
                (left.get("sharpe_ratio") or 0.0) - (right.get("sharpe_ratio") or 0.0)
                if left.get("sharpe_ratio") is not None and right.get("sharpe_ratio") is not None
                else None
            ),
        },
        "menus": {
            "decision_times_compared": len(shared_decision_times),
            "avg_finalist_overlap": _mean(finalist_overlaps),
            "decision_times_with_identical_finalists": sum(
                1 for decision_time in shared_decision_times
                if left_finalists[decision_time] == right_finalists[decision_time]
            ),
            "avg_proposed_symbol_overlap": _mean(proposal_overlaps),
            "decision_times_with_identical_proposals": sum(
                1 for decision_time in shared_proposal_times
                if left_proposals[decision_time] == right_proposals[decision_time]
            ),
        },
        "fills": {
            "shared_trade_signatures": len(shared_trade_signatures),
            "left_only_trade_signatures": len(set(left_trade_map) - set(right_trade_map)),
            "right_only_trade_signatures": len(set(right_trade_map) - set(left_trade_map)),
            "avg_entry_premium_delta": _mean(entry_price_deltas),
            "avg_exit_premium_delta": _mean(exit_price_deltas),
            "underlying_overlap_ratio": _jaccard_similarity(left_underlyings, right_underlyings),
        },
    }


def _comparison_markdown(comparison: dict) -> str:
    left = comparison["left"]
    right = comparison["right"]
    metrics = comparison["metrics"]
    menus = comparison["menus"]
    fills = comparison["fills"]
    lines = [
        "# Provider Comparison",
        "",
        f"- Generated: {comparison['generated_at']}",
        f"- Left: `{left['historical_options_provider']}` | `{left['result_path']}`",
        f"- Right: `{right['historical_options_provider']}` | `{right['result_path']}`",
        "",
        "## Headline Metrics",
        "",
        f"- Return delta (left-right): {_fmt_pct(metrics['return_delta_pct'])}",
        f"- Net PnL delta (left-right): {_fmt_money(metrics['net_pnl_delta'])}",
        f"- Trade count delta (left-right): {metrics['trade_count_delta']:+d}",
        f"- Max drawdown delta (left-right): {_fmt_money(metrics['max_drawdown_delta'])}",
        f"- Sharpe delta (left-right): {_fmt_float(metrics['sharpe_delta'])}",
        "",
        "## Menu Parity",
        "",
        f"- Decision times compared: {menus['decision_times_compared']}",
        f"- Avg finalist overlap: {_fmt_float(menus['avg_finalist_overlap'])}",
        f"- Identical finalist menus: {menus['decision_times_with_identical_finalists']}",
        f"- Avg proposed-symbol overlap: {_fmt_float(menus['avg_proposed_symbol_overlap'])}",
        f"- Identical proposed-symbol sets: {menus['decision_times_with_identical_proposals']}",
        "",
        "## Fill Parity",
        "",
        f"- Shared trade signatures: {fills['shared_trade_signatures']}",
        f"- Left-only trade signatures: {fills['left_only_trade_signatures']}",
        f"- Right-only trade signatures: {fills['right_only_trade_signatures']}",
        f"- Avg entry premium delta: {_fmt_float(fills['avg_entry_premium_delta'])}",
        f"- Avg exit premium delta: {_fmt_float(fills['avg_exit_premium_delta'])}",
        f"- Underlying overlap ratio: {_fmt_float(fills['underlying_overlap_ratio'])}",
    ]
    return "\n".join(lines)


def _summary_markdown(
    *,
    label: str,
    git_commit: str | None,
    summary: ExperimentSummary | None,
    window_results: list[ExperimentWindowResult],
    baseline: ExperimentBaseline,
    prepare_result: PrepareBacktestResult | None,
) -> str:
    lines = [f"# Experiment: {label}", ""]
    lines.append(f"- Generated: {now_eastern().isoformat()}")
    lines.append(f"- Git commit: `{git_commit or 'unknown'}`")
    lines.append(
        f"- Historical options provider: `{config.resolved_historical_options_provider()}`"
    )
    lines.append(
        f"- README baseline: {baseline.start_date} to {baseline.end_date} | "
        f"return {_fmt_pct(baseline.total_return_pct)} | "
        f"Sharpe {_fmt_float(baseline.sharpe_ratio)} | "
        f"max DD {_fmt_money(baseline.max_drawdown)} | "
        f"LLM failure days {baseline.llm_failure_days}"
    )
    if prepare_result is not None:
        lines.append(
            f"- Cache prep: {prepare_result.start_date} to {prepare_result.end_date} | "
            f"{prepare_result.days_prepared} trading days | "
            f"{prepare_result.cache_entries} cache entries"
        )
    lines.append("")
    lines.append("## Windows")
    lines.append("")
    for window_result in window_results:
        result = window_result.result
        lines.append(
            f"- `{window_result.window.label}` ({window_result.window.kind}) "
            f"{window_result.window.start_date} -> {window_result.window.end_date} | "
            f"provider `{result.historical_options_provider or 'unknown'}` | "
            f"return {_fmt_pct(result.total_return_pct)} | "
            f"PnL {_fmt_money(result.net_pnl)} | "
            f"Sharpe {_fmt_float(result.sharpe_ratio)} | "
            f"PF {_fmt_float(result.profit_factor)} | "
            f"max DD {_fmt_money(result.max_drawdown)} | "
            f"trades {result.total_trades} | "
            f"LLM failure days {result.llm_failure_days}"
        )
        if result.log_db_path:
            lines.append(f"  - DB log: `{result.log_db_path}`")
    if summary is None:
        return "\n".join(lines)

    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(
        f"- Rolling windows positive: {summary.positive_windows}/{summary.window_count} "
        f"({summary.positive_window_ratio:.0%})"
    )
    lines.append(f"- Avg return: {_fmt_pct(summary.avg_return_pct)}")
    lines.append(f"- Median return: {_fmt_pct(summary.median_return_pct)}")
    lines.append(f"- Avg Sharpe: {_fmt_float(summary.avg_sharpe_ratio)}")
    lines.append(f"- Avg profit factor: {_fmt_float(summary.avg_profit_factor)}")
    lines.append(f"- Worst drawdown: {_fmt_money(summary.worst_max_drawdown)}")
    lines.append(f"- Total LLM failure days: {summary.total_llm_failure_days}")
    lines.append(f"- Total LLM error cycles: {summary.total_llm_error_cycles}")
    lines.append(f"- Promotion decision: `{summary.promotion_status}`")
    for reason in summary.promotion_reasons:
        lines.append(f"- Reason: {reason}")
    if summary.reference_comparison is not None:
        reference = summary.reference_comparison
        lines.append("")
        lines.append("## Anchor Comparison")
        lines.append("")
        lines.append(
            f"- Anchor return delta vs README baseline: {_fmt_pct(reference.return_delta_pct)}"
        )
        lines.append(f"- Anchor Sharpe delta: {_fmt_float(reference.sharpe_delta)}")
        lines.append(
            f"- Anchor max drawdown delta: {_fmt_money(reference.max_drawdown_delta)}"
        )
        lines.append(
            f"- Anchor LLM failure day delta: {reference.llm_failure_day_delta:+d}"
        )
    return "\n".join(lines)


def _print_summary(
    summary: ExperimentSummary | None,
    window_results: list[ExperimentWindowResult],
    baseline: ExperimentBaseline,
    label: str,
    output_dir: Path,
) -> None:
    print("=" * 72)
    print("  AI TRADER EXPERIMENT SUITE")
    print("=" * 72)
    print(f"  Label:              {label}")
    print(f"  Output dir:         {output_dir}")
    print(
        "  Hist. provider:    "
        f"{config.resolved_historical_options_provider()}"
    )
    print(
        "  README baseline:    "
        f"{baseline.start_date} -> {baseline.end_date} | "
        f"return {_fmt_pct(baseline.total_return_pct)} | "
        f"Sharpe {_fmt_float(baseline.sharpe_ratio)} | "
        f"max DD {_fmt_money(baseline.max_drawdown)}"
    )
    print("")
    print("  Windows:")
    for window_result in window_results:
        result = window_result.result
        print(
            f"    {window_result.window.label} ({window_result.window.kind}) "
            f"{window_result.window.start_date} -> {window_result.window.end_date} | "
            f"provider {result.historical_options_provider or 'unknown'} | "
            f"return {_fmt_pct(result.total_return_pct)} | "
            f"Sharpe {_fmt_float(result.sharpe_ratio)} | "
            f"PF {_fmt_float(result.profit_factor)} | "
            f"max DD {_fmt_money(result.max_drawdown)} | "
            f"trades {result.total_trades} | "
            f"LLM failure days {result.llm_failure_days}"
        )
        if result.log_db_path:
            print(f"      DB log: {result.log_db_path}")
    if summary is None:
        print("=" * 72)
        return
    print("")
    print(
            "  Rolling summary:    "
            f"{summary.positive_windows}/{summary.window_count} positive "
            f"({summary.positive_window_ratio:.0%}) | "
            f"avg return {_fmt_pct(summary.avg_return_pct)} | "
            f"median return {_fmt_pct(summary.median_return_pct)} | "
            f"worst max DD {_fmt_money(summary.worst_max_drawdown)} | "
            f"LLM failures {summary.total_llm_failure_days} day(s) / "
            f"{summary.total_llm_error_cycles} cycle(s)"
        )
    if summary.reference_comparison is not None:
        reference = summary.reference_comparison
        print(
            "  Anchor delta:       "
            f"return {_fmt_pct(reference.return_delta_pct)} | "
            f"Sharpe {_fmt_float(reference.sharpe_delta)} | "
            f"max DD {_fmt_money(reference.max_drawdown_delta)} | "
            f"LLM failures {reference.llm_failure_day_delta:+d}"
        )
    print(f"  Promotion:          {summary.promotion_status}")
    for reason in summary.promotion_reasons:
        print(f"    - {reason}")
    print("=" * 72)


def _base_backtest_config(args: argparse.Namespace) -> BacktestConfig:
    return BacktestConfig(
        start_date=README_BASELINE.start_date,
        end_date=README_BASELINE.end_date,
        initial_equity=args.equity,
        llm_delay_seconds=args.delay,
        decision_interval_minutes=args.decision_interval,
        signal_bar_minutes=args.bar_minutes,
        news_lookback_hours=args.news_lookback_hours,
        use_journal=not args.no_journal,
        journal_max_active=args.journal_max_active,
        journal_max_full_display=args.journal_max_display,
        offline=args.offline,
        prepare_only=False,
        cache_db_path=Path(args.cache_db) if args.cache_db else None,
        prepare_prefetch_symbols=args.prepare_prefetch_symbols,
        prepare_prefetch_contracts_per_side=args.prepare_prefetch_contracts,
    )


def _suite_config_to_dict(
    base_config: BacktestConfig,
    *,
    anchor_end: date,
    window_count: int,
    window_months: int,
    step_months: int,
) -> dict:
    return {
        "anchor_end": anchor_end.isoformat(),
        "window_count": window_count,
        "window_months": window_months,
        "step_months": step_months,
        "initial_equity": base_config.initial_equity,
        "llm_delay_seconds": base_config.llm_delay_seconds,
        "decision_interval_minutes": base_config.decision_interval_minutes,
        "signal_bar_minutes": base_config.signal_bar_minutes,
        "news_lookback_hours": base_config.news_lookback_hours,
        "use_journal": base_config.use_journal,
        "journal_max_active": base_config.journal_max_active,
        "journal_max_full_display": base_config.journal_max_full_display,
        "offline": base_config.offline,
        "cache_db_path": str(base_config.cache_db_path) if base_config.cache_db_path else None,
        "prepare_prefetch_symbols": base_config.prepare_prefetch_symbols,
        "prepare_prefetch_contracts_per_side": base_config.prepare_prefetch_contracts_per_side,
        "resolved_model": config.resolved_llm_model(),
        "historical_options_provider": config.resolved_historical_options_provider(),
    }


def run() -> None:
    parser = argparse.ArgumentParser(
        description="Run rolling 3-month experiment suites for the AI trader."
    )
    parser.add_argument(
        "--label",
        type=str,
        default="manual-experiment",
        help="Short label used in saved experiment artifacts",
    )
    parser.add_argument(
        "--anchor-end",
        type=str,
        default=DEFAULT_ANCHOR_END.isoformat(),
        help="Last window end date (YYYY-MM-DD). Default matches the README baseline end.",
    )
    parser.add_argument(
        "--window-count",
        type=int,
        default=DEFAULT_WINDOW_COUNT,
        help="How many rolling windows to evaluate (default: 6)",
    )
    parser.add_argument(
        "--window-months",
        type=int,
        default=DEFAULT_WINDOW_MONTHS,
        help="Months per evaluation window (default: 3)",
    )
    parser.add_argument(
        "--step-months",
        type=int,
        default=DEFAULT_STEP_MONTHS,
        help="Months between successive window end dates (default: 1)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for experiment outputs (default: ai_trader/logs/experiments/<timestamp>-<label>)",
    )
    parser.add_argument(
        "--compare-left",
        type=str,
        default=None,
        help="Compare mode: left experiment dir or backtest result JSON",
    )
    parser.add_argument(
        "--compare-right",
        type=str,
        default=None,
        help="Compare mode: right experiment dir or backtest result JSON",
    )
    parser.add_argument(
        "--equity",
        type=float,
        default=100_000,
        help="Starting equity for each backtest window (default: 100000)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Seconds between LLM calls during each window run (default: 1.0)",
    )
    parser.add_argument(
        "--decision-interval",
        type=int,
        default=config.SCAN_INTERVAL_MINUTES,
        help="Minutes between decision points (default: live scan interval)",
    )
    parser.add_argument(
        "--bar-minutes",
        type=int,
        default=5,
        help="Intraday bar size for state and fills (default: 5)",
    )
    parser.add_argument(
        "--news-lookback-hours",
        type=int,
        default=config.NEWS_LOOKBACK_HOURS,
        help="Hours of news context to include before each decision time",
    )
    parser.add_argument(
        "--no-journal",
        action="store_true",
        help="Disable thesis journal and trade history context",
    )
    parser.add_argument(
        "--journal-max-active",
        type=int,
        default=8,
        help="Max active theses in the journal (default: 8)",
    )
    parser.add_argument(
        "--journal-max-display",
        type=int,
        default=5,
        help="Max theses shown in full detail (default: 5)",
    )
    parser.add_argument(
        "--cache-db",
        type=str,
        default=None,
        help="SQLite DB path for the persistent historical data cache",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="Fail on historical cache misses instead of using the network",
    )
    parser.add_argument(
        "--prepare-data",
        action="store_true",
        help="Warm the historical data cache for the full suite span before window runs",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Warm the cache for the full suite span and stop before LLM backtests",
    )
    parser.add_argument(
        "--prepare-prefetch-symbols",
        type=int,
        default=config.PREPARE_PREFETCH_SYMBOLS,
        help="In cache prep mode, number of symbols to broaden option prefetch for",
    )
    parser.add_argument(
        "--prepare-prefetch-contracts",
        type=int,
        default=config.PREPARE_PREFETCH_CONTRACTS_PER_SIDE,
        help="In cache prep mode, contracts per side to prefetch for each symbol",
    )
    args = parser.parse_args()

    env_path = Path(__file__).with_name(".env")
    if not env_path.exists():
        env_path = Path(__file__).parent.parent / "momentum_trader" / ".env"
    load_dotenv(env_path, override=True)

    if bool(args.compare_left) != bool(args.compare_right):
        raise ValueError("--compare-left and --compare-right must be used together")
    if args.compare_left and args.compare_right:
        left_path = Path(args.compare_left)
        right_path = Path(args.compare_right)
        comparison = compare_experiment_runs(left_path, right_path)
        output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.label)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "provider_compare.json").write_text(json.dumps(comparison, indent=2))
        (output_dir / "provider_compare.md").write_text(_comparison_markdown(comparison))
        print("=" * 72)
        print("  AI TRADER PROVIDER COMPARISON")
        print("=" * 72)
        print(f"  Left:               {comparison['left']['historical_options_provider']}")
        print(f"  Right:              {comparison['right']['historical_options_provider']}")
        print(f"  Output dir:         {output_dir}")
        print(
            "  Menu parity:        "
            f"{comparison['menus']['decision_times_compared']} shared decision times | "
            f"avg finalist overlap {_fmt_float(comparison['menus']['avg_finalist_overlap'])} | "
            f"avg proposal overlap {_fmt_float(comparison['menus']['avg_proposed_symbol_overlap'])}"
        )
        print(
            "  Fill parity:        "
            f"{comparison['fills']['shared_trade_signatures']} shared trades | "
            f"entry delta {_fmt_float(comparison['fills']['avg_entry_premium_delta'])} | "
            f"exit delta {_fmt_float(comparison['fills']['avg_exit_premium_delta'])}"
        )
        print(
            "  Metrics delta:      "
            f"return {_fmt_pct(comparison['metrics']['return_delta_pct'])} | "
            f"PnL {_fmt_money(comparison['metrics']['net_pnl_delta'])} | "
            f"max DD {_fmt_money(comparison['metrics']['max_drawdown_delta'])}"
        )
        print("=" * 72)
        return

    anchor_end = date.fromisoformat(args.anchor_end)
    base_config = _base_backtest_config(args)
    rolling_windows = build_rolling_windows(
        anchor_end,
        count=args.window_count,
        window_months=args.window_months,
        step_months=args.step_months,
    )
    baseline_key = _window_key(README_BASELINE.start_date, README_BASELINE.end_date)
    needs_reference_window = baseline_key not in {
        _window_key(window.start_date, window.end_date)
        for window in rolling_windows
    }
    windows_to_run = list(rolling_windows)
    if needs_reference_window:
        windows_to_run.append(
            ExperimentWindow(
                label="reference_baseline",
                start_date=README_BASELINE.start_date,
                end_date=README_BASELINE.end_date,
                kind="reference",
            )
        )

    output_dir = Path(args.output_dir) if args.output_dir else _default_output_dir(args.label)
    output_dir.mkdir(parents=True, exist_ok=True)

    git_commit = _git_commit(Path(__file__).resolve().parent.parent)
    prepare_result = None
    if args.prepare_data or args.prepare_only:
        suite_start = min(window.start_date for window in windows_to_run)
        suite_end = max(window.end_date for window in windows_to_run)
        prepare_config = replace(
            base_config,
            start_date=suite_start,
            end_date=suite_end,
            llm_delay_seconds=0.0,
            prepare_only=True,
        )
        log(
            f"preparing historical cache for suite span {suite_start} -> {suite_end}"
        )
        prepare_result = prepare_backtest_data(prepare_config)

    window_results: list[ExperimentWindowResult] = []
    summary: ExperimentSummary | None = None
    if not args.prepare_only:
        for window in windows_to_run:
            log(
                f"running {window.label} ({window.kind}) "
                f"{window.start_date} -> {window.end_date}"
            )
            window_config = replace(
                base_config,
                start_date=window.start_date,
                end_date=window.end_date,
                prepare_only=False,
                log_db_path=output_dir / (
                    f"{window.label}_{window.start_date.isoformat()}_{window.end_date.isoformat()}.db"
                ),
            )
            result = run_backtest(window_config)
            result_path = output_dir / (
                f"{window.label}_{window.start_date.isoformat()}_{window.end_date.isoformat()}.json"
            )
            debug_path = Path(f"{result_path}.debug.md")
            save_backtest_result(result, result_path)
            save_debug_log(result, debug_path)
            window_results.append(
                ExperimentWindowResult(
                    window=window,
                    result=result,
                    result_path=result_path,
                    debug_log_path=debug_path,
                )
            )
        summary = summarize_experiment_suite(window_results, baseline=README_BASELINE)

    manifest = {
        "label": args.label,
        "generated_at": now_eastern().isoformat(),
        "git_commit": git_commit,
        "output_dir": str(output_dir),
        "readme_baseline": _baseline_to_dict(README_BASELINE),
        "suite_config": _suite_config_to_dict(
            base_config,
            anchor_end=anchor_end,
            window_count=args.window_count,
            window_months=args.window_months,
            step_months=args.step_months,
        ),
        "prepare_result": _prepare_result_to_dict(prepare_result),
        "windows": [summarize_window_result(result) for result in window_results],
        "summary": _summary_to_dict(summary) if summary is not None else None,
    }
    (output_dir / "summary.json").write_text(json.dumps(manifest, indent=2))
    (output_dir / "summary.md").write_text(
        _summary_markdown(
            label=args.label,
            git_commit=git_commit,
            summary=summary,
            window_results=window_results,
            baseline=README_BASELINE,
            prepare_result=prepare_result,
        )
    )
    _print_summary(summary, window_results, README_BASELINE, args.label, output_dir)


if __name__ == "__main__":
    run()
