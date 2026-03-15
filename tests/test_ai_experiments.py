from datetime import date
from pathlib import Path

from ai_trader.backtest import BacktestResult, backtest_result_to_dict
from ai_trader.experiments import (
    README_BASELINE,
    ExperimentWindow,
    ExperimentWindowResult,
    _shift_months,
    build_rolling_windows,
    compare_to_readme_baseline,
    summarize_experiment_suite,
    summarize_window_result,
)


def _window_result(
    label: str,
    start_date: date,
    end_date: date,
    *,
    kind: str = "rolling",
    total_return_pct: float,
    sharpe_ratio: float | None,
    profit_factor: float | None,
    max_drawdown: float,
    total_trades: int,
    llm_failure_days: int = 0,
    log_db_path: str | None = None,
) -> ExperimentWindowResult:
    initial_equity = 100_000.0
    final_equity = initial_equity * (1 + total_return_pct)
    return ExperimentWindowResult(
        window=ExperimentWindow(
            label=label,
            start_date=start_date,
            end_date=end_date,
            kind=kind,
        ),
        result=BacktestResult(
            initial_equity=initial_equity,
            final_equity=final_equity,
            total_return_pct=total_return_pct,
            net_pnl=final_equity - initial_equity,
            total_trades=total_trades,
            wins=max(total_trades // 2, 0),
            losses=max(total_trades - (total_trades // 2), 0),
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            days_tested=63,
            llm_failure_days=llm_failure_days,
            log_db_path=log_db_path,
        ),
        result_path=Path("/tmp/result.json"),
        debug_log_path=Path("/tmp/result.json.debug.md"),
    )


def test_shift_months_clamps_end_of_month():
    assert _shift_months(date(2026, 3, 31), -1) == date(2026, 2, 28)


def test_build_rolling_windows_includes_readme_baseline_as_latest():
    windows = build_rolling_windows(README_BASELINE.end_date, count=3)

    assert [(window.start_date, window.end_date) for window in windows] == [
        (date(2025, 9, 13), date(2025, 12, 13)),
        (date(2025, 10, 13), date(2026, 1, 13)),
        (README_BASELINE.start_date, README_BASELINE.end_date),
    ]


def test_compare_to_readme_baseline_reports_metric_deltas():
    comparison = compare_to_readme_baseline(
        _window_result(
            "rolling_03",
            README_BASELINE.start_date,
            README_BASELINE.end_date,
            total_return_pct=0.15,
            sharpe_ratio=1.05,
            profit_factor=1.3,
            max_drawdown=12_000.0,
            total_trades=12,
        )
    )

    assert comparison.return_delta_pct == 0.15 - README_BASELINE.total_return_pct
    assert comparison.sharpe_delta == 1.05 - README_BASELINE.sharpe_ratio
    assert comparison.max_drawdown_delta == 12_000.0 - README_BASELINE.max_drawdown
    assert comparison.llm_failure_day_delta == 0


def test_summarize_experiment_suite_marks_keep_for_stronger_reference_and_positive_windows():
    summary = summarize_experiment_suite(
        [
            _window_result(
                "rolling_01",
                date(2025, 9, 13),
                date(2025, 12, 13),
                total_return_pct=0.08,
                sharpe_ratio=0.75,
                profit_factor=1.10,
                max_drawdown=12_500.0,
                total_trades=8,
            ),
            _window_result(
                "rolling_02",
                date(2025, 10, 13),
                date(2026, 1, 13),
                total_return_pct=0.10,
                sharpe_ratio=0.82,
                profit_factor=1.18,
                max_drawdown=12_900.0,
                total_trades=9,
            ),
            _window_result(
                "rolling_03",
                README_BASELINE.start_date,
                README_BASELINE.end_date,
                total_return_pct=0.14,
                sharpe_ratio=1.00,
                profit_factor=1.25,
                max_drawdown=12_000.0,
                total_trades=11,
            ),
        ]
    )

    assert summary.promotion_status == "keep"
    assert summary.reference_comparison is not None
    assert summary.reference_comparison.return_delta_pct > 0
    assert summary.positive_windows == 3


def test_summarize_experiment_suite_can_keep_when_anchor_lags_but_suite_is_robust():
    summary = summarize_experiment_suite(
        [
            _window_result(
                "rolling_01",
                date(2025, 9, 13),
                date(2025, 12, 13),
                total_return_pct=0.07,
                sharpe_ratio=0.70,
                profit_factor=1.08,
                max_drawdown=12_400.0,
                total_trades=8,
            ),
            _window_result(
                "rolling_02",
                date(2025, 10, 13),
                date(2026, 1, 13),
                total_return_pct=0.09,
                sharpe_ratio=0.80,
                profit_factor=1.14,
                max_drawdown=12_900.0,
                total_trades=9,
            ),
            _window_result(
                "rolling_03",
                README_BASELINE.start_date,
                README_BASELINE.end_date,
                total_return_pct=0.115,
                sharpe_ratio=0.90,
                profit_factor=1.16,
                max_drawdown=13_400.0,
                total_trades=11,
            ),
        ]
    )

    assert summary.promotion_status == "keep"
    assert summary.reference_comparison is not None
    assert summary.reference_comparison.return_delta_pct < 0
    assert summary.median_return_pct is not None
    assert summary.median_return_pct > 0


def test_summarize_experiment_suite_marks_discard_for_broad_regression():
    summary = summarize_experiment_suite(
        [
            _window_result(
                "rolling_01",
                date(2025, 9, 13),
                date(2025, 12, 13),
                total_return_pct=-0.04,
                sharpe_ratio=-0.20,
                profit_factor=0.80,
                max_drawdown=14_500.0,
                total_trades=8,
            ),
            _window_result(
                "rolling_02",
                date(2025, 10, 13),
                date(2026, 1, 13),
                total_return_pct=0.01,
                sharpe_ratio=0.10,
                profit_factor=1.01,
                max_drawdown=15_000.0,
                total_trades=7,
            ),
            _window_result(
                "rolling_03",
                README_BASELINE.start_date,
                README_BASELINE.end_date,
                total_return_pct=-0.03,
                sharpe_ratio=0.20,
                profit_factor=0.92,
                max_drawdown=16_500.0,
                total_trades=10,
            ),
        ]
    )

    assert summary.promotion_status == "discard"
    assert summary.positive_windows == 1
    assert summary.reference_comparison is not None
    assert summary.reference_comparison.return_delta_pct < 0


def test_summarize_window_result_stays_lightweight():
    summary = summarize_window_result(
        _window_result(
            "rolling_03",
            README_BASELINE.start_date,
            README_BASELINE.end_date,
            total_return_pct=0.12,
            sharpe_ratio=0.90,
            profit_factor=1.20,
            max_drawdown=13_000.0,
            total_trades=11,
            log_db_path="/tmp/window.db",
        )
    )

    assert summary["label"] == "rolling_03"
    assert "decision_log" not in summary
    assert "trades" not in summary
    assert summary["log_db_path"] == "/tmp/window.db"


def test_summarize_experiment_suite_deduplicates_failure_days_across_overlapping_windows():
    first = _window_result(
        "rolling_01",
        date(2025, 10, 13),
        date(2026, 1, 13),
        total_return_pct=0.02,
        sharpe_ratio=0.40,
        profit_factor=1.05,
        max_drawdown=12_000.0,
        total_trades=7,
        llm_failure_days=1,
    )
    second = _window_result(
        "rolling_02",
        README_BASELINE.start_date,
        README_BASELINE.end_date,
        total_return_pct=0.13,
        sharpe_ratio=0.92,
        profit_factor=1.20,
        max_drawdown=13_000.0,
        total_trades=11,
        llm_failure_days=1,
    )
    first.result.decision_log = [{"date": "2025-12-02", "llm_error": True}]
    second.result.decision_log = [{"date": "2025-12-02", "llm_error": True}]

    summary = summarize_experiment_suite([first, second])

    assert summary.total_llm_failure_days == 1


def test_summarize_experiment_suite_deduplicates_error_cycles_across_overlapping_windows():
    first = _window_result(
        "rolling_01",
        date(2025, 10, 13),
        date(2026, 1, 13),
        total_return_pct=0.02,
        sharpe_ratio=0.40,
        profit_factor=1.05,
        max_drawdown=12_000.0,
        total_trades=7,
    )
    second = _window_result(
        "rolling_02",
        README_BASELINE.start_date,
        README_BASELINE.end_date,
        total_return_pct=0.13,
        sharpe_ratio=0.92,
        profit_factor=1.20,
        max_drawdown=13_000.0,
        total_trades=11,
    )
    first.result.llm_error_cycles = 1
    second.result.llm_error_cycles = 1
    first.result.decision_log = [{"decision_time": "2025-12-02T10:05:00-05:00", "llm_error": True}]
    second.result.decision_log = [{"decision_time": "2025-12-02T10:05:00-05:00", "llm_error": True}]

    summary = summarize_experiment_suite([first, second])

    assert summary.total_llm_error_cycles == 1


def test_backtest_result_to_dict_includes_llm_reliability_fields():
    payload = backtest_result_to_dict(
        BacktestResult(
            llm_error_cycles=2,
            llm_failure_days=1,
            log_db_path="/tmp/backtest.db",
        )
    )

    assert payload["llm_error_cycles"] == 2
    assert payload["llm_failure_days"] == 1
    assert payload["log_db_path"] == "/tmp/backtest.db"
