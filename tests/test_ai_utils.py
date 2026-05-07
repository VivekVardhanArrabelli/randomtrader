"""Tests for shared AI trader utilities."""

from ai_trader.utils import is_equity_candidate_symbol, prioritized_symbol_watchlist


def test_prioritized_symbol_watchlist_preserves_priority_groups():
    watchlist = prioritized_symbol_watchlist(
        ["xlp", "qqq"],
        ["xlp", "xom"],
        ["xle", "qqq", "pld"],
        limit=4,
    )

    assert watchlist == ["XLP", "QQQ", "XOM", "XLE"]


def test_is_equity_candidate_symbol_filters_warrants_units_and_rights():
    assert is_equity_candidate_symbol("AAPL")
    assert is_equity_candidate_symbol("W")
    assert not is_equity_candidate_symbol("VWAVW")
    assert not is_equity_candidate_symbol("ABCDU")
    assert not is_equity_candidate_symbol("XYZRR")
