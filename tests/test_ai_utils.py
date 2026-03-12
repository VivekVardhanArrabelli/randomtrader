"""Tests for shared AI trader utilities."""

from ai_trader.utils import prioritized_symbol_watchlist


def test_prioritized_symbol_watchlist_preserves_priority_groups():
    watchlist = prioritized_symbol_watchlist(
        ["xlp", "qqq"],
        ["xlp", "xom"],
        ["xle", "qqq", "pld"],
        limit=4,
    )

    assert watchlist == ["XLP", "QQQ", "XOM", "XLE"]
