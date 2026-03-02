"""Tests for bar helpers and basic scanner filters."""

from __future__ import annotations

import unittest
from datetime import datetime

from momentum_trader.bars import Bar, atr, consolidation_breakout, vwap
from momentum_trader.scanner import MarketScanner, MarketSnapshot
from momentum_trader.utils import EASTERN_TZ


class TestBars(unittest.TestCase):
    def test_atr_constant_range(self) -> None:
        now = datetime(2024, 1, 2, 10, 0, tzinfo=EASTERN_TZ)
        bars = [
            Bar(
                timestamp=now,
                open=10.0,
                high=11.0,
                low=9.0,
                close=10.0,
                volume=100.0,
            )
            for _ in range(15)
        ]
        self.assertEqual(atr(bars, period=14), 2.0)

    def test_vwap_weighting(self) -> None:
        now = datetime(2024, 1, 2, 10, 0, tzinfo=EASTERN_TZ)
        bars = [
            Bar(timestamp=now, open=10.0, high=10.0, low=10.0, close=10.0, volume=1.0),
            Bar(timestamp=now, open=20.0, high=20.0, low=20.0, close=20.0, volume=3.0),
        ]
        self.assertAlmostEqual(vwap(bars) or 0.0, 17.5, places=6)

    def test_consolidation_breakout_true(self) -> None:
        now = datetime(2024, 1, 2, 10, 0, tzinfo=EASTERN_TZ)
        consolidation = [
            Bar(timestamp=now, open=9.5, high=10.0, low=9.0, close=9.6, volume=100.0)
            for _ in range(5)
        ]
        breakout = Bar(timestamp=now, open=9.9, high=10.2, low=9.8, close=10.02, volume=250.0)
        ok, _ = consolidation_breakout(
            consolidation + [breakout],
            consolidation_bars=5,
            max_range_pct=0.15,
            breakout_buffer_pct=0.001,
            volume_multiplier=1.5,
        )
        self.assertTrue(ok)


class TestScannerFilters(unittest.TestCase):
    def test_scanner_excludes_untradeable(self) -> None:
        snapshots = [
            MarketSnapshot(
                symbol="TESTW",
                open_price=1.0,
                current_price=2.0,
                relative_volume=10.0,
                market_cap=100_000_000.0,
                shares_outstanding=10_000_000.0,
                is_tradeable=False,
            )
        ]
        candidates = MarketScanner(snapshots).scan()
        self.assertEqual(candidates, [])

    def test_scanner_allows_common_stock(self) -> None:
        snapshots = [
            MarketSnapshot(
                symbol="TEST",
                open_price=1.0,
                current_price=2.0,
                relative_volume=10.0,
                market_cap=100_000_000.0,
                shares_outstanding=10_000_000.0,
                is_tradeable=True,
            )
        ]
        candidates = MarketScanner(snapshots).scan()
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].symbol, "TEST")


if __name__ == "__main__":
    unittest.main()

