import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest


from IndicatorSignals import IndicatorSignals

class TestIndicatorSignals:

    def test_ma_cross_signal(self):
        dates = pd.date_range(start="2021-01-01", periods=10)
        price = pd.Series([10, 11, 12, 13, 14, 13, 12, 14, 15, 16], index=dates)
        price_data = pd.DataFrame({"Close": price})
        signals = IndicatorSignals(price_data)
        ma_signal = signals.generate_ma_cross_signal(short_window=2, long_window=5)

        assert set(ma_signal.unique()).issubset({0, 1})

    def test_rsi_signal(self):
        dates = pd.date_range(start="2021-01-01", periods=30)
        price = pd.Series(np.linspace(100, 80, 30), index=dates)
        price_data = pd.DataFrame({"Close": price})
        signals = IndicatorSignals(price_data)
        rsi_signal = signals.generate_rsi_signal(rsi_length=14)

        assert set(rsi_signal.unique()).issubset({0, 1})

    def test_macd_signal(self):
        dates = pd.date_range(start="2021-01-01", periods=50)
        price = pd.Series(np.random.normal(100, 1, 50), index=dates)
        price_data = pd.DataFrame({"Close": price})
        signals = IndicatorSignals(price_data)
        macd_signal = signals.generate_macd_signal()
        assert set(macd_signal.unique()).issubset({0, 1})

    def test_bollinger_bands_signal(self):
        dates = pd.date_range(start="2021-01-01", periods=50)
        price = pd.Series(np.random.normal(100, 1, 50), index=dates)
        price_data = pd.DataFrame({"Close": price})
        signals = IndicatorSignals(price_data)
        bb_signal = signals.generate_bollinger_bands_signal()
        assert set(bb_signal.unique()).issubset({0, 1})

    def test_adx_signal(self):
        dates = pd.date_range(start="2021-01-01", periods=50)
        price = pd.Series(np.random.normal(100, 1, 50), index=dates)
        high = price + np.random.normal(1, 0.5, 50)
        low = price - np.random.normal(1, 0.5, 50)
        price_data = pd.DataFrame({"High": high, "Low": low, "Close": price})
        signals = IndicatorSignals(price_data)
        adx_signal = signals.generate_adx_signal()
        assert set(adx_signal.unique()).issubset({0, 1})