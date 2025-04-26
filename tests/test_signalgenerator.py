


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pytest
from SignalGenerator import SignalGenerator

def test_generate_ma_crossover_signal_basic():
    # Create dummy price data
    dates = pd.date_range(start="2021-01-01", periods=50)
    close_prices = np.linspace(100, 150, 50)
    price_data = pd.DataFrame({"Close": close_prices}, index=dates)

    # Generate signals
    generator = SignalGenerator(price_data)
    signals = generator.generate_ma_crossover_signal(short_window=5, long_window=10)

    # Check signals DataFrame
    assert 'signal' in signals.columns, "Signals should have a 'signal' column."
    assert len(signals) == len(price_data), "Signals should match price data length."

def test_generate_ma_crossover_signal_values():
    dates = pd.date_range(start="2021-01-01", periods=20)
    close_prices = np.concatenate([np.ones(10)*100, np.ones(10)*200])  # clear crossover
    price_data = pd.DataFrame({"Close": close_prices}, index=dates)

    generator = SignalGenerator(price_data)
    signals = generator.generate_ma_crossover_signal(short_window=3, long_window=5)

    # After crossover should give positive signals
    later_signals = signals['signal'].iloc[10:]
    assert (later_signals == 1).sum() > 0, "After crossover, there should be positive signals."
def test_generate_rsi_signal_basic():
    # Create dummy price data with alternating gains and losses
    dates = pd.date_range(start="2021-01-01", periods=30)
    close_prices = np.array([100 + (-1)**i * 2 for i in range(30)])
    price_data = pd.DataFrame({"Close": close_prices}, index=dates)

    generator = SignalGenerator(price_data)
    signals = generator.generate_rsi_signal(period=5, overbought=60, oversold=40)

    # Check signals DataFrame
    assert 'rsi_signal' in signals.columns, "Signals should have a 'rsi_signal' column."
    assert len(signals) == len(price_data), "Signals should match price data length."