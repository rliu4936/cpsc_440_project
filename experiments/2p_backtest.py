import os
import pickle
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
from tqdm import tqdm
import concurrent.futures
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import ThresholdRebalanceStrategy
from src.data_handler import DataHandler
from src.indicator_signals import IndicatorSignals
from src.plot_utils import plot_log_scale_equity, plot_strategy_vs_buyhold, plot_strategy_vs_buyhold_with_markers, plot_smoothed_return_contour
from src.backtest_runner import BacktestRunner
import backtrader as bt
from src.ThresholdRebalanceStrategy import ThresholdRebalanceStrategy, testThresholdReblance



DEBUG = True

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

import backtrader as bt


def main():
    # # Load data
    # backtest_ticker = "7552.HK" # long
    # # backtest_ticker = "7226.HK"

    # # backtest_ticker = "TQQQ" # long
    # backtest_ticker = "QQQ" # long
    # backtest_ticker = "2007.HK" # long

    backtest_tickers = ["003041.SZ", "301276.SZ", "688459.SS", "600639.SS", "605003.SS", "605288.SS", "600422.SS", "3380.HK", "600420.SS", "600380.SS", "600216.SS",
                        "600256.SS", "600219.SS",
                        # newly added
    "600048.SS", "600518.SS", "000002.SZ", "601162.SS", "000988.SZ", "002555.SZ", "601088.SS", "600036.SS", "601006.SS", "600887.SS", "600585.SS"]

    start_date = "2025-03-12"
    end_date = datetime.today().strftime("%Y-%m-%d")
    buy_and_hold_per = 0
     
    actual_returns = []
    buy_hold_returns = []
    ticker_labels = []
    for ticker in backtest_tickers:
        result = testThresholdReblance(ticker, start_date, end_date, buy_and_hold_per, 0.98, 1.02, 0.2)
        # Assuming testThresholdReblance returns (actual_return, buy_hold_return)
        # If not, adapt this unpacking accordingly.
        if result is not None and isinstance(result, (tuple, list)) and len(result) == 2:
            actual_returns.append(result[0])
            buy_hold_returns.append(result[1])
            ticker_labels.append(ticker)
        print()

    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.scatter(buy_hold_returns, actual_returns)
    # Fit line of best fit without bias (force through origin)
    x = np.array(buy_hold_returns)
    y = np.array(actual_returns)
    slope = np.dot(x, y) / np.dot(x, x)
    y_pred = slope * x
    plt.plot(x, y_pred, color='red', label=f'Best Fit (no bias): y = {slope:.2f}x')
    plt.xlabel("Buy and Hold Return")
    plt.ylabel("Strategy Return")
    plt.plot([min(buy_hold_returns), max(buy_hold_returns)],
            [min(buy_hold_returns), max(buy_hold_returns)],
            linestyle='--', color='gray', label='y = x')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title("Strategy vs Buy and Hold Returns")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
