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
from src.ThresholdRebalanceStrategy import ThresholdRebalanceStrategy



DEBUG = True

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

import backtrader as bt


def main():
    # Load data
    backtest_ticker = "7552.HK" # long
    # backtest_ticker = "7226.HK"

    # backtest_ticker = "TQQQ" # long
    backtest_ticker = "QQQ" # long

    start_date = "2025-03-11"
    end_date = datetime.today().strftime("%Y-%m-%d")
    handler = DataHandler(backtest_ticker, start_date=start_date, end_date=end_date, interval_param="5m")
    price_data = handler.download_data()

    #price_data = handler.generate_ar1_data(mu = 0.001, sigma=0.95, n_steps=3000)
    debug_print("[DEBUG] price_data.head():\n", price_data.head())
    debug_print("[DEBUG] price_data.columns:", price_data.columns)

    start_price = price_data['close'].iloc[0]
    end_price = price_data['close'].iloc[-1]
    buy_hold_return = (end_price - start_price) / start_price


     # Backtrader setup
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=price_data)
    cerebro.adddata(data)
    cerebro.addstrategy(
        ThresholdRebalanceStrategy,
        buy_threshold=0.98,
        sell_threshold=1.02,
        position_frac=0.10
    )


    cerebro.broker.set_cash(5000000)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Portfolio statistics

    start_value = cerebro.broker.startingcash + 5000000
    end_value = cerebro.broker.getvalue() + (1 + buy_hold_return) * 5000000
    
    total_return = (end_value - start_value) / start_value

    print(f"Total Return: {total_return * 100:.2f}%")
    print(f"Buy and Hold Return: {buy_hold_return * 100:.2f}%")
    cerebro.plot(style='candlestick')
    

if __name__ == "__main__":
    main()
