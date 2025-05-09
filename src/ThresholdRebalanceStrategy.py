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
from src.data_handler import DataHandler
from src.indicator_signals import IndicatorSignals
from src.plot_utils import plot_log_scale_equity, plot_strategy_vs_buyhold, plot_strategy_vs_buyhold_with_markers, plot_smoothed_return_contour
from src.backtest_runner import BacktestRunner

DEBUG = False

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

import backtrader as bt

class ThresholdRebalanceStrategy(bt.Strategy):
    params = (
        ("buy_threshold", 0.98),
        ("sell_threshold", 1.02),
        ("position_frac", 0.15),
    )
    def __init__(self):
        self.dataclose = self.datas[0].close
        self.trade_stack = []
        self.x = None

    def notify_order(self, order):
        if order.status in [order.Completed]:
            action = "BUY" if order.isbuy() else "SELL"
            dt = self.data.datetime.datetime(0)
            price = order.executed.price
            size = order.executed.size
            debug_print(f"{dt} - {action} EXECUTED: Price = {price:.2f}, Size = {size}")

    def next(self):
        # print(f"Current time: {self.datas[0].datetime.datetime(0)}")
        price = self.dataclose[0]
        cash = self.broker.get_cash()
        current_position = self.getposition(self.data).size

        if self.x is None:
            capital = self.broker.get_value()
            self.x = max(1, int((self.params.position_frac * capital) / price))  # Ensure at least 1 share
            self.buy(size=self.x)
            self.trade_stack.append(price)
            return

        # Check if we should sell (latest buy earned 2%)
        while self.trade_stack and price >= self.trade_stack[-1] * self.params.sell_threshold:
            sell_size = min(self.x, current_position)
            if sell_size > 0:
                self.sell(size=sell_size)
                self.trade_stack.pop()
                current_position -= sell_size
            else:
                break

        # Check if we should buy (price dropped 2% from last buy)
        if not self.trade_stack or price <= self.trade_stack[-1] * self.params.buy_threshold:
            self.buy(size=self.x)
            self.trade_stack.append(price)
            stock_value = self.getposition(self.data).size * price
            total_value = self.broker.get_value()
            debug_print(f"Buying at {price:.2f}. Stock Value: {stock_value:.2f}, Total Value: {total_value:.2f}, Allocation: {stock_value / total_value:.2%}")

        # Sell everything at the start of a new day
        current_date = self.datas[0].datetime.date(0)
        if hasattr(self, 'prev_date') and current_date != self.prev_date:
            if current_position > 0:
                self.sell(size=current_position)
                current_position = 0
            self.trade_stack = []
            self.x = None
        self.prev_date = current_date

def testThresholdReblance(backtest_ticker, start_date, end_date, buy_and_hold_per,
        buy_threshold_,
        sell_threshold_,
        position_frac_):
#price_data = handler.generate_ar1_data(mu = 0.001, sigma=0.95, n_steps=3000)
    handler = DataHandler(backtest_ticker, start_date=start_date, end_date=end_date, interval_param="5m")
    price_data = handler.download_data()

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
        buy_threshold=buy_threshold_,
        sell_threshold=sell_threshold_,
        position_frac=position_frac_
    )


    cerebro.broker.set_cash(10000000 * (1 - buy_and_hold_per))

    debug_print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    debug_print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Portfolio statistics

    start_value = cerebro.broker.startingcash + 10000000 * buy_and_hold_per
    end_value = cerebro.broker.getvalue() + (1 + buy_hold_return) * 10000000 * buy_and_hold_per
    
    total_return = (end_value - start_value) / start_value

    print(f"Total Return: {total_return * 100:.2f}%")
    print(f"Buy and Hold Return: {buy_hold_return * 100:.2f}%")
    # cerebro.plot(style='candlestick')
    return total_return, buy_hold_return