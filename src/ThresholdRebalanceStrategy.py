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

DEBUG = True

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
            print(f"{dt} - {action} EXECUTED: Price = {price:.2f}, Size = {size}")

    def next(self):
        print(f"Current time: {self.datas[0].datetime.datetime(0)}")
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
            print(f"Buying at {price:.2f}. Stock Value: {stock_value:.2f}, Total Value: {total_value:.2f}, Allocation: {stock_value / total_value:.2%}")

        # Sell everything every 2nd day assuming minute data
        if len(self) % 156 == 0:
            if current_position > 0:
                self.sell(size=current_position)
                current_position = 0
            self.trade_stack = []
            self.x = None
