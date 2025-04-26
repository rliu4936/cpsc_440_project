
import yfinance as yf
import pandas as pd
import backtrader as bt
from datetime import datetime

# DataHandler class for data download and preparation
class DataHandler:
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def download_data(self):
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        df.columns = df.columns.get_level_values(0)  # Flatten if multiindex
        df.index.name = "Date"
        self.data = df
        return df

    def get_backtrader_data(self):
        if self.data is None:
            raise ValueError("Data not downloaded yet. Call download_data() first.")
        return bt.feeds.PandasData(dataname=self.data)

# Download and prepare data using DataHandler
handler = DataHandler("QQQ", start_date="2021-01-01", end_date=datetime.today().strftime("%Y-%m-%d"))
handler.download_data()
data = handler.get_backtrader_data()

class SmaCross(bt.Strategy):
    def __init__(self):
        sma1 = bt.ind.SMA(period=10)
        sma2 = bt.ind.SMA(period=30)
        self.crossover = bt.ind.CrossOver(sma1, sma2)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()

cerebro = bt.Cerebro()
cerebro.addstrategy(SmaCross)
cerebro.adddata(data)
cerebro.run()
figs = figs = cerebro.plot(
    style='candlestick',   # Candlestick bars
    volume=False,          # Hide volume subplot (less clutter)
    barup='green',         # Color for up bars
    bardown='red',         # Color for down bars
    grid=True,             # Add grid for better readability
    figsize=(12, 6),       # Make plot larger
    tight=False            # Avoid clipping of labels
)