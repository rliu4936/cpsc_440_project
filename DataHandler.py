import yfinance as yf
import pandas as pd
import backtrader as bt
from datetime import datetime

class DataHandler:
    def __init__(self, tickers, start_date, end_date):
        if isinstance(tickers, str):
            tickers = [tickers]
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def download_data(self):
        data = []
        for ticker in self.tickers:
            df = yf.download(ticker, start=self.start_date, end=self.end_date)
            df.columns = df.columns.get_level_values(0)  # Flatten if multiindex
            df.index.name = "Date"
            df['Ticker'] = ticker
            data.append(df)
        combined = pd.concat(data)
        self.data = combined
        self.data = combined.sort_index()
        return self.data

    def get_backtrader_data(self):
        if self.data is None:
            raise ValueError("Data not downloaded yet. Call download_data() first.")
        if len(self.tickers) == 1:
            return bt.feeds.PandasData(dataname=self.data)
        else:
            return {ticker: bt.feeds.PandasData(dataname=self.data[self.data['Ticker'] == ticker]) for ticker in self.tickers}

    def save_to_csv(self, filepath_prefix):
        if self.data is None:
            raise ValueError("No data to save. Call download_data() first.")
        if len(self.tickers) == 1:
            self.data.to_csv(f"{filepath_prefix}_{self.tickers[0]}.csv")
        else:
            for ticker in self.tickers:
                ticker_data = self.data[self.data['Ticker'] == ticker]
                ticker_data.to_csv(f"{filepath_prefix}_{ticker}.csv")

    def load_from_csv(self, filepath_prefix):
        data = []
        for ticker in self.tickers:
            df = pd.read_csv(f"{filepath_prefix}_{ticker}.csv", index_col="Date", parse_dates=True)
            df['Ticker'] = ticker
            data.append(df)
        self.data = pd.concat(data).sort_index()
        return self.data