import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
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

# Download and prepare data
handler = DataHandler("QQQ", start_date="2021-01-01", end_date=datetime.today().strftime("%Y-%m-%d"))
df = handler.download_data()

# Plotting stock closing prices
plt.figure(figsize=(12, 6))
plt.plot(df.index, df["Close"], label="Close Price", linewidth=2)
plt.title("QQQ Closing Prices")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()