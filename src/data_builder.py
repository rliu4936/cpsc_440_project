from curl_cffi import requests
import yfinance as yf
import pandas as pd
import time
from datetime import datetime, timedelta

def download_full_intraday(ticker, start_date, end_date, interval='5m', chunk_size=59):
    session = requests.Session(impersonate="chrome")
    ticker_obj = yf.Ticker(ticker, session=session)

    all_data = []

    # Convert to datetime objects
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)

    while start_dt < end_dt:
        chunk_end = min(start_dt + timedelta(days=chunk_size), end_dt)

        print(f"[INFO] Downloading {ticker} from {start_dt.date()} to {chunk_end.date()}")

        try:
            data = ticker_obj.history(
                start=start_dt,
                end=chunk_end,
                interval=interval
            )
            if not data.empty:
                all_data.append(data)
        except Exception as e:
            print(f"[ERROR] Failed for {start_dt} to {chunk_end}: {e}")

        start_dt = chunk_end
        time.sleep(2)  # pause to avoid rate limiting

    if all_data:
        df = pd.concat(all_data)
        df = df[~df.index.duplicated(keep='first')]  # remove duplicates if any
        df.to_csv(f"{ticker}_{interval}_from_{start_date}_to_{end_date}.csv")
        print(f"[SUCCESS] Saved to {ticker}_{interval}_from_{start_date}_to_{end_date}.csv")
        return df
    else:
        print("[WARNING] No data downloaded.")
        return pd.DataFrame()

# Example usage
df = download_full_intraday(
    ticker="7226.HK",
    start_date="2023-01-01",
    end_date="2025-05-08"
)