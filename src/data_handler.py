import os
import yfinance as yf
import pandas as pd
import backtrader as bt
from datetime import datetime
class DataHandler:
    def __init__(self, ticker,  start_date = "2000-01-03", end_date = "2025-01-01", interval_param="1D"):

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.interval_param = interval_param

    def generate_ar1_data(self, sigma=1.0, n_steps=1000, freq="1T", mu=0):
        import numpy as np
        import pandas as pd

        eps = np.random.normal(loc=0, scale=sigma, size=n_steps)
        x = [100]
        for i in range(1, n_steps):
            x.append((x[-1] + mu) + eps[i])

        time_index = pd.date_range(start=pd.Timestamp.now(), periods=n_steps, freq=freq)
        df = pd.DataFrame({"close": x}, index=time_index)
        df["open"] = df["high"] = df["low"] = df["close"]
        df["volume"] = 1000
        df.index.name = "datetime"
        self.data = df
        return df

    def download_data(self):

        local_path = f"data/tickers/{self.ticker}.csv"

        if os.path.exists(local_path):
            print(f"[INFO] Loading cached data for {self.ticker}")
            df = pd.read_csv(local_path, parse_dates=["date"])
            self.data = df
            return self.data

        print(f"[INFO] Downloading data for {self.ticker}")

        from curl_cffi import requests

        session = requests.Session(impersonate="chrome")
        ticker_obj = yf.Ticker(self.ticker, session=session)

        df = ticker_obj.history(
            start=self.start_date,
            end=self.end_date,
            interval=self.interval_param,
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_cols = ["Close", "High", "Low", "Open", "Volume"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise KeyError(f"Missing expected columns in download: {missing}")

        df = df[required_cols].copy()
        df.reset_index(inplace=True)
        df.columns = ["datetime", "close", "high", "low", "open", "volume"]
        df["datetime"] = pd.to_datetime(df["datetime"], utc=True)
        df["datetime"] = df["datetime"].dt.tz_convert("Asia/Hong_Kong")
        df.set_index("datetime", inplace=True)
        self.data = df
        return self.data

    def save_to_csv(self, filepath_prefix):

        if self.data is None:
            raise ValueError("No data to save. Call download_data() first.")
        self.data.to_csv(f"{filepath_prefix}.csv")

    def get_backtrader_data(self):

        if self.data is None:
            raise ValueError("Data not downloaded yet. Call download_data() first.")
        
        df = self.data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        return bt.feeds.PandasData(dataname=df)

    def load_from_csv(self, filepath_prefix):

        df = pd.read_csv(f"{filepath_prefix}_{self.ticker}.csv", index_col="Date", parse_dates=True)
        df['Ticker'] = self.ticker
        self.data = df.sort_index()
        return self.data

    


if __name__ == "__main__":
    tickers = ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK.B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'BK', 'BA', 'BKNG', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CAT', 'CBOE', 'CBRE', 'CDW', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DVA', 'DAY', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DG', 'DLTR', 'D', 'DPZ', 'DASH', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXE', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TER', 'TSLA', 'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TKO', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WSM', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZTS']
    start_date = "2000-01-03"
    end_date = "2025-01-01"
    save_path = "data/tickers/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    valid_tickers = []

    for ticker in tickers:
        print(f"[INFO] Downloading data for {ticker}")
        handler = DataHandler(ticker, start_date, end_date)
        df = handler.download_data()

        if df.empty or df['date'].iloc[0] > pd.Timestamp(start_date):
            print(f"[WARNING] No data available for {ticker} from {start_date}")
            continue

        handler.save_to_csv(os.path.join(save_path, ticker))

        valid_tickers.append(ticker)

        print(f"[INFO] Saved data for {ticker} to {save_path}/{ticker}.csv")

    valid_tickers_df = pd.DataFrame(valid_tickers, columns=["Ticker"])
    valid_tickers_df.to_csv("data/valid_tickers.csv", index=False)
    print("[INFO] Saved valid tickers to data/valid_tickers.csv")
   