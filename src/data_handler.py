import os
import yfinance as yf
import pandas as pd
import backtrader as bt
from datetime import datetime


class DataHandler:
    """
    A class responsible for downloading, processing, and managing financial market data for backtesting.
    Now simplified to work with a single ticker at a time.
    """

    def __init__(self, ticker,  start_date = "2000-01-03", end_date = "2025-01-01"):
        """
        Initializes the DataHandler for a single ticker.
        
        Args:
        ticker : str
            The ticker symbol for the stock to download.
        start_date : str
            The start date for downloading the data (format: "YYYY-MM-DD").
        end_date : str
            The end date for downloading the data (format: "YYYY-MM-DD").
        """

        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    def download_data(self):
        """
        Downloads financial data for the single ticker from Yahoo Finance
        within the specified date range and returns a dataframe with
        exactly the columns: date, close, high, low, open, volume.
        If the local CSV file exists, load from it instead.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame with columns
            ['date', 'close', 'high', 'low', 'open', 'volume'].
        """
        local_path = f"data/tickers/{self.ticker}.csv"

        if os.path.exists(local_path):
            print(f"[INFO] Loading cached data for {self.ticker}")
            df = pd.read_csv(local_path, parse_dates=["date"])
            self.data = df
            return self.data

        print(f"[INFO] Downloading data for {self.ticker}")
        df = yf.download(
            self.ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False,
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        required_cols = ["Close", "High", "Low", "Open", "Volume"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise KeyError(f"Missing expected columns in download: {missing}")

        df = df[required_cols].copy()
        df.reset_index(inplace=True)
        df.columns = ["date", "close", "high", "low", "open", "volume"]

        self.data = df
        return self.data

    def save_to_csv(self, filepath_prefix):
        """
        Saves the downloaded data to a CSV file for the single ticker.
        
        Args:
        filepath_prefix : str
            The prefix for the CSV file (e.g., "data/ticker").
        """
        if self.data is None:
            raise ValueError("No data to save. Call download_data() first.")
        self.data.to_csv(f"{filepath_prefix}.csv")

    def get_backtrader_data(self):
        """
        Prepares the data for usage in Backtrader by converting it into the appropriate format.
        
        Returns:
        bt.feeds.PandasData
            A Backtrader feed for the single ticker.
        """
        if self.data is None:
            raise ValueError("Data not downloaded yet. Call download_data() first.")
        
        # Ensure 'date' is datetime and set as index
        df = self.data.copy()
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        return bt.feeds.PandasData(dataname=df)

    def load_from_csv(self, filepath_prefix):
        """
        Loads the data from a CSV file for the single ticker.
        
        Args:
        filepath_prefix : str
            The prefix for the CSV file (e.g., "data/ticker").
        
        Returns:
        pd.DataFrame
            The loaded data for the single ticker.
        """
        df = pd.read_csv(f"{filepath_prefix}_{self.ticker}.csv", index_col="Date", parse_dates=True)
        df['Ticker'] = self.ticker
        self.data = df.sort_index()
        return self.data


if __name__ == "__main__":
    # List of tickers (can be from a different source, like the S&P 500 list)
    tickers = ['MMM', 'AOS', 'ABT', 'ABBV', 'ACN', 'ADBE', 'AMD', 'AES', 'AFL', 'A', 'APD', 'ABNB', 'AKAM', 'ALB', 'ARE', 'ALGN', 'ALLE', 'LNT', 'ALL', 'GOOGL', 'GOOG', 'MO', 'AMZN', 'AMCR', 'AEE', 'AEP', 'AXP', 'AIG', 'AMT', 'AWK', 'AMP', 'AME', 'AMGN', 'APH', 'ADI', 'ANSS', 'AON', 'APA', 'APO', 'AAPL', 'AMAT', 'APTV', 'ACGL', 'ADM', 'ANET', 'AJG', 'AIZ', 'T', 'ATO', 'ADSK', 'ADP', 'AZO', 'AVB', 'AVY', 'AXON', 'BKR', 'BALL', 'BAC', 'BAX', 'BDX', 'BRK.B', 'BBY', 'TECH', 'BIIB', 'BLK', 'BX', 'BK', 'BA', 'BKNG', 'BSX', 'BMY', 'AVGO', 'BR', 'BRO', 'BF.B', 'BLDR', 'BG', 'BXP', 'CHRW', 'CDNS', 'CZR', 'CPT', 'CPB', 'COF', 'CAH', 'KMX', 'CCL', 'CARR', 'CAT', 'CBOE', 'CBRE', 'CDW', 'COR', 'CNC', 'CNP', 'CF', 'CRL', 'SCHW', 'CHTR', 'CVX', 'CMG', 'CB', 'CHD', 'CI', 'CINF', 'CTAS', 'CSCO', 'C', 'CFG', 'CLX', 'CME', 'CMS', 'KO', 'CTSH', 'CL', 'CMCSA', 'CAG', 'COP', 'ED', 'STZ', 'CEG', 'COO', 'CPRT', 'GLW', 'CPAY', 'CTVA', 'CSGP', 'COST', 'CTRA', 'CRWD', 'CCI', 'CSX', 'CMI', 'CVS', 'DHR', 'DRI', 'DVA', 'DAY', 'DECK', 'DE', 'DELL', 'DAL', 'DVN', 'DXCM', 'FANG', 'DLR', 'DFS', 'DG', 'DLTR', 'D', 'DPZ', 'DASH', 'DOV', 'DOW', 'DHI', 'DTE', 'DUK', 'DD', 'EMN', 'ETN', 'EBAY', 'ECL', 'EIX', 'EW', 'EA', 'ELV', 'EMR', 'ENPH', 'ETR', 'EOG', 'EPAM', 'EQT', 'EFX', 'EQIX', 'EQR', 'ERIE', 'ESS', 'EL', 'EG', 'EVRG', 'ES', 'EXC', 'EXE', 'EXPE', 'EXPD', 'EXR', 'XOM', 'FFIV', 'FDS', 'FICO', 'FAST', 'FRT', 'FDX', 'FIS', 'FITB', 'FSLR', 'FE', 'FI', 'F', 'FTNT', 'FTV', 'FOXA', 'FOX', 'BEN', 'FCX', 'GRMN', 'IT', 'GE', 'GEHC', 'GEV', 'GEN', 'GNRC', 'GD', 'GIS', 'GM', 'GPC', 'GILD', 'GPN', 'GL', 'GDDY', 'GS', 'HAL', 'HIG', 'HAS', 'HCA', 'DOC', 'HSIC', 'HSY', 'HES', 'HPE', 'HLT', 'HOLX', 'HD', 'HON', 'HRL', 'HST', 'HWM', 'HPQ', 'HUBB', 'HUM', 'HBAN', 'HII', 'IBM', 'IEX', 'IDXX', 'ITW', 'INCY', 'IR', 'PODD', 'INTC', 'ICE', 'IFF', 'IP', 'IPG', 'INTU', 'ISRG', 'IVZ', 'INVH', 'IQV', 'IRM', 'JBHT', 'JBL', 'JKHY', 'J', 'JNJ', 'JCI', 'JPM', 'JNPR', 'K', 'KVUE', 'KDP', 'KEY', 'KEYS', 'KMB', 'KIM', 'KMI', 'KKR', 'KLAC', 'KHC', 'KR', 'LHX', 'LH', 'LRCX', 'LW', 'LVS', 'LDOS', 'LEN', 'LII', 'LLY', 'LIN', 'LYV', 'LKQ', 'LMT', 'L', 'LOW', 'LULU', 'LYB', 'MTB', 'MPC', 'MKTX', 'MAR', 'MMC', 'MLM', 'MAS', 'MA', 'MTCH', 'MKC', 'MCD', 'MCK', 'MDT', 'MRK', 'META', 'MET', 'MTD', 'MGM', 'MCHP', 'MU', 'MSFT', 'MAA', 'MRNA', 'MHK', 'MOH', 'TAP', 'MDLZ', 'MPWR', 'MNST', 'MCO', 'MS', 'MOS', 'MSI', 'MSCI', 'NDAQ', 'NTAP', 'NFLX', 'NEM', 'NWSA', 'NWS', 'NEE', 'NKE', 'NI', 'NDSN', 'NSC', 'NTRS', 'NOC', 'NCLH', 'NRG', 'NUE', 'NVDA', 'NVR', 'NXPI', 'ORLY', 'OXY', 'ODFL', 'OMC', 'ON', 'OKE', 'ORCL', 'OTIS', 'PCAR', 'PKG', 'PLTR', 'PANW', 'PARA', 'PH', 'PAYX', 'PAYC', 'PYPL', 'PNR', 'PEP', 'PFE', 'PCG', 'PM', 'PSX', 'PNW', 'PNC', 'POOL', 'PPG', 'PPL', 'PFG', 'PG', 'PGR', 'PLD', 'PRU', 'PEG', 'PTC', 'PSA', 'PHM', 'PWR', 'QCOM', 'DGX', 'RL', 'RJF', 'RTX', 'O', 'REG', 'REGN', 'RF', 'RSG', 'RMD', 'RVTY', 'ROK', 'ROL', 'ROP', 'ROST', 'RCL', 'SPGI', 'CRM', 'SBAC', 'SLB', 'STX', 'SRE', 'NOW', 'SHW', 'SPG', 'SWKS', 'SJM', 'SW', 'SNA', 'SOLV', 'SO', 'LUV', 'SWK', 'SBUX', 'STT', 'STLD', 'STE', 'SYK', 'SMCI', 'SYF', 'SNPS', 'SYY', 'TMUS', 'TROW', 'TTWO', 'TPR', 'TRGP', 'TGT', 'TEL', 'TDY', 'TER', 'TSLA', 'TXN', 'TPL', 'TXT', 'TMO', 'TJX', 'TKO', 'TSCO', 'TT', 'TDG', 'TRV', 'TRMB', 'TFC', 'TYL', 'TSN', 'USB', 'UBER', 'UDR', 'ULTA', 'UNP', 'UAL', 'UPS', 'URI', 'UNH', 'UHS', 'VLO', 'VTR', 'VLTO', 'VRSN', 'VRSK', 'VZ', 'VRTX', 'VTRS', 'VICI', 'V', 'VST', 'VMC', 'WRB', 'GWW', 'WAB', 'WBA', 'WMT', 'DIS', 'WBD', 'WM', 'WAT', 'WEC', 'WFC', 'WELL', 'WST', 'WDC', 'WY', 'WSM', 'WMB', 'WTW', 'WDAY', 'WYNN', 'XEL', 'XYL', 'YUM', 'ZBRA', 'ZBH', 'ZTS']  # example tickers
    start_date = "2000-01-03"
    end_date = "2025-01-01"
    save_path = "data/tickers/"  # directory to save the CSVs

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    valid_tickers = []  # List to store valid tickers

    for ticker in tickers:
        print(f"[INFO] Downloading data for {ticker}")
        handler = DataHandler(ticker, start_date, end_date)
        df = handler.download_data()

        # Check if the data has entries starting from 2000-01-01
        if df.empty or df['date'].iloc[0] > pd.Timestamp(start_date):
            print(f"[WARNING] No data available for {ticker} from {start_date}")
            continue

        # Save with only the ticker (removing the duplicate ticker in the filename)
        handler.save_to_csv(os.path.join(save_path, ticker))  # Only use the ticker as the filename

        # Add the ticker to the valid tickers list
        valid_tickers.append(ticker)

        print(f"[INFO] Saved data for {ticker} to {save_path}/{ticker}.csv")

    # Save valid tickers to a CSV file
    valid_tickers_df = pd.DataFrame(valid_tickers, columns=["Ticker"])
    valid_tickers_df.to_csv("data/valid_tickers.csv", index=False)
    print("[INFO] Saved valid tickers to data/valid_tickers.csv")