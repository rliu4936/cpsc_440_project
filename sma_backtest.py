import backtrader as bt
import pandas as pd
import numpy as np
from DataHandler import DataHandler
from IndicatorSignals import IndicatorSignals
from datetime import datetime
import seaborn as sns

class MAStrategy(bt.Strategy):
    def __init__(self, feature_data):
        self.feature_data = feature_data
        self.dataclose = self.datas[0].close
        self.order = None
        self.portfolio_values = []
        self.trade_log = []

    def buy_all(self):
        cash = self.broker.getcash()
        price = self.dataclose[0]
        size = int(cash / price)
        if size > 0:
            self.order = self.buy(size=size)

    def next(self):
        current_dt = pd.Timestamp(self.datas[0].datetime.date(0))

        target_position = self.feature_data.get(current_dt, 0)

        if target_position == 1:
            if self.position.size <= 0:
                cash = self.broker.getcash()
                size = int(cash / self.dataclose[0])
                if size > 0:
                    self.order = self.buy(size=size)
                    print(f"[{current_dt.date()}] BUY {size} @ {self.dataclose[0]:.2f}")
                    self.trade_log.append({"date": current_dt, "action": "BUY", "price": self.dataclose[0], "size": size})
        elif target_position == 0:
            if self.position.size > 0:
                self.order = self.close()
                print(f"[{current_dt.date()}] SELL @ {self.dataclose[0]:.2f}")
                self.trade_log.append({"date": current_dt, "action": "SELL", "price": self.dataclose[0], "size": self.position.size})
        else:
            print(f"[{current_dt.date()}] HOLD")

        self.portfolio_values.append(self.broker.getvalue())

def main():
    # Load data
    tickers = ["QQQ", "SPY", "AAPL", "MSFT"]
    backtest_ticker = "QQQ"
    handler = DataHandler(tickers, start_date="2000-01-01", end_date=datetime.today().strftime("%Y-%m-%d"))
    price_data = handler.download_data()

    print("[DEBUG] price_data.head():")
    print(price_data.head())
    print("[DEBUG] price_data.columns:")
    print(price_data.columns)

    if 'Ticker' in price_data.columns:
        price_data_single = price_data[price_data['Ticker'] == backtest_ticker].drop(columns='Ticker').dropna()
    else:
        raise ValueError(f"[ERROR] Cannot find 'Ticker' column in price_data.columns: {price_data.columns}")

    print("[DEBUG] Selected price_data_single.head():")
    print(price_data_single.head())

    results = []

    best_total_return = float('-inf')
    best_short_window = None
    best_long_window = None
    best_strat = None
    best_equity_df = None
    best_cerebro = None

    for short_window in range(5, 61, 1):  # Try 5,6,7,...,60
        for long_window in range(20, 201, 5):  # Try 20,25,30,...,200
            if short_window >= long_window:
                continue

            signals = IndicatorSignals(price_data_single)
            position_features = signals.generate_ma_cross_signal(short_window=short_window, long_window=long_window)

            cerebro = bt.Cerebro()
            data = bt.feeds.PandasData(dataname=price_data_single)
            cerebro.adddata(data)
            cerebro.addstrategy(MAStrategy, feature_data=position_features)
            cerebro.broker.set_cash(100000)

            strat = cerebro.run(runonce=True)[0]

            start_value = 100000
            end_value = strat.broker.getvalue()
            total_return = (end_value - start_value) / start_value

            results.append((short_window, long_window, total_return))

            if total_return > best_total_return:
                best_total_return = total_return
                best_short_window = short_window
                best_long_window = long_window
                best_strat = strat
                best_cerebro = cerebro  # <-- Save the best cerebro
                best_equity_df = pd.DataFrame({'Portfolio Value': strat.portfolio_values}, index=price_data_single.index[:len(strat.portfolio_values)])

    print(f"\nBest short_window={best_short_window}, long_window={best_long_window} with return {best_total_return * 100:.2f}%")
    daily_returns = best_equity_df['Portfolio Value'].pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    print(f"Best Parameters Summary:")
    print(f"Short Window: {best_short_window}")
    print(f"Long Window: {best_long_window}")
    print(f"Total Return: {best_total_return * 100:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")

    # Plot equity curve
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(best_equity_df.index, best_equity_df['Portfolio Value'])
    plt.title('Equity Curve (Best Parameters)')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.show()

    # Plot strategy vs QQQ
    plt.figure(figsize=(12, 6))
    plt.plot(best_equity_df.index, best_equity_df['Portfolio Value'] / best_equity_df['Portfolio Value'].iloc[0], label="Strategy (Best Params)")
    plt.plot(price_data_single.index[:len(best_equity_df)], price_data_single['Close'].iloc[:len(best_equity_df)] / price_data_single['Close'].iloc[0], label="QQQ Buy & Hold")
    plt.title('Strategy vs QQQ Buy & Hold')
    plt.xlabel('Date')
    plt.ylabel('Normalized Value')
    plt.legend()
    plt.grid(True)
    plt.show()

    if best_cerebro is not None:
        best_cerebro.plot()

    # Visualize hyperparameter search using already computed results
    results_df = pd.DataFrame(results, columns=['Short Window', 'Long Window', 'Return'])

    pivot_table = results_df.pivot(index="Short Window", columns="Long Window", values="Return")

    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title('Return Heatmap: Short Window vs Long Window')
    plt.xlabel('Long Window')
    plt.ylabel('Short Window')
    plt.show()

if __name__ == "__main__":
    main()