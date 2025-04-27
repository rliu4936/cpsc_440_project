import backtrader as bt
import pandas as pd
import numpy as np
from DataHandler import DataHandler
from IndicatorSignals import IndicatorSignals
from datetime import datetime

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

    # DEBUG: Show structure of price_data
    print("[DEBUG] price_data.head():")
    print(price_data.head())
    print("[DEBUG] price_data.columns:")
    print(price_data.columns)

    # Select one ticker for backtesting
    if 'Ticker' in price_data.columns:
        price_data_single = price_data[price_data['Ticker'] == backtest_ticker].drop(columns='Ticker').dropna()
    else:
        raise ValueError(f"[ERROR] Cannot find 'Ticker' column in price_data.columns: {price_data.columns}")

    print("[DEBUG] Selected price_data_single.head():")
    print(price_data_single.head())

    # Generate signals (can be dummy for training)
    signals = IndicatorSignals(price_data_single)
    position_features = signals.generate_ema_cross_signal(short_window=10, long_window=30)

    # Backtrader setup
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=price_data_single)
    cerebro.adddata(data)
    cerebro.addstrategy(MAStrategy, feature_data=position_features)


    cerebro.broker.set_cash(100000)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Portfolio statistics
    start_value = cerebro.broker.startingcash
    end_value = cerebro.broker.getvalue()
    total_return = (end_value - start_value) / start_value

    print(f"Total Return: {total_return * 100:.2f}%")

    strat = cerebro.runstrats[0][0]  # Get the first strategy
    trade_log_df = pd.DataFrame(strat.trade_log)
    print("\n[TRADE LOG]")
    print(trade_log_df)

    # Optionally save to CSV
    trade_log_df.to_csv("trade_log.csv", index=False)
    print("\nTrade log saved to trade_log.csv")

    cerebro.plot(style='candlestick')

if __name__ == "__main__":
    main()