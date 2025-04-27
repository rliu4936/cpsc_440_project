import backtrader as bt
import pandas as pd
import numpy as np

class BacktestRunner:
    def __init__(self, price_data, signal_features, initial_cash=100000):
        self.price_data = price_data
        self.signal_features = signal_features
        self.initial_cash = initial_cash
        self.strategy = None
        self.result_cerebro = None
        self.result_strategy = None

    def run(self, short_window, long_window):
        cerebro = bt.Cerebro()
        data = bt.feeds.PandasData(dataname=self.price_data)
        cerebro.adddata(data)

        cerebro.addstrategy(self._build_strategy_class(short_window, long_window), feature_data=self.signal_features)
        cerebro.broker.set_cash(self.initial_cash)

        strategies = cerebro.run(runonce=True)
        self.result_strategy = strategies[0]
        self.result_cerebro = cerebro

    def backtest(self, short_window, long_window):
        self.run(short_window, long_window)

    def _build_strategy_class(self, short_window, long_window):
        features = self.signal_features

        class DynamicMAStrategy(bt.Strategy):
            def __init__(self):
                self.feature_data = features  # Accept feature_data argument
                self.dataclose = self.datas[0].close
                self.portfolio_values = []
                self.trade_log = []

            def next(self):
                current_dt = pd.Timestamp(self.datas[0].datetime.date(0))
                target_position = self.feature_data.get(current_dt, 0)
                cash = self.broker.getcash()
                price = self.dataclose[0]

                if target_position == 1 and self.position.size <= 0:
                    size = int(cash / price)
                    if size > 0:
                        self.buy(size=size)
                        self.trade_log.append({"date": current_dt, "action": "BUY", "price": price, "size": size})
                elif target_position == 0 and self.position.size > 0:
                    self.close()
                    self.trade_log.append({"date": current_dt, "action": "SELL", "price": price, "size": self.position.size})

                self.portfolio_values.append(self.broker.getvalue())

        return DynamicMAStrategy

    def get_equity_curve(self):
        return pd.DataFrame({"Portfolio Value": self.result_strategy.portfolio_values},
                            index=self.price_data.index[:len(self.result_strategy.portfolio_values)])

    def get_trade_log(self):
        return pd.DataFrame(self.result_strategy.trade_log)

    def get_total_return(self):
        return (self.result_strategy.broker.getvalue() - self.initial_cash) / self.initial_cash

    def get_sharpe_ratio(self):
        equity_curve = self.get_equity_curve()
        daily_returns = equity_curve["Portfolio Value"].pct_change().dropna()
        return (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
