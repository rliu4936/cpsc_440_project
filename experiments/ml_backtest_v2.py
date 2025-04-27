import backtrader as bt
import pandas as pd
import numpy as np
from DataHandler import DataHandler
from SignalGenerator import SignalGenerator
from IndicatorSignals import FeatureEngineer
from MetaModel import MetaModel
from datetime import datetime

class MLStrategy(bt.Strategy):
    def __init__(self, model, feature_data):
        self.model = model
        self.feature_data = feature_data
        self.dataclose = self.datas[0].close
        self.order = None
        self.portfolio_values = []
        self.start_trading_date = feature_data.index[int(0.7 * len(feature_data))]
        print(f"[DEBUG] Trading will start from {self.start_trading_date}")
        self.prediction_counts = {"long": 0, "none": 0}
        self.prediction_counter = 0  # for periodic printing

    def buy_all(self):
        cash = self.broker.getcash()
        price = self.dataclose[0]
        size = int(cash / price)
        if size > 0:
            self.order = self.buy(size=size)

    def next(self):
        if pd.Timestamp(self.datas[0].datetime.date(0)) < self.start_trading_date:
            return

        current_dt = pd.Timestamp(self.datas[0].datetime.date(0))

        if current_dt not in self.feature_data.index:
            return

        X_today = self.feature_data.loc[[current_dt]]
        prediction = self.model.predict(X_today)[0]
        print(f"{current_dt}: Prediction = {prediction}, Position Size = {self.position.size}")

        # Track prediction counts and live summary
        if prediction > 0:
            self.prediction_counts["long"] += 1
        else:
            self.prediction_counts["none"] += 1
        self.prediction_counter += 1

        if self.prediction_counter % 20 == 0:
            print(f"[LIVE SUMMARY] After {self.prediction_counter} predictions:")
            total_preds = sum(self.prediction_counts.values())
            for k, v in self.prediction_counts.items():
                pct = (v / total_preds) * 100 if total_preds > 0 else 0
                label = {"long": "Long (>0)", "none": "No Position (<=0)"}[k]
                print(f"  {label}: {v} ({pct:.2f}%)")
            print("-" * 30)

        if prediction > 0:
            if not self.position:
                self.buy_all()
        else:
            if self.position.size > 0:
                self.close()

        self.portfolio_values.append(self.broker.getvalue())

    def stop(self):
        total_preds = sum(self.prediction_counts.values())
        print("\n[FINAL SUMMARY] Prediction Counts During Trading:")
        for k, v in self.prediction_counts.items():
            pct = (v / total_preds) * 100 if total_preds > 0 else 0
            label = {"long": "Long (>0)", "none": "No Position (<=0)"}[k]
            print(f"  {label}: {v} ({pct:.2f}%)")

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
    signal_gen = SignalGenerator(price_data_single)
    signals = signal_gen.generate_ma_crossover_signal(short_window=10, long_window=30)

    # Feature engineering
    fe = FeatureEngineer(price_data_single)
    fe.compute_momentum_features()
    fe.compute_volatility_features()
    fe.compute_volume_features()
    features = fe.get_features()

    # Triple-barrier labeling using MA crossover signals
    from TripleBarrierLabeler import TripleBarrierLabeler
    # Create events DataFrame for triple-barrier labeling
    events = pd.DataFrame(index=signals.index)
    events['t1'] = signals.index.shift(5, freq='D')  # horizon of 5 days
    events['trgt'] = price_data_single['Close'].rolling(10).std()  # use rolling volatility as target
    events['side'] = signals['signal']  # direction of trade

    labeler = TripleBarrierLabeler(price_data_single['Close'])
    touches = labeler.apply_pt_sl_on_t1(events, pt_sl=(0.01, 0.01), molecule=events.index)
    labels = labeler.get_labels(touches)
    labels = labels.loc[features.index]  # align with feature index

    split_idx = int(0.7 * len(features))
    X_train, y_train = features.iloc[:split_idx], labels.iloc[:split_idx]
    X_test, y_test = features.iloc[split_idx:], labels.iloc[split_idx:]
    all_features = features  # Use all features for backtesting

    model = MetaModel(model_type='random_forest')
    model.train(X_train, y_train)

    # Evaluate model on test set
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_pred = model.model.predict(X_test)

    # Clip predictions to [-1, 1] just to be safe
    y_pred = np.clip(y_pred, -1, 1)

    # Regression evaluation metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nMean Squared Error (MSE): {mse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

    # Optional: show prediction value distribution
    pred_counts = pd.Series(np.round(y_pred, 2)).value_counts().sort_index()
    print("\nPrediction Counts (rounded to 2 decimals):")
    print(pred_counts)

    # Also show percentage breakdown
    pred_percentages = (pred_counts / len(y_pred)) * 100
    pred_summary = pd.DataFrame({
        'Count': pred_counts,
        'Percentage': pred_percentages.round(2)
    })
    print("\nPrediction Summary:")
    print(pred_summary)

    # Backtrader setup
    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=price_data_single)
    cerebro.adddata(data)
    cerebro.addstrategy(MLStrategy, model=model.model, feature_data=all_features)

    # Mark trading start date
    start_trading_date = features.index[int(0.7 * len(features))]
    print(f"Trading starts from: {start_trading_date}")

    cerebro.broker.set_cash(100000)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Portfolio statistics
    start_value = cerebro.broker.startingcash
    end_value = cerebro.broker.getvalue()
    total_return = (end_value - start_value) / start_value

    print(f"Total Return: {total_return * 100:.2f}%")

    # Estimate daily returns for Sharpe ratio
    # (Assume 252 trading days per year)
    returns = []
    strategy = cerebro.runstrats[0][0]
    portfolio_values = strategy.portfolio_values

    for i in range(1, len(portfolio_values)):
        prev = portfolio_values[i - 1]
        curr = portfolio_values[i]
        returns.append((curr - prev) / prev)

    if returns:
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        sharpe_ratio = (avg_return / std_return) * (252 ** 0.5)
        print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")
    else:
        print("Not enough returns data to compute Sharpe ratio.")

    cerebro.plot(style='candlestick')

if __name__ == "__main__":
    main()