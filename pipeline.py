from DataHandler import DataHandler
from SignalGenerator import SignalGenerator
from TripleBarrierLabeler import TripleBarrierLabeler
from IndicatorSignals import FeatureEngineer
from MetaModel import MetaModel
import pandas as pd
from datetime import datetime

def main():
    # Step 1: Get price data
    handler = DataHandler("QQQ", start_date="2020-01-01", end_date=datetime.today().strftime("%Y-%m-%d"))
    price_data = handler.download_data()

    # Step 2: Generate trading signals
    signal_gen = SignalGenerator(price_data)
    signals = signal_gen.generate_ma_crossover_signal(short_window=10, long_window=30)

    # Step 3: Label signals using Triple Barrier
    events = pd.DataFrame(index=signals.index)
    events['t1'] = signals.index.shift(5, freq='D')
    events['trgt'] = price_data['Close'].rolling(10).std()
    events['side'] = signals['signal']

    labeler = TripleBarrierLabeler(price_data['Close'])
    touches = labeler.apply_pt_sl_on_t1(events, pt_sl=(0.02, 0.02), molecule=events.index)
    labels = labeler.get_labels(touches)

    # Step 4: Engineer features
    fe = FeatureEngineer(price_data)
    fe.compute_momentum_features()
    fe.compute_volatility_features()
    fe.compute_volume_features()
    features = fe.get_features()

    # Align features and labels safely
    common_idx = features.index.intersection(labels.index)
    X = features.loc[common_idx]
    y = labels.loc[common_idx]

    # Step 5: Train and evaluate model
    split_idx = int(0.7 * len(X))
    X_train, y_train = X.iloc[:split_idx], y.iloc[:split_idx]
    X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]

    model = MetaModel(model_type='random_forest')
    model.train(X_train, y_train)

    report = model.evaluate(X_test, y_test)
    print("Classification Report:\n", report)

if __name__ == "__main__":
    main()