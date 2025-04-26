import pandas as pd
import numpy as np

class SignalGenerator:
    def __init__(self, price_data):
        self.price_data = price_data
        self.signals = pd.DataFrame(index=price_data.index)

    def generate_ma_crossover_signal(self, short_window=10, long_window=30):
        short_ma = self.price_data['Close'].rolling(window=short_window).mean()
        long_ma = self.price_data['Close'].rolling(window=long_window).mean()
        self.signals['signal'] = 0
        self.signals.iloc[short_window:, self.signals.columns.get_loc('signal')] = np.where(
            short_ma.iloc[short_window:] > long_ma.iloc[short_window:], 1, -1
        )
        return self.signals

    def get_signals(self):
        return self.signals
    def generate_rsi_signal(self, period=14, overbought=70, oversold=30):
        delta = self.price_data['Close'].diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(window=period).mean()
        avg_loss = pd.Series(loss).rolling(window=period).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        self.signals['rsi_signal'] = 0
        self.signals.loc[rsi.index[rsi < oversold], 'rsi_signal'] = 1
        self.signals.loc[rsi.index[rsi > overbought], 'rsi_signal'] = -1
        return self.signals