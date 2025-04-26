import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands

class IndicatorSignals:
    def __init__(self, price_data):
        self.price_data = price_data
        self.signals = pd.DataFrame(index=price_data.index)
        self.feature_functions = [
            self.compute_momentum_features,
            self.compute_volatility_features,
            self.compute_volume_features,
            self.compute_additional_features,
        ]

    def compute_momentum_features(self):
        short_ma = SMAIndicator(self.price_data['Close'], window=10).sma_indicator()
        long_ma = SMAIndicator(self.price_data['Close'], window=50).sma_indicator()
        self.signals['ma_crossover'] = (short_ma > long_ma).astype(int)
    
    def compute_volatility_features(self):
        volatility_5d = self.price_data['Close'].pct_change().rolling(window=5).std()
        volatility_20d_avg = volatility_5d.rolling(window=20).mean()
        self.signals['volatility_signal'] = (volatility_5d < volatility_20d_avg).astype(int)

    def compute_volume_features(self):
        volume_5d_avg = self.price_data['Volume'].rolling(window=5).mean()
        self.signals['volume_signal'] = (self.price_data['Volume'] > volume_5d_avg).astype(int)

    def compute_additional_features(self):
        rsi_14 = RSIIndicator(self.price_data['Close'], window=14).rsi()
        self.signals['rsi_signal'] = (rsi_14 < 30).astype(int)

        macd = MACD(self.price_data['Close'])
        self.signals['macd_signal'] = (macd.macd() > macd.macd_signal()).astype(int)

        bbands = BollingerBands(self.price_data['Close'], window=20, window_dev=2)
        self.signals['bollinger_signal'] = (self.price_data['Close'] < bbands.bollinger_lband()).astype(int)

    def get_signals(self):
        for func in self.feature_functions:
            func()
        return self.signals.dropna()
    def generate_ma_cross_signal(self, short_window=10, long_window=50):
        short_ma = SMAIndicator(self.price_data['Close'], window=short_window).sma_indicator()
        long_ma = SMAIndicator(self.price_data['Close'], window=long_window).sma_indicator()
        return (short_ma > long_ma).astype(int)

    def generate_rsi_signal(self, rsi_length=14):
        rsi = RSIIndicator(self.price_data['Close'], window=rsi_length).rsi()
        return (rsi < 30).astype(int)

    def generate_macd_signal(self):
        macd = MACD(self.price_data['Close'])
        return (macd.macd() > macd.macd_signal()).astype(int)

    def generate_bollinger_bands_signal(self):
        bb = BollingerBands(self.price_data['Close'], window=20, window_dev=2)
        return (self.price_data['Close'] < bb.bollinger_lband()).astype(int)

    def generate_adx_signal(self):
        from ta.trend import ADXIndicator
        adx = ADXIndicator(self.price_data['High'], self.price_data['Low'], self.price_data['Close'], window=14)
        return (adx.adx() > 25).astype(int)

    def generate_position_from_signal(self, signal):
        position = signal.copy()
        position = position.shift(1).fillna(0)
        return position