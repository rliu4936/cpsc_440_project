import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
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
        short_ma = SMAIndicator(self.price_data['close'], window=10).sma_indicator()
        long_ma = SMAIndicator(self.price_data['close'], window=50).sma_indicator()
        ma_crossover = (short_ma > long_ma).astype(int)
        self.signals['ma_crossover'] = ma_crossover.fillna(0)

        # 200-day SMA trend
        sma_200 = SMAIndicator(self.price_data['close'], window=200).sma_indicator()
        self.signals['sma_200_trend'] = (self.price_data['close'] > sma_200).astype(int)

        # EMA crossover signal
        short_ema = EMAIndicator(self.price_data['close'], window=10).ema_indicator()
        long_ema = EMAIndicator(self.price_data['close'], window=50).ema_indicator()
        ema_cross_signal = (short_ema > long_ema).astype(int)
        self.signals['ema_cross_signal'] = ema_cross_signal.fillna(0)

        # Price above/below EMA20
        ema_20 = EMAIndicator(self.price_data['close'], window=20).ema_indicator()
        self.signals['price_above_ema20'] = (self.price_data['close'] > ema_20).astype(int)
    
    def compute_volatility_features(self):
        volatility_5d = self.price_data['close'].pct_change().rolling(window=5).std()
        volatility_20d_avg = volatility_5d.rolling(window=20).mean()
        self.signals['volatility_signal'] = (volatility_5d < volatility_20d_avg).astype(int)

    def compute_volume_features(self):
        volume_5d_avg = self.price_data['volume'].rolling(window=5).mean()
        self.signals['volume_signal'] = (self.price_data['volume'] > volume_5d_avg).astype(int)

    def compute_additional_features(self):
        # --- Tuned RSI Signals ---
        rsi_14 = RSIIndicator(self.price_data['close'], window=14).rsi()
        self.signals['rsi_signal'] = (rsi_14 < 40).astype(int).fillna(0)
        self.signals['rsi_overbought_signal'] = (rsi_14 < 60).astype(int)

        # --- MACD Signals ---
        macd = MACD(self.price_data['close'])
        self.signals['macd_signal'] = (macd.macd() > macd.macd_signal()).astype(int).fillna(0)
        self.signals['macd_positive_trend'] = (macd.macd() > 0).astype(int)

        # --- Tuned Bollinger Bands Signal ---
        bbands = BollingerBands(self.price_data['close'], window=20, window_dev=2)
        lband_adjusted = bbands.bollinger_lband() - 0.5 * (bbands.bollinger_hband() - bbands.bollinger_lband())

        # --- ADX Signal ---
        adx = ADXIndicator(self.price_data['high'], self.price_data['low'], self.price_data['close'], window=14)
        self.signals['adx_signal'] = (adx.adx() > 25).astype(int).fillna(0)

        # --- NEW: Stochastic Oscillator Signals ---
        from ta.momentum import StochasticOscillator
        stoch = StochasticOscillator(self.price_data['high'], self.price_data['low'], self.price_data['close'])
        self.signals['stochastic_oversold'] = (stoch.stoch() < 20).astype(int).fillna(0)
        self.signals['stochastic_overbought'] = (stoch.stoch() > 80).astype(int).fillna(0)

        # --- NEW: ATR Volatility Spike Signal ---
        from ta.volatility import AverageTrueRange
        atr = AverageTrueRange(self.price_data['high'], self.price_data['low'], self.price_data['close']).average_true_range()
        self.signals['atr_volatility_spike'] = ((atr / self.price_data['close']) > 0.02).astype(int).fillna(0)

        # --- NEW: EMA20 Slope Positive Signal ---
        ema20 = EMAIndicator(self.price_data['close'], window=20).ema_indicator()
        ema20_slope = ema20.diff()
        self.signals['ema20_slope_positive'] = (ema20_slope > 0).astype(int).fillna(0)

        # --- NEW: Bollinger Band Width Wide Signal ---
        bandwidth = bbands.bollinger_hband() - bbands.bollinger_lband()
        median_bandwidth = bandwidth.rolling(window=100).median()
        self.signals['bollinger_band_width_wide'] = (bandwidth > median_bandwidth).astype(int).fillna(0)

        # --- Additional Moving Average Crossovers ---
        sma_5 = SMAIndicator(self.price_data['close'], window=5).sma_indicator()
        sma_20 = SMAIndicator(self.price_data['close'], window=20).sma_indicator()
        sma_100 = SMAIndicator(self.price_data['close'], window=100).sma_indicator()
        self.signals['ma_crossover_5_20'] = (sma_5 > sma_20).astype(int).fillna(0)
        self.signals['ma_crossover_20_100'] = (sma_20 > sma_100).astype(int).fillna(0)

        # --- EMA Crossovers ---
        ema_8 = EMAIndicator(self.price_data['close'], window=8).ema_indicator()
        ema_21 = EMAIndicator(self.price_data['close'], window=21).ema_indicator()
        ema_12 = EMAIndicator(self.price_data['close'], window=12).ema_indicator()
        ema_26 = EMAIndicator(self.price_data['close'], window=26).ema_indicator()
        self.signals['ema_crossover_8_21'] = (ema_8 > ema_21).astype(int).fillna(0)
        self.signals['ema_crossover_12_26'] = (ema_12 > ema_26).astype(int).fillna(0)

        # --- Price above EMA50 and EMA100 ---
        ema_50 = EMAIndicator(self.price_data['close'], window=50).ema_indicator()
        ema_100 = EMAIndicator(self.price_data['close'], window=100).ema_indicator()
        self.signals['price_above_ema50'] = (self.price_data['close'] > ema_50).astype(int).fillna(0)
        self.signals['price_above_ema100'] = (self.price_data['close'] > ema_100).astype(int).fillna(0)

        # --- EMA50 Slope Positive ---
        ema50_slope = ema_50.diff()
        self.signals['ema50_slope_positive'] = (ema50_slope > 0).astype(int).fillna(0)

        # --- RSI Variants ---
        self.signals['rsi_oversold'] = (rsi_14 < 30).astype(int).fillna(0)
        self.signals['rsi_overbought'] = (rsi_14 > 70).astype(int).fillna(0)

        # --- Stochastic %K < 10 ---
        self.signals['stochastic_percentk_below_10'] = (stoch.stoch() < 10).astype(int).fillna(0)

        # --- MACD Histogram Positive ---
        macd_hist = macd.macd_diff()
        self.signals['macd_histogram_positive'] = (macd_hist > 0).astype(int).fillna(0)

        # --- ATR Volatility Spike at 3% ---
        self.signals['atr_volatility_spike_3pct'] = ((atr / self.price_data['close']) > 0.03).astype(int).fillna(0)

        # --- Bollinger Close Above Upper Band ---
        self.signals['bollinger_close_above_upper'] = (self.price_data['close'] > bbands.bollinger_hband()).astype(int).fillna(0)

        # --- Bandwidth Top 10% Wide ---
        bandwidth_rank = bandwidth.rolling(window=100).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        self.signals['bollinger_bandwidth_top_10pct'] = (bandwidth_rank > 0.9).astype(int).fillna(0)

        # --- Volume 10-day and 20-day Spikes ---
        volume_10d_avg = self.price_data['volume'].rolling(window=10).mean()
        volume_20d_avg = self.price_data['volume'].rolling(window=20).mean()
        self.signals['volume_spike_10d'] = (self.price_data['volume'] > volume_10d_avg).astype(int).fillna(0)
        self.signals['volume_spike_20d'] = (self.price_data['volume'] > volume_20d_avg).astype(int).fillna(0)

        # --- Return-based Signals (1-day and 5-day up/down) ---
        ret_1d = self.price_data['close'].pct_change(1)
        ret_5d = self.price_data['close'].pct_change(5)
        self.signals['return_1d_up'] = (ret_1d > 0).astype(int).fillna(0)
        self.signals['return_1d_down'] = (ret_1d < 0).astype(int).fillna(0)
        self.signals['return_5d_up'] = (ret_5d > 0).astype(int).fillna(0)
        self.signals['return_5d_down'] = (ret_5d < 0).astype(int).fillna(0)

        # --- ADX > 30 and ADX > 40 ---
        self.signals['adx_above_30'] = (adx.adx() > 30).astype(int).fillna(0)
        self.signals['adx_above_40'] = (adx.adx() > 40).astype(int).fillna(0)

        # --- Price at 20-day and 50-day Highs/Lows ---
        high_20d = self.price_data['high'].rolling(window=20).max()
        low_20d = self.price_data['low'].rolling(window=20).min()
        high_50d = self.price_data['high'].rolling(window=50).max()
        low_50d = self.price_data['low'].rolling(window=50).min()

        # --- Close Above Yesterday High, Close Below Yesterday Low ---
        yesterday_high = self.price_data['high'].shift(1)
        yesterday_low = self.price_data['low'].shift(1)
        self.signals['close_above_yesterday_high'] = (self.price_data['close'] > yesterday_high).astype(int).fillna(0)
        self.signals['close_below_yesterday_low'] = (self.price_data['close'] < yesterday_low).astype(int).fillna(0)

    def get_signals(self, dropna: bool = True):
        """
        Generate all configured signals and return them.

        Parameters
        ----------
        dropna : bool, default True
            Remove rows consisting only of NaNs.

        Returns
        -------
        pd.DataFrame
            The DataFrame of generated signals.
        """
        for func in self.feature_functions:
            func()
        return self.signals.dropna(how="all") if dropna else self.signals