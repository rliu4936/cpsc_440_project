import os
import pandas as pd
from src.feature_label_builder import FeatureLabelBuilder
from src.data_handler import DataHandler
from src.indicator_signals import IndicatorSignals

class LabeledDesignMatrixBuilder:
    def __init__(self, tickers, labeler, start_date="2000-01-03", end_date="2025-01-01"):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.labeler = labeler

    def build(self):
        X_list = []
        y_list = []

        for ticker in self.tickers:
            print(f"\n=== Processing {ticker} ===")
            handler = DataHandler(ticker, start_date=self.start_date, end_date=self.end_date)
            df = handler.download_data()

            if df is None or df.empty:
                print(f"[WARNING] No data for {ticker}. Skipping.")
                continue

            builder = FeatureLabelBuilder(IndicatorSignals, self.labeler)
            X, y = builder.build(df)

            X_list.append(X)
            y_list.append(y)

        X_combined = pd.concat(X_list, axis=0)
        y_combined = pd.concat(y_list, axis=0)

        print(f"\nFeature matrix X shape: {X_combined.shape}")
        print(f"Label vector y shape: {y_combined.shape}")

        return X_combined, y_combined