# src/labeled_design_matrix_builder.py
import os
import pandas as pd
from src.feature_label_builder import FeatureLabelBuilder
from src.data_handler import DataHandler

class LabeledDesignMatrixBuilder:
    def __init__(self, tickers_csv, labeler_class, feature_engineer_class, start_date="2000-01-03", end_date="2025-01-01"):
        """
        Initializes the LabeledDesignMatrixBuilder.

        Args:
            tickers_csv (str): Path to the CSV file containing the tickers.
            labeler_class (class): The class for the labeler (e.g., ForwardReturnLabeler).
            feature_engineer_class (class): The class for the feature engineer (e.g., IndicatorSignals).
            start_date (str): The start date for data download.
            end_date (str): The end date for data download.
        """
        self.tickers_csv = tickers_csv
        self.start_date = start_date
        self.end_date = end_date
        self.labeler_class = labeler_class
        self.feature_engineer_class = feature_engineer_class

    def build(self):
        """
        Loops through all tickers, generates features and labels, and combines them into a single design matrix.

        Returns:
            X (pd.DataFrame): Combined feature matrix for all tickers.
            y (pd.Series): Combined label vector for all tickers.
        """
        X_list = []
        y_list = []

        # Read valid tickers from the CSV
        valid_tickers = pd.read_csv(self.tickers_csv)["Ticker"].tolist()

        for ticker in valid_tickers:
            print(f"\n=== Processing {ticker} ===")
            handler = DataHandler(ticker, start_date=self.start_date, end_date=self.end_date)
            df = handler.download_data()

            if df is None or df.empty:
                print(f"[WARNING] No data for {ticker}. Skipping.")
                continue

            # Generate features and labels using FeatureLabelBuilder
            builder = FeatureLabelBuilder(self.feature_engineer_class, self.labeler_class)
            X, y = builder.build(df)

            # Append results to lists
            X_list.append(X)
            y_list.append(y)

        # Combine all the individual X and y into single matrices
        X_combined = pd.concat(X_list, axis=0)
        y_combined = pd.concat(y_list, axis=0)

        print(f"\nFeature matrix X shape: {X_combined.shape}")
        print(f"Label vector y shape: {y_combined.shape}")

        # Return combined data
        return X_combined, y_combined