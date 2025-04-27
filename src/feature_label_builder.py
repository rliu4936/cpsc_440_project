# src/feature_label_builder.py
import os
import pandas as pd

class FeatureLabelBuilder:
    def __init__(self, feature_engineer_class, labeler):
        """
        Args:
            feature_engineer_class: A class that generates features (e.g., IndicatorSignals)
            labeler: An instance of the labeler class (e.g., ForwardReturnLabeler)
        """
        self.feature_engineer_class = feature_engineer_class
        self.labeler = labeler

    def build(self, df: pd.DataFrame):
        """
        Process a single DataFrame to produce X, y.

        Args:
            df (pd.DataFrame): Raw stock data with 'close', 'open', 'high', 'low', 'volume'.

        Returns:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Label vector
        """
        # Check if the cached CSV file exists
        cached_file = 'data/X.csv'
        if os.path.exists(cached_file):
            print(f"=== Loading cached data from {cached_file} ===")
            cached_data = pd.read_csv(cached_file)
            X = cached_data.drop(columns=["label"])
            y = cached_data["label"]
        else:
            print(f"=== No cached data found, computing features and labels ===")
            df = df.copy()
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

            # Instantiate feature_engineer
            feature_engineer = self.feature_engineer_class(df)
            signals = feature_engineer.get_signals()  # Call get_signals on the instance of the feature_engineer

            # Instantiate labeler and compute labels
            labels = self.labeler.label(df)  # Call label on the instance of the labeler
            

            # Align signals with the labels and prepare the final DataFrame
            signals = signals.loc[labels.index]
            combined = signals.copy()
            combined["label"] = labels

            combined = combined.dropna()  # Drop rows with NaN values

            X = combined.drop(columns=["label"])
            y = combined["label"]

            # Save the computed data to CSV
            combined.to_csv(cached_file, index=False)
            print(f"=== Data computed and saved to {cached_file} ===")

        return X, y