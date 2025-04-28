import os
import pandas as pd

class FeatureLabelBuilder:
    def __init__(self, feature_engineer_class, labeler):
        self.feature_engineer_class = feature_engineer_class
        self.labeler = labeler

    def build(self, df: pd.DataFrame):
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
            feature_engineer = self.feature_engineer_class(df)
            signals = feature_engineer.get_signals()
            labels = self.labeler.label(df)
            signals = signals.loc[labels.index]
            combined = signals.copy()
            combined["label"] = labels
            combined = combined.dropna()
            X = combined.drop(columns=["label"])
            y = combined["label"]
            combined.to_csv(cached_file, index=False)
            print(f"=== Data computed and saved to {cached_file} ===")

        return X, y