# src/feature_label_builder.py
import pandas as pd

class FeatureLabelBuilder:
    def __init__(self, feature_engineer_class, labeler):
        """
        Args:
            feature_engineer_class: A class that generates features (e.g., IndicatorSignals)
            labeler_class: A class that generates labels (e.g., ForwardReturnLabeler)
            labeler_args: Additional positional arguments for the labeler class constructor
            labeler_kwargs: Additional keyword arguments for the labeler class constructor
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
        df = df.copy()
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        # Instantiate feature_engineer
        feature_engineer = self.feature_engineer_class(df)
        signals = feature_engineer.get_signals()  # Call get_signals on the instance of the feature_engineer

        # Instantiate labeler with args and kwargs
        labels = self.labeler.label(df)  # Call label on the instance of the labeler

        # Align signals with the labels and prepare the final DataFrame
        signals = signals.loc[labels.index]
        combined = signals.copy()
        combined["label"] = labels

        combined = combined.dropna()  # Drop rows with NaN values

        X = combined.drop(columns=["label"])
        y = combined["label"]

        return X, y