import pandas as pd
import numpy as np
from src.labelers.base_labeler import BaseLabeler

class ForwardReturnLabeler(BaseLabeler):
    def __init__(self, forward_days=20, pos_threshold=0.01, neg_threshold=-0.01, three_class=True):
        self.forward_days = forward_days
        self.pos_threshold = pos_threshold  # Lowered threshold
        self.neg_threshold = neg_threshold  # Lowered threshold
        self.three_class = three_class

    def label(self, df: pd.DataFrame) -> pd.Series:
        future_return = df["close"].pct_change(self.forward_days).shift(-self.forward_days)

        if self.three_class:
            labels = pd.Series(0, index=future_return.index)
            labels[future_return > self.pos_threshold] = 1
            labels[future_return < self.neg_threshold] = -1
        else:
            # Amplifying small returns by adjusting scaling
            scaled_returns = future_return * 5  # Increase sensitivity
            labels = np.tanh(scaled_returns)

        return labels