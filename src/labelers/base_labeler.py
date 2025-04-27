

from abc import ABC, abstractmethod
import pandas as pd

class BaseLabeler(ABC):
    @abstractmethod
    def label(self, df: pd.DataFrame) -> pd.Series:
        """
        Given a DataFrame with at least a 'close' column and a datetime index,
        return a pd.Series of labels indexed the same way.

        Args:
            df (pd.DataFrame): DataFrame containing market data (must include 'close').

        Returns:
            pd.Series: Series of labels indexed the same as df.
        """
        pass