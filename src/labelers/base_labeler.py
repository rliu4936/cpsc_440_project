

from abc import ABC, abstractmethod
import pandas as pd

class BaseLabeler(ABC):
    @abstractmethod
    def label(self, df: pd.DataFrame) -> pd.Series:
        
        pass