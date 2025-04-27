from src.labelers.forward_return_labeler import ForwardReturnLabeler
from src.indicator_signals import IndicatorSignals
from src.labeled_design_matrix_builder import LabeledDesignMatrixBuilder

builder = LabeledDesignMatrixBuilder(
    tickers_csv="data/valid_tickers.csv",  # Path to your valid_tickers.csv file
    labeler_class=ForwardReturnLabeler
)

X, y = builder.build()

print("\n=== Feature and Label Summary ===")
print(f"Feature matrix X shape: {X.shape}")
print(f"Label vector y shape: {y.shape}")