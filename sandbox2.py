from src.labelers.forward_return_labeler import ForwardReturnLabeler
from src.indicator_signals import IndicatorSignals
from src.labeled_design_matrix_builder import LabeledDesignMatrixBuilder

# Create an instance of ForwardReturnLabeler
labeler_instance = ForwardReturnLabeler(
    forward_days=20,  # Adjust the number of forward days as needed
    pos_threshold=0.01,  # Positive threshold
    neg_threshold=-0.01,  # Negative threshold
    three_class=True  # Set whether to use three-class classification or not
)

# Pass the labeler instance to the builder
builder = LabeledDesignMatrixBuilder(
    tickers_csv="data/valid_tickers.csv",  # Path to your valid_tickers.csv file
    labeler=labeler_instance  # Pass the labeler instance here
)

X, y = builder.build()

print("\n=== Feature and Label Summary ===")
print(f"Feature matrix X shape: {X.shape}")
print(f"Label vector y shape: {y.shape}")