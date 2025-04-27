from src.labelers.forward_return_labeler import ForwardReturnLabeler
from src.indicator_signals import IndicatorSignals
from src.labelers.triple_barrier_labeler import TripleBarrierLabeler
from src.labeled_design_matrix_builder import LabeledDesignMatrixBuilder

# Create an instance of ForwardReturnLabeler
labeler_instance = TripleBarrierLabeler()

# Pass the labeler instance to the builder
builder = LabeledDesignMatrixBuilder(
    tickers_csv="data/valid_tickers.csv",  # Path to your valid_tickers.csv file
    labeler=labeler_instance  # Pass the labeler instance here
)

X, y = builder.build()

print("\n=== Feature and Label Summary ===")
print(f"Feature matrix X shape: {X.shape}")
print(f"Label vector y shape: {y.shape}")