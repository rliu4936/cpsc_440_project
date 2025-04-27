# Initialize the LabeledDesignMatrixBuilder with the path to your valid_tickers.csv
builder = LabeledDesignMatrixBuilder(tickers_csv="data/valid_tickers.csv")

# Build the feature matrix X and label vector y
X, y = builder.build()

# Print summary of the results
print("\n=== Feature and Label Summary ===")
print(f"Feature matrix X shape: {X.shape}")
print(f"Label vector y shape: {y.shape}")

# Print label distribution
label_counts = y.value_counts()
total_labels = len(y)
print("\n=== Label Distribution Summary ===")
for label, count in label_counts.items():
    pct = count / total_labels * 100
    print(f"Label {label}: {count} ({pct:.2f}%)")
    