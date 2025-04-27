import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from src.data_handler import DataHandler
from src.indicator_signals import IndicatorSignals

X_list = []
y_list = []


# Read tickers
valid_tickers = pd.read_csv("data/valid_tickers.csv", header=None)[0].tolist()

start_date = "2000-01-01"
end_date = "2025-01-01"

# Initialize a dict to store global counts
global_counts = {}

for ticker in valid_tickers:
    print(f"\n=== Processing {ticker} ===")
    handler = DataHandler(ticker, start_date=start_date, end_date=end_date)
    try:
        df = handler.download_data()
        if df is None or df.empty:
            print(f"[WARNING] No data for {ticker}. Skipping.")
            continue

        # Ensure 'date' is datetime and set as index
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        signals = IndicatorSignals(df).get_signals()

        # Create labels based on 5-day forward return > 2%
        df["future_return"] = df["close"].pct_change(5).shift(-5)
        df["label"] = (df["future_return"] > 0.02).astype(int)
        
        # Align signals and labels
        signals = signals.loc[df.index]
        combined = signals.copy()
        combined["label"] = df["label"]
        
        combined = combined.dropna()
        X_list.append(combined.drop(columns=["label"]))
        y_list.append(combined["label"])

        for column in signals.columns:
            counts = signals[column].value_counts().to_dict()
            zeros = counts.get(0, 0)
            ones = counts.get(1, 0)

            if column not in global_counts:
                global_counts[column] = {"zeros": 0, "ones": 0}
            
            global_counts[column]["zeros"] += zeros
            global_counts[column]["ones"] += ones

    except Exception as e:
        print(f"[ERROR] Failed for {ticker}: {e}")

# Build final summary
summary_rows = []
for signal_name, counts in global_counts.items():
    total = counts["zeros"] + counts["ones"]
    pct_ones = counts["ones"] / total * 100 if total > 0 else 0
    summary_rows.append({
        "signal": signal_name,
        "total_zeros": counts["zeros"],
        "total_ones": counts["ones"],
        "percent_ones": round(pct_ones, 2)
    })

summary_df = pd.DataFrame(summary_rows)
summary_df = summary_df.sort_values("signal")

# Print nicely
print("\n=== Global Signal Summary Across All Tickers ===")
print(summary_df)

# Optional: save
summary_df.to_csv("global_signal_summary.csv", index=False)

X = pd.concat(X_list, axis=0)
y = pd.concat(y_list, axis=0)

print(f"\nFeature matrix X shape: {X.shape}")
print(f"Label vector y shape: {y.shape}")

# Print label distribution
label_counts = y.value_counts()
total_labels = len(y)
print("\n=== Label Distribution Summary ===")
for label, count in label_counts.items():
    pct = count / total_labels * 100
    print(f"Label {label}: {count} ({pct:.2f}%)")

X.to_csv("X.csv", index=False)
y.to_csv("y.csv", index=False)
