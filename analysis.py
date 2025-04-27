from src.labelers.forward_return_labeler import ForwardReturnLabeler
from src.labelers.triple_barrier_labeler import TripleBarrierLabeler
from src.labeled_design_matrix_builder import LabeledDesignMatrixBuilder
from src.backtest_runner2 import BacktestRunner
from src.indicator_signals import IndicatorSignals
import pandas as pd


tickers_df = pd.read_csv("data/valid_tickers.csv")  # Path to your valid_tickers.csv file
tickers = tickers_df["Ticker"].tolist()  # Extract the tickers from the 'Ticker' column
from sklearn.model_selection import train_test_split

# Set the seed for reproducibility
seed = 42

# Randomly split the tickers into 20 for testing and the rest for training
train_tickers, test_tickers = train_test_split(tickers, test_size=20, random_state=seed)

# Verify the split
print(f"Train tickers: {len(train_tickers)}")
print(f"Test tickers: {len(test_tickers)}")

# Step 2: Initialize the labeler instance
labeler_instance = TripleBarrierLabeler()

# Step 3: Create the LabeledDesignMatrixBuilder for the training tickers
builder = LabeledDesignMatrixBuilder(
    tickers=train_tickers,  # Pass the list of training tickers directly
    labeler=labeler_instance  # Pass the labeler instance here
)

# Step 4: Generate features and labels for the training data
X_train, y_train = builder.build()

# Step 5: Train your model (e.g., Logistic Regression, Random Forest, etc.)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler


# Standardize the features (important for models like SVM, etc.)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# Train the Logistic Regression model
print(f"Training Model")
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(f"Mode Trained")

# Step 6: Initialize variables to store the returns
total_returns = []
buy_and_hold_returns = []

# Iterate through the 20 test tickers, generate signals, get predicted labels, and backtest
for ticker in test_tickers:
    print(f"\n=== Backtesting for {ticker} ===")
    
    # Load the price data for the ticker (assuming it's available in CSV)
    price_data = pd.read_csv(f"data/tickers/{ticker}.csv")  # Adjust path if necessary
    if "date" in price_data.columns:
        price_data["date"] = pd.to_datetime(price_data["date"])
        price_data = price_data.set_index("date")
    
    # Generate the feature signals for this ticker
    feature_engineer = IndicatorSignals(price_data)  # Assuming you're using the same feature engineer for test data
    signals = feature_engineer.get_signals()  # Get the signals (features) for the test data
    
    # Standardize the test features using the same scaler as the training set
    signals_scaled = scaler.transform(signals)  # Standardize signals using the same scaler
    
    # Predict the labels using the trained model
    predicted_labels = model.predict(signals_scaled)  # Get the predicted labels
    predicted_labels_series = pd.Series(predicted_labels, index=price_data.index)
    
    # Step 7: Initialize the BacktestRunner with the price data and predicted labels
    backtest_runner = BacktestRunner(price_data=price_data, signal_features=predicted_labels_series)
    
    # Step 8: Run the backtest
    backtest_runner.backtest()

    # Step 9: Collect performance statistics
    total_return = backtest_runner.get_total_return()
    buy_and_hold_return = backtest_runner.get_buy_and_hold_return()

    # Append returns to the respective lists
    total_returns.append(total_return)
    buy_and_hold_returns.append(buy_and_hold_return)

    # Print individual performance statistics
    print(f"Total Return for {ticker}: {total_return:.4f}")
    print(f"Sharpe Ratio for {ticker}: {backtest_runner.get_sharpe_ratio():.4f}")
    print(f"Buy and Hold Return for {ticker}: {buy_and_hold_return:.4f}")

# Step 10: Calculate and print average performance statistics
average_total_return = sum(total_returns) / len(total_returns)
average_buy_and_hold_return = sum(buy_and_hold_returns) / len(buy_and_hold_returns)

print("\n=== Average Performance for 20 Test Tickers ===")
print(f"Average Total Return: {average_total_return:.4f}")
print(f"Average Buy and Hold Return: {average_buy_and_hold_return:.4f}")