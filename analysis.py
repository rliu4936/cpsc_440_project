import random
from src.labelers.forward_return_labeler import ForwardReturnLabeler
from src.labelers.triple_barrier_labeler import TripleBarrierLabeler
from src.labeled_design_matrix_builder import LabeledDesignMatrixBuilder
from src.backtest_runner2 import BacktestRunner
from src.indicator_signals import IndicatorSignals
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier


tickers_df = pd.read_csv("data/valid_tickers.csv")  # Path to your valid_tickers.csv file
tickers = tickers_df["Ticker"].tolist()  # Extract the tickers from the 'Ticker' column

# Set the seed for reproducibility
seed = 42

# Randomly split the tickers into 20 for testing and the rest for training
train_tickers, test_tickers = train_test_split(tickers, test_size=30, random_state=seed)

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

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Gradient Boosting": GradientBoostingClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB(),
    "Ridge Classifier": RidgeClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "Perceptron": Perceptron(),
    "SGD": SGDClassifier()
}

# Step 5: Initialize variables to store the returns
total_returns = []
buy_and_hold_returns = []

# Select a random year from the available date range in the price data
def get_random_year(price_data):
    # Extract the year from the date index
    price_data['year'] = price_data.index.year
    unique_years = price_data['year'].unique()  # Get unique years in the data
    random_year = random.choice(unique_years)  # Select a random year
    return random_year

# Step 6: Train and evaluate each model
for model_name, model in models.items():
    print(f"\n=== Training {model_name} ===")
    
    # Step 5: Standardize the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train the model
    model.fit(X_train_scaled, y_train)
    print(f"{model_name} trained.")

    # Step 6: Iterate through the 20 test tickers, generate signals, get predicted labels, and backtest
    for ticker in test_tickers:
        print(f"\n=== Backtesting for {ticker} ===")

        # Load the price data for the ticker (assuming it's available in CSV)
        price_data = pd.read_csv(f"data/tickers/{ticker}.csv")  # Adjust path if necessary
        if "date" in price_data.columns:
            price_data["date"] = pd.to_datetime(price_data["date"])
            price_data = price_data.set_index("date")

        # Select a random year (you can comment out the line to test with all years)
        random_year = get_random_year(price_data)  # Get a random year
        print(f"Using Random Year: {random_year}")

        # Filter the data for the random year
        price_data = price_data[price_data.index.year == random_year]
        
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
        print(f"Total Return for {ticker} with {model_name}: {total_return:.4f}")
        print(f"Sharpe Ratio for {ticker} with {model_name}: {backtest_runner.get_sharpe_ratio():.4f}")
        print(f"Buy and Hold Return for {ticker} with {model_name}: {buy_and_hold_return:.4f}")

# Step 10: Calculate and print average performance statistics for each model
average_total_return = sum(total_returns) / len(total_returns)
average_buy_and_hold_return = sum(buy_and_hold_returns) / len(buy_and_hold_returns)

print("\n=== Average Performance for 20 Test Tickers ===")
print(f"Average Total Return: {average_total_return:.4f}")
print(f"Average Buy and Hold Return: {average_buy_and_hold_return:.4f}")