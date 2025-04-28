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


tickers_df = pd.read_csv("data/valid_tickers.csv")
tickers = tickers_df["Ticker"].tolist()

seed = 42

train_tickers, test_tickers = train_test_split(tickers, test_size=30, random_state=seed)

print(f"Train tickers: {len(train_tickers)}")
print(f"Test tickers: {len(test_tickers)}")

labeler_instance = TripleBarrierLabeler()

builder = LabeledDesignMatrixBuilder(
    tickers=train_tickers,
    labeler=labeler_instance
)

X_train, y_train = builder.build()

models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "AdaBoost": AdaBoostClassifier(),
    "Naive Bayes": GaussianNB(),
    "Ridge Classifier": RidgeClassifier(),
    "LDA": LinearDiscriminantAnalysis(),
    "QDA": QuadraticDiscriminantAnalysis(),
    "Perceptron": Perceptron(),
    "SGD": SGDClassifier()
}

model_performance = {model_name: {"total_returns": [], "buy_and_hold_returns": []} for model_name in models}

def get_random_year(price_data):
    price_data['year'] = price_data.index.year
    unique_years = price_data['year'].unique()
    random_year = random.choice(unique_years)
    return random_year

sample_ticker = test_tickers[0]
price_data = pd.read_csv(f"data/tickers/{sample_ticker}.csv")
if "date" in price_data.columns:
    price_data["date"] = pd.to_datetime(price_data["date"])
    price_data = price_data.set_index("date")

random_year = get_random_year(price_data)
print(f"Using Random Year: {random_year}")

for model_name, model in models.items():
    print(f"\n=== Training {model_name} ===")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model.fit(X_train_scaled, y_train)
    print(f"{model_name} trained.")

    for ticker in test_tickers:
        print(f"\n=== Backtesting for {ticker} ===")

        price_data = pd.read_csv(f"data/tickers/{ticker}.csv")
        if "date" in price_data.columns:
            price_data["date"] = pd.to_datetime(price_data["date"])
            price_data = price_data.set_index("date")
        
        feature_engineer = IndicatorSignals(price_data)
        signals = feature_engineer.get_signals()
        
        signals_scaled = scaler.transform(signals)
        
        predicted_labels = model.predict(signals_scaled)
        predicted_labels_series = pd.Series(predicted_labels, index=price_data.index)
        
        backtest_runner = BacktestRunner(price_data=price_data, signal_features=predicted_labels_series)
        
        backtest_runner.backtest()

        total_return = backtest_runner.get_total_return()
        buy_and_hold_return = backtest_runner.get_buy_and_hold_return()

        model_performance[model_name]["total_returns"].append(total_return)
        model_performance[model_name]["buy_and_hold_returns"].append(buy_and_hold_return)

        print(f"Total Return for {ticker} with {model_name}: {total_return:.4f}")
        print(f"Sharpe Ratio for {ticker} with {model_name}: {backtest_runner.get_sharpe_ratio():.4f}")
        print(f"Buy and Hold Return for {ticker} with {model_name}: {buy_and_hold_return:.4f}")

print("\n=== Summary of Results ===")
for model_name, performance in model_performance.items():
    avg_total_return = sum(performance["total_returns"]) / len(performance["total_returns"])
    avg_buy_and_hold_return = sum(performance["buy_and_hold_returns"]) / len(performance["buy_and_hold_returns"])
    print(f"\nModel: {model_name}")
    print(f"Average Total Return: {avg_total_return:.4f}")
    print(f"Average Buy and Hold Return: {avg_buy_and_hold_return:.4f}")