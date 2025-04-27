from src.labelers.forward_return_labeler import ForwardReturnLabeler
from src.labelers.triple_barrier_labeler import TripleBarrierLabeler
from src.labeled_design_matrix_builder import LabeledDesignMatrixBuilder
from src.backtest_runner2 import BacktestRunner
from src.indicator_signals import IndicatorSignals
import pandas as pd


tickers_df = pd.read_csv("data/valid_tickers.csv")  # Path to your valid_tickers.csv file
tickers = tickers_df["Ticker"].tolist()  # Extract the tickers from the 'Ticker' column

test_tickers = tickers[:20]  # Use the first 20 for testing the returns
# Step 3: Initialize the labeler instance
labeler_instance = TripleBarrierLabeler()  # Create an instance of TripleBarrierLabeler

# Step 4: Iterate through the 20 test tickers, generate signals, get predicted labels, and backtest
for ticker in test_tickers:
    print(f"\n=== Backtesting for {ticker} ===")
    
    # Load the price data for the ticker (assuming it's available in CSV)
    df = pd.read_csv(f"data/tickers/{ticker}.csv")  # Adjust path if necessary
    if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

    # Generate the labels using the TripleBarrierLabeler instance
    labels = labeler_instance.label(df)  # Use the instance method to get labels
    # print(labels)
    # Step 8: Run the backtest
    backtest_runner = BacktestRunner(price_data=df, signal_features=labels)
    
    backtest_runner.backtest()

    # Step 9: Print the performance statistics
    print(f"Total Return for {ticker}: {backtest_runner.get_total_return():.4f}")
    print(f"Sharpe Ratio for {ticker}: {backtest_runner.get_sharpe_ratio():.4f}")
    # Print the buy and hold return
    print(f"Buy and Hold Return for {ticker}: {backtest_runner.get_buy_and_hold_return():.4f}")