from src.labelers.forward_return_labeler import ForwardReturnLabeler
from src.labelers.triple_barrier_labeler import TripleBarrierLabeler
from src.labeled_design_matrix_builder import LabeledDesignMatrixBuilder
from src.backtest_runner2 import BacktestRunner
from src.indicator_signals import IndicatorSignals
import pandas as pd


tickers_df = pd.read_csv("data/valid_tickers.csv")  
tickers = tickers_df["Ticker"].tolist()  

test_tickers = tickers[:20]  
labeler_instance = TripleBarrierLabeler()  

for ticker in test_tickers:
    print(f"\n=== Backtesting for {ticker} ===")
    
    df = pd.read_csv(f"data/tickers/{ticker}.csv")  
    if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
                df = df.set_index("date")

    labels = labeler_instance.label(df)  
    backtest_runner = BacktestRunner(price_data=df, signal_features=labels)
    
    backtest_runner.backtest()

    print(f"Total Return for {ticker}: {backtest_runner.get_total_return():.4f}")
    print(f"Sharpe Ratio for {ticker}: {backtest_runner.get_sharpe_ratio():.4f}")
    print(f"Buy and Hold Return for {ticker}: {backtest_runner.get_buy_and_hold_return():.4f}")