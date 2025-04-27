import os
import pickle
import backtrader as bt
import pandas as pd
import numpy as np
from datetime import datetime
import seaborn as sns
from tqdm import tqdm
import concurrent.futures
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_handler import DataHandler
from src.indicator_signals import IndicatorSignals
from src.plot_utils import plot_log_scale_equity, plot_strategy_vs_buyhold, plot_strategy_vs_buyhold_with_markers, plot_smoothed_return_contour
from src.backtest_runner import BacktestRunner

DEBUG = True

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

def main():
    # Load data
    tickers = ["QQQ"]
    indicator = "SMA"
    backtest_ticker = "QQQ"
    start_date = "2000-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    handler = DataHandler(backtest_ticker, start_date=start_date, end_date=end_date)
    price_data = handler.download_data()

    debug_print("[DEBUG] price_data.head():", price_data.head())
    debug_print("[DEBUG] price_data.columns:", price_data.columns)

    price_data_single = price_data
    

    debug_print("[DEBUG] Selected price_data_single.head():", price_data_single.head())

    import json

    # Ensure cache directory exists
    os.makedirs('cache', exist_ok=True)

    cache_file = os.path.join('cache', f"results_cache_{backtest_ticker}_{start_date}_to_{end_date}_{indicator}.json")
    cached_results = []

    if os.path.exists(cache_file):
        try:
            with open(cache_file, "r") as f:
                cached_results = json.load(f)
            print(f"[INFO] Loaded {len(cached_results)} cached results from {cache_file}")
        except Exception as e:
            print(f"[WARNING] Failed to load {cache_file}: {e}")

    cached_lookup = {(row['short'], row['long']): row['return'] for row in cached_results}

    # Prepare list of valid (short_window, long_window) pairs
    valid_pairs = [(s, l) for s in range(5, 121, 2) for l in range(20, 250, 5) if s < l]

    # Generate position features (signals)
    signals = IndicatorSignals(price_data_single)
    position_features = signals.generate_ma_cross_signal(short_window=41, long_window=140)  # You can adjust this based on best parameters

    # Create the BacktestRunner instance with the signal features
    runner = BacktestRunner(price_data_single, position_features)

    if len(cached_lookup) == len(valid_pairs):
        print("[INFO] All results cached. Skipping computations.")
        results = [(s, l, cached_lookup[(s, l)]) for s, l in valid_pairs]
    else:
        futures = {}
        results = []

        pbar = tqdm(total=len(valid_pairs), desc="Overall Progress")

        with concurrent.futures.ProcessPoolExecutor() as executor:
            for s, l in valid_pairs:
                if (s, l) in cached_lookup:
                    results.append((s, l, cached_lookup[(s, l)]))
                    pbar.update(1)
                else:
                    futures[executor.submit(runner.backtest, s, l)] = (s, l)

            for future in concurrent.futures.as_completed(futures):
                debug_print(f"[DEBUG] Waiting for backtest result for short_window={futures[future][0]}, long_window={futures[future][1]}")
                short_window, long_window, total_return = future.result()
                debug_print(f"[DEBUG] Received result: short={short_window}, long={long_window}, return={total_return:.4f}")
                results.append((short_window, long_window, total_return))
                pbar.update(1)
                # Save intermediate results after each computation
                cached_results = [{"short": s, "long": l, "return": r} for (s, l, r) in results]
                with open(cache_file, "w") as f:
                    json.dump(cached_results, f, indent=2)
                debug_print(f"[DEBUG] Intermediate results saved ({len(cached_results)} entries)")

        pbar.close()

    cached_results = [{"short": s, "long": l, "return": r} for (s, l, r) in results]
    with open(cache_file, "w") as f:
        json.dump(cached_results, f, indent=2)
    print(f"[INFO] Saved {len(cached_results)} results to {cache_file}")

    best = max(results, key=lambda x: x[2])  # third element = return
    best_short_window, best_long_window, best_total_return = best

    print(f"\nBest short_window={best_short_window}, long_window={best_long_window} with return {best_total_return * 100:.2f}%")

    # Calculate and print buy-and-hold return for the backtest_ticker
    buy_and_hold_return = (price_data_single['close'].iloc[-1] - price_data_single['close'].iloc[0]) / price_data_single['close'].iloc[0]
    print(f"Buy-and-Hold Return for {backtest_ticker}: {buy_and_hold_return * 100:.2f}%")

    debug_print("[DEBUG] Running best strategy for plotting...")
    # Re-run the best strategy for plotting
    runner.backtest(best_short_window, best_long_window)
    best_equity_df = runner.get_equity_curve()

    initial_close = price_data_single['close'].iloc[0]
    best_equity_df['Buy and Hold Value'] = (price_data_single['close'].iloc[:len(best_equity_df)] / initial_close) * 100000
    debug_print(f"[DEBUG] Created best_equity_df with {len(best_equity_df)} rows")

    trade_log_df = runner.get_trade_log()
    if not trade_log_df.empty and {"action", "price", "date"}.issubset(trade_log_df.columns):
        buys_raw = trade_log_df[trade_log_df["action"].str.upper() == "BUY"]
        sells_raw = trade_log_df[trade_log_df["action"].str.upper() == "SELL"]
        # Prepare buy_signals and sell_signals DataFrames with expected columns
        buys = pd.DataFrame({
            "Price": buys_raw["price"].values,
            "Date": pd.to_datetime(buys_raw["date"]).values
        })
        sells = pd.DataFrame({
            "Price": sells_raw["price"].values,
            "Date": pd.to_datetime(sells_raw["date"]).values
        })
    else:
        buys = sells = None
        print("[WARNING] Trade log missing expected columns or empty. Skipping markers.")

    daily_returns = best_equity_df['Portfolio Value'].pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    print(f"Best Parameters Summary:")
    print(f"Short Window: {best_short_window}")
    print(f"Long Window: {best_long_window}")
    print(f"Total Return: {best_total_return * 100:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")

    # Use plotting functions from plot_utils.py
    plot_log_scale_equity(best_equity_df, save_path="plots/log_scale_portfolio_value.png")
    plot_strategy_vs_buyhold(price_data_single, best_equity_df, trade_log_df, backtest_ticker, start_date, end_date)
    plot_strategy_vs_buyhold_with_markers(best_equity_df, buys, sells, ticker=backtest_ticker, start_date=start_date, end_date=end_date)

    results_df = pd.DataFrame(results, columns=["short", "long", "return"])
    pivot_table = results_df.pivot(index="short", columns="long", values="return")
    plot_smoothed_return_contour(pivot_table)

if __name__ == "__main__":
    main()
