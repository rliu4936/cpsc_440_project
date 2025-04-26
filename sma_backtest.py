import os
import pickle
import backtrader as bt
import pandas as pd
import numpy as np
from DataHandler import DataHandler
from IndicatorSignals import IndicatorSignals
from datetime import datetime
import seaborn as sns
from tqdm import tqdm
import concurrent.futures

class MAStrategy(bt.Strategy):
    def __init__(self, feature_data):
        self.feature_data = feature_data
        self.dataclose = self.datas[0].close
        self.order = None
        self.portfolio_values = []
        self.trade_log = []

    def buy_all(self):
        cash = self.broker.getcash()
        price = self.dataclose[0]
        size = int(cash / price)
        if size > 0:
            self.order = self.buy(size=size)

    def next(self):
        current_dt = pd.Timestamp(self.datas[0].datetime.date(0))

        target_position = self.feature_data.get(current_dt, 0)

        if target_position == 1:
            if self.position.size <= 0:
                cash = self.broker.getcash()
                size = int(cash / self.dataclose[0])
                if size > 0:
                    self.order = self.buy(size=size)
                    # print(f"[{current_dt.date()}] BUY {size} @ {self.dataclose[0]:.2f}")
                    self.trade_log.append({"date": current_dt, "action": "BUY", "price": self.dataclose[0], "size": size})
        elif target_position == 0:
            if self.position.size > 0:
                self.order = self.close()
                # print(f"[{current_dt.date()}] SELL @ {self.dataclose[0]:.2f}")
                self.trade_log.append({"date": current_dt, "action": "SELL", "price": self.dataclose[0], "size": self.position.size})
        else:
            # print(f"[{current_dt.date()}] HOLD")
            pass

        self.portfolio_values.append(self.broker.getvalue())

def run_backtest(short_window, long_window, price_data_single):
    signals = IndicatorSignals(price_data_single)
    position_features = signals.generate_ma_cross_signal(short_window=short_window, long_window=long_window)

    cerebro = bt.Cerebro()
    data = bt.feeds.PandasData(dataname=price_data_single)
    cerebro.adddata(data)
    cerebro.addstrategy(MAStrategy, feature_data=position_features)
    cerebro.broker.set_cash(100000)

    strat = cerebro.run(runonce=True)[0]

    start_value = 100000
    end_value = strat.broker.getvalue()
    total_return = (end_value - start_value) / start_value

    return (short_window, long_window, total_return, strat, cerebro)

def main():
    # Load data
    tickers = ["QQQ", "SPY", "AAPL", "MSFT", "EWH"]
    indicator = "SMA"
    backtest_ticker = "QQQ"
    start_date = "2000-01-01"
    end_date = datetime.today().strftime("%Y-%m-%d")
    # end_date = "2015-01-01"
    handler = DataHandler(tickers, start_date=start_date, end_date=end_date)
    price_data = handler.download_data()

    print("[DEBUG] price_data.head():")
    print(price_data.head())
    print("[DEBUG] price_data.columns:")
    print(price_data.columns)

    if 'Ticker' in price_data.columns:
        price_data_single = price_data[price_data['Ticker'] == backtest_ticker].drop(columns='Ticker').dropna()
    else:
        raise ValueError(f"[ERROR] Cannot find 'Ticker' column in price_data.columns: {price_data.columns}")

    print("[DEBUG] Selected price_data_single.head():")
    print(price_data_single.head())

    import json

    cache_file = f"results_cache_{backtest_ticker}_{start_date}_to_{end_date}_{indicator}.json"
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
                    futures[executor.submit(run_backtest, s, l, price_data_single)] = (s, l)

            for future in concurrent.futures.as_completed(futures):
                short_window, long_window, total_return, strat, cerebro = future.result()
                results.append((short_window, long_window, total_return))
                pbar.update(1)
                # Save intermediate results after each computation
                cached_results = [{"short": s, "long": l, "return": r} for (s, l, r) in results]
                with open(cache_file, "w") as f:
                    json.dump(cached_results, f, indent=2)

        pbar.close()

    cached_results = [{"short": s, "long": l, "return": r} for (s, l, r) in results]
    with open(cache_file, "w") as f:
        json.dump(cached_results, f, indent=2)
    print(f"[INFO] Saved {len(cached_results)} results to {cache_file}")

    best = max(results, key=lambda x: x[2])  # third element = return
    best_short_window, best_long_window, best_total_return = best

    print(f"\nBest short_window={best_short_window}, long_window={best_long_window} with return {best_total_return * 100:.2f}%")

    # Calculate and print buy-and-hold return for the backtest_ticker
    buy_and_hold_return = (price_data_single['Close'].iloc[-1] - price_data_single['Close'].iloc[0]) / price_data_single['Close'].iloc[0]
    print(f"Buy-and-Hold Return for {backtest_ticker}: {buy_and_hold_return * 100:.2f}%")

    # Re-run the best strategy for plotting
    _, _, _, best_strat, best_cerebro = run_backtest(best_short_window, best_long_window, price_data_single)
    best_equity_df = pd.DataFrame({'Portfolio Value': best_strat.portfolio_values}, index=price_data_single.index[:len(best_strat.portfolio_values)])

    daily_returns = best_equity_df['Portfolio Value'].pct_change().dropna()
    sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
    print(f"Best Parameters Summary:")
    print(f"Short Window: {best_short_window}")
    print(f"Long Window: {best_long_window}")
    print(f"Total Return: {best_total_return * 100:.2f}%")
    print(f"Annualized Sharpe Ratio: {sharpe_ratio:.2f}")

    # Plot equity curve
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter

    os.makedirs('plots', exist_ok=True)

    plt.style.use('seaborn-v0_8-whitegrid')

    # Manual log-scale plot with trade markers
    plt.figure(figsize=(14, 7))
    plt.plot(best_equity_df.index, best_equity_df['Portfolio Value'], label="Portfolio Value", color='black', linewidth=2)
    plt.yscale('log')  # LOG SCALE
    plt.title('Log-Scale Portfolio Value with Trades', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Portfolio Value (Log Scale)', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.savefig('log_scale_portfolio_value.png')
    plt.close()

    # Plot strategy vs QQQ (improved clean style)
    plt.style.use('seaborn-v0_8-muted')
    plt.figure(figsize=(12, 7))
    plt.plot(best_equity_df.index, best_equity_df['Portfolio Value'] / best_equity_df['Portfolio Value'].iloc[0],
             label="Strategy (Best Params)", linewidth=2, color='steelblue')
    plt.plot(price_data_single.index[:len(best_equity_df)], price_data_single['Close'].iloc[:len(best_equity_df)] / price_data_single['Close'].iloc[0],
             label=f"{backtest_ticker} Buy & Hold", linewidth=2, color='indianred')

    # Add buy/sell markers on QQQ series
    buy_dates = [log["date"] for log in best_strat.trade_log if log["action"] == "BUY"]
    sell_dates = [log["date"] for log in best_strat.trade_log if log["action"] == "SELL"]

    buy_prices = []
    for d in buy_dates:
        if d in price_data_single.index:
            normalized_price = price_data_single.loc[d, "Close"] / price_data_single['Close'].iloc[0]
            buy_prices.append(normalized_price)
        else:
            buy_prices.append(np.nan)

    sell_prices = []
    for d in sell_dates:
        if d in price_data_single.index:
            normalized_price = price_data_single.loc[d, "Close"] / price_data_single['Close'].iloc[0]
            sell_prices.append(normalized_price)
        else:
            sell_prices.append(np.nan)

    buy_dates_filtered = [d for d, p in zip(buy_dates, buy_prices) if not np.isnan(p)]
    buy_prices_filtered = [p for p in buy_prices if not np.isnan(p)]

    sell_dates_filtered = [d for d, p in zip(sell_dates, sell_prices) if not np.isnan(p)]
    sell_prices_filtered = [p for p in sell_prices if not np.isnan(p)]

    plt.scatter(buy_dates_filtered, buy_prices_filtered, marker='^', color='darkgreen', label='Buy', s=80, edgecolors='black', zorder=5)
    plt.scatter(sell_dates_filtered, sell_prices_filtered, marker='v', color='firebrick', label='Sell', s=80, edgecolors='black', zorder=5)

    plt.title('Strategy vs Buy & Hold Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Normalized Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/strategy_vs_buyhold_{backtest_ticker}_{start_date}_to_{end_date}.png', dpi=300)
    plt.close()

    # Plot strategy vs QQQ with buy/sell markers
    plt.style.use('seaborn-v0_8-muted')
    plt.figure(figsize=(12, 7))
    plt.plot(best_equity_df.index, best_equity_df['Portfolio Value'] / best_equity_df['Portfolio Value'].iloc[0],
             label="Strategy (Best Params)", linewidth=2, color='steelblue')
    plt.plot(price_data_single.index[:len(best_equity_df)], price_data_single['Close'].iloc[:len(best_equity_df)] / price_data_single['Close'].iloc[0],
             label=f"{backtest_ticker} Buy & Hold", linewidth=2, color='indianred')

    # Add buy/sell markers
    buy_dates = [log["date"] for log in best_strat.trade_log if log["action"] == "BUY"]
    sell_dates = [log["date"] for log in best_strat.trade_log if log["action"] == "SELL"]

    buy_prices = []
    for d in buy_dates:
        if d in price_data_single.index:
            normalized_price = price_data_single.loc[d, "Close"] / price_data_single['Close'].iloc[0]
            buy_prices.append(normalized_price)
        else:
            buy_prices.append(np.nan)

    sell_prices = []
    for d in sell_dates:
        if d in price_data_single.index:
            normalized_price = price_data_single.loc[d, "Close"] / price_data_single['Close'].iloc[0]
            sell_prices.append(normalized_price)
        else:
            sell_prices.append(np.nan)

    buy_dates_filtered = [d for d, p in zip(buy_dates, buy_prices) if not np.isnan(p)]
    buy_prices_filtered = [p for p in buy_prices if not np.isnan(p)]

    sell_dates_filtered = [d for d, p in zip(sell_dates, sell_prices) if not np.isnan(p)]
    sell_prices_filtered = [p for p in sell_prices if not np.isnan(p)]

    plt.scatter(buy_dates_filtered, buy_prices_filtered, marker='^', color='darkgreen', label='Buy', s=80, edgecolors='black', zorder=5)
    plt.scatter(sell_dates_filtered, sell_prices_filtered, marker='v', color='firebrick', label='Sell', s=80, edgecolors='black', zorder=5)

    plt.title('Strategy vs Buy & Hold with Buy/Sell Markers', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Normalized Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/strategy_vs_buyhold_with_markers_{backtest_ticker}_{start_date}_to_{end_date}.png', dpi=300)
    plt.close()

    # New clean plot for strategy vs buy-and-hold returns
    plt.style.use('seaborn-v0_8-muted')
    plt.figure(figsize=(12, 7))
    plt.plot(best_equity_df.index, best_equity_df['Portfolio Value'] / best_equity_df['Portfolio Value'].iloc[0],
             label="Strategy (Best Params)", linewidth=2, color='steelblue')
    plt.plot(price_data_single.index[:len(best_equity_df)], price_data_single['Close'].iloc[:len(best_equity_df)] / price_data_single['Close'].iloc[0],
             label=f"{backtest_ticker} Buy & Hold", linewidth=2, color='indianred')

    plt.title('Strategy vs Buy & Hold Comparison', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Normalized Value', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    plt.tight_layout()
    os.makedirs('plots', exist_ok=True)
    plt.savefig(f'plots/strategy_vs_buyhold_{backtest_ticker}_{start_date}_to_{end_date}.png', dpi=300)
    plt.close()

    # Plot cerebro chart
    best_cerebro.plot(iplot=False, volume=False)

    # Visualize hyperparameter search using already computed results
    results_df = pd.DataFrame(results, columns=['Short Window', 'Long Window', 'Return'])

    pivot_table = results_df.pivot(index="Short Window", columns="Long Window", values="Return")

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 8))
    pivot_table_filled = pivot_table.fillna(0)
    smoothed = gaussian_filter(pivot_table_filled.values, sigma=1)

    X, Y = np.meshgrid(pivot_table.columns, pivot_table.index)
    contour = plt.contourf(X, Y, smoothed, levels=20, cmap="coolwarm")
    plt.colorbar(contour)
    plt.title('Smoothed Return Contour: Short Window vs Long Window', fontsize=16)
    plt.xlabel('Long Window', fontsize=14)
    plt.ylabel('Short Window', fontsize=14)
    plt.tight_layout()
    plt.savefig('smoothed_return_contour.png')
    plt.close()

if __name__ == "__main__":
    main()