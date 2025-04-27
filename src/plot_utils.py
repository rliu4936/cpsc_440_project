import matplotlib.pyplot as plt
import os
import pandas as pd

# Plotting Functions

def plot_equity_curve(equity_df, output_dir, strategy_name="strategy"):
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df.index, equity_df["Portfolio Value"], label="Strategy")
    plt.title(f"{strategy_name} Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    save_plot(plt, f"{strategy_name}_equity_curve.png", output_dir)
    plt.close()

def plot_log_scale_equity(equity_df, save_path=None, strategy_name="strategy"):
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df.index, equity_df["Portfolio Value"], label="Portfolio Value")
    plt.yscale("log")

    plt.title(f"Portfolio Value Over Time (Log Scale) ({strategy_name})")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value (log scale)")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"[INFO] Saved plot to {save_path}")
    else:
        plt.show()

    plt.close()

def plot_strategy_vs_buyhold(price_data_single, equity_df, trade_log, ticker, start_date, end_date, output_dir="plots", log_scale=True):
    plt.figure(figsize=(14, 7))
    print(f"[DEBUG] price_data_single.columns: {price_data_single.columns}")

    plt.plot(price_data_single.index, price_data_single["Close"], label=f"{ticker} Close Price", color="black", linewidth=1.5)

    if trade_log:
        trade_log_df = pd.DataFrame(trade_log)

        if "action" in trade_log_df.columns and "price" in trade_log_df.columns and "date" in trade_log_df.columns:
            buys = trade_log_df[trade_log_df["action"].str.upper() == "BUY"]
            sells = trade_log_df[trade_log_df["action"].str.upper() == "SELL"]

            plt.scatter(buys["date"], buys["price"], marker="^", color="green", label="Buy Signal", s=50)
            plt.scatter(sells["date"], sells["price"], marker="v", color="red", label="Sell Signal", s=50)
        else:
            print("[WARNING] Trade log missing expected columns.")

    if log_scale:
        plt.yscale('log')

    plt.title(f"Strategy vs Buy and Hold for {ticker} ({start_date} to {end_date})")
    plt.xlabel("Date")
    plt.ylabel("Price" + (" (Log Scale)" if log_scale else ""))
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    save_plot(plt, f"{ticker}_strategy_vs_buyhold_{start_date}_to_{end_date}.png", output_dir)
    plt.close()

def plot_smoothed_return_contour(pivot_table, output_dir="plots", strategy_name="strategy"):
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(
        pivot_table.columns, pivot_table.index, pivot_table.values,
        levels=20, cmap="plasma"
    )
    cbar = plt.colorbar(contour)
    cbar.set_label("Return (%)", fontsize=12)

    plt.title("Smoothed Strategy Return (%)", fontsize=14, pad=10)
    plt.xlabel("Long Window", fontsize=12)
    plt.ylabel("Short Window", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(False)
    plt.tight_layout()
    save_plot(plt, f"{strategy_name}_smoothed_contour.png", output_dir)
    plt.close()

def plot_strategy_vs_buyhold_with_markers(equity_df, buys=None, sells=None, output_dir="plots", ticker="Ticker", strategy_name="strategy", start_date=None, end_date=None):
    plt.figure(figsize=(12, 6))
    plt.plot(equity_df.index, equity_df["Portfolio Value"], label="Strategy", linewidth=2)

    if buys is not None and not buys.empty:
        plt.scatter(buys["date"], buys["price"], marker="^", color="green", label="Buy", s=50)
    if sells is not None and not sells.empty:
        plt.scatter(sells["date"], sells["price"], marker="v", color="red", label="Sell", s=50)

    plt.title(f"{ticker} Strategy with Buy/Sell Markers ({start_date} to {end_date})")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
    plt.tight_layout()
    save_plot(plt, f"{ticker}_strategy_vs_buyhold_with_markers_{start_date}_to_{end_date}.png", output_dir)
    plt.close()

def save_plot(fig, filename, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, filename))