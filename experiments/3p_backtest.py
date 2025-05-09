import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    plt.scatter(buy_hold_returns, actual_returns)
    # Fit line of best fit without bias (force through origin)
    x = np.array(buy_hold_returns)
    y = np.array(actual_returns)
    slope = np.dot(x, y) / np.dot(x, x)
    y_pred = slope * x
    plt.plot(x, y_pred, color='red', label=f'Best Fit (no bias): y = {slope:.2f}x')
    plt.xlabel("Buy and Hold Return")
    plt.ylabel("Strategy Return")
    plt.plot([min(buy_hold_returns), max(buy_hold_returns)],
            [min(buy_hold_returns), max(buy_hold_returns)],
            linestyle='--', color='gray', label='y = x')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title("Strategy vs Buy and Hold Returns")
    plt.grid(True)
    plt.show()