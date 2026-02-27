import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# -----------------------------
# 1) Synthetic market data
# -----------------------------
np.random.seed(42)

def generate_price(n=600):
    trend = np.linspace(0, 12, n)
    noise = np.random.normal(0, 2.0, n)
    seasonal = 1.5 * np.sin(np.linspace(0, 10 * np.pi, n))
    return 100 + trend + seasonal + noise

prices = generate_price()


# -----------------------------
# 2) Strategy simulation + signal extraction
# -----------------------------
def simulate_and_signals(params, prices):
    """
    Returns:
      final_capital: float
      ma: np.ndarray (len(prices)) with NaN before ma_window
      buys: list[int] indices of buy points
      sells: list[int] indices of sell points
    """
    ma_window, buy_th, sell_th, stop_loss = params
    ma_window = int(ma_window)

    capital = 1000.0
    position = 0.0
    entry_price = 0.0

    ma = np.full_like(prices, np.nan, dtype=float)
    buys, sells = [], []

    for i in range(ma_window, len(prices)):
        ma[i] = np.mean(prices[i - ma_window:i])
        price = prices[i]

        # BUY: when price breaks above MA by buy_th
        if position == 0.0 and price > ma[i] * (1.0 + buy_th):
            position = capital / price
            entry_price = price
            capital = 0.0
            buys.append(i)

        # SELL: when price drops below MA by sell_th OR hits stop-loss from entry
        elif position > 0.0:
            if price < ma[i] * (1.0 - sell_th) or price < entry_price * (1.0 - stop_loss):
                capital = position * price
                position = 0.0
                sells.append(i)

    # Close any open position at the end
    if position > 0.0:
        capital = position * prices[-1]
        position = 0.0

    return capital, ma, buys, sells


# -----------------------------
# 3) Genetic Algorithm (simple)
# -----------------------------
POP_SIZE = 30
GENERATIONS = 22
MUTATION_RATE = 0.25

def random_individual():
    return [
        np.random.randint(5, 50),          # ma_window
        np.random.uniform(0.001, 0.02),    # buy_threshold (0.1%..2%)
        np.random.uniform(0.001, 0.02),    # sell_threshold
        np.random.uniform(0.01, 0.10),     # stop_loss (1%..10%)
    ]

def mutate(ind):
    ind = ind.copy()
    if np.random.rand() < MUTATION_RATE:
        ind[0] = np.random.randint(5, 50)
    if np.random.rand() < MUTATION_RATE:
        ind[1] = np.random.uniform(0.001, 0.02)
    if np.random.rand() < MUTATION_RATE:
        ind[2] = np.random.uniform(0.001, 0.02)
    if np.random.rand() < MUTATION_RATE:
        ind[3] = np.random.uniform(0.01, 0.10)
    return ind

def crossover(p1, p2):
    point = np.random.randint(1, 4)
    return p1[:point] + p2[point:]

population = [random_individual() for _ in range(POP_SIZE)]
best_history = []

best_ind = None
best_fit = -np.inf

for gen in range(GENERATIONS):
    fitness = []
    for ind in population:
        fit, _, _, _ = simulate_and_signals(ind, prices)
        fitness.append(fit)

    order = np.argsort(fitness)[::-1]
    population = [population[i] for i in order]
    fitness = [fitness[i] for i in order]

    if fitness[0] > best_fit:
        best_fit = fitness[0]
        best_ind = population[0]

    best_history.append(fitness[0])
    print(f"Gen {gen:02d} | best capital = {fitness[0]:.2f} | params = {population[0]}")

    # Selection: top half
    survivors = population[:POP_SIZE // 2]

    # Reproduce
    new_population = survivors.copy()
    while len(new_population) < POP_SIZE:
        a, b = np.random.choice(len(survivors), 2, replace=False)
        child = crossover(survivors[a], survivors[b])
        child = mutate(child)
        new_population.append(child)

    population = new_population

print("\nBest overall:", best_ind, "capital:", best_fit)


# -----------------------------
# 4) Prepare best strategy signals for animation
# -----------------------------
final_capital, ma, buys, sells = simulate_and_signals(best_ind, prices)

# For easier incremental drawing
buys_set = set(buys)
sells_set = set(sells)
buy_x, buy_y = [], []
sell_x, sell_y = [], []


# -----------------------------
# 5) Animation (price + MA + BUY/SELL)
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})
fig.suptitle("GA-evolved Trading Strategy: Price + Signals", fontsize=14)

# Top plot: price + MA + markers
line_price, = ax1.plot([], [], lw=1.8, label="Price")
line_ma, = ax1.plot([], [], lw=1.2, label="Moving Avg")
sc_buys = ax1.scatter([], [], marker="^", s=60, label="BUY")
sc_sells = ax1.scatter([], [], marker="v", s=60, label="SELL")

ax1.set_xlim(0, len(prices) - 1)
pad = 5
ax1.set_ylim(np.min(prices) - pad, np.max(prices) + pad)
ax1.set_xlabel("t")
ax1.set_ylabel("price")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.25)

# Bottom plot: best fitness history
line_fit, = ax2.plot(range(len(best_history)), best_history, lw=1.8)
ax2.set_title("Best capital per generation")
ax2.set_xlabel("generation")
ax2.set_ylabel("capital")
ax2.grid(True, alpha=0.25)

# Text box with parameters
param_text = ax1.text(
    0.01, 0.98, "",
    transform=ax1.transAxes,
    va="top",
    bbox=dict(boxstyle="round", alpha=0.15)
)

def init():
    line_price.set_data([], [])
    line_ma.set_data([], [])
    sc_buys.set_offsets(np.empty((0, 2)))
    sc_sells.set_offsets(np.empty((0, 2)))
    param_text.set_text("")
    buy_x.clear(); buy_y.clear()
    sell_x.clear(); sell_y.clear()
    return line_price, line_ma, sc_buys, sc_sells, param_text

def update(frame):
    # frame is time index on the price chart
    x = np.arange(frame + 1)
    y = prices[:frame + 1]
    y_ma = ma[:frame + 1]

    line_price.set_data(x, y)
    line_ma.set_data(x, y_ma)

    # Add markers when signals occur
    if frame in buys_set:
        buy_x.append(frame)
        buy_y.append(prices[frame])
    if frame in sells_set:
        sell_x.append(frame)
        sell_y.append(prices[frame])

    if buy_x:
        sc_buys.set_offsets(np.column_stack([buy_x, buy_y]))
    if sell_x:
        sc_sells.set_offsets(np.column_stack([sell_x, sell_y]))

    ma_window, buy_th, sell_th, stop_loss = best_ind
    param_text.set_text(
        f"Best params:\n"
        f"ma_window={int(ma_window)}\n"
        f"buy_th={buy_th*100:.2f}%\n"
        f"sell_th={sell_th*100:.2f}%\n"
        f"stop_loss={stop_loss*100:.1f}%\n"
        f"final_capital={final_capital:.2f}"
    )
    return line_price, line_ma, sc_buys, sc_sells, param_text

ani = FuncAnimation(
    fig,
    update,
    frames=len(prices),
    init_func=init,
    interval=20,     # ms; increase to slow down
    blit=False
)

plt.tight_layout()
plt.show()
