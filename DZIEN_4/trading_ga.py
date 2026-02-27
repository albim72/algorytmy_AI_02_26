import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------
# 1. Synthetic market data
# ----------------------------------

np.random.seed(42)

def generate_price(n=500):
    trend = np.linspace(0, 10, n)
    noise = np.random.normal(0, 2, n)
    return 100 + trend + noise

prices = generate_price()


# ----------------------------------
# 2. Strategy simulation
# ----------------------------------

def simulate_strategy(params, prices):
    ma_window, buy_th, sell_th, stop_loss = params
    ma_window = int(ma_window)

    capital = 1000
    position = 0
    entry_price = 0

    for i in range(ma_window, len(prices)):
        ma = np.mean(prices[i-ma_window:i])
        price = prices[i]

        # Buy
        if position == 0 and price > ma * (1 + buy_th):
            position = capital / price
            entry_price = price
            capital = 0

        # Sell
        elif position > 0:
            if price < ma * (1 - sell_th) or price < entry_price * (1 - stop_loss):
                capital = position * price
                position = 0

    if position > 0:
        capital = position * prices[-1]

    return capital


# ----------------------------------
# 3. Genetic Algorithm
# ----------------------------------

POP_SIZE = 30
GENERATIONS = 25
MUTATION_RATE = 0.2

def random_individual():
    return [
        np.random.randint(5, 40),      # ma_window
        np.random.uniform(0.001, 0.02), # buy_threshold
        np.random.uniform(0.001, 0.02), # sell_threshold
        np.random.uniform(0.01, 0.1),   # stop_loss
    ]

def mutate(ind):
    if np.random.rand() < MUTATION_RATE:
        ind[0] = np.random.randint(5, 40)
    if np.random.rand() < MUTATION_RATE:
        ind[1] = np.random.uniform(0.001, 0.02)
    if np.random.rand() < MUTATION_RATE:
        ind[2] = np.random.uniform(0.001, 0.02)
    if np.random.rand() < MUTATION_RATE:
        ind[3] = np.random.uniform(0.01, 0.1)
    return ind

def crossover(p1, p2):
    point = np.random.randint(1, 4)
    return p1[:point] + p2[point:]

population = [random_individual() for _ in range(POP_SIZE)]

best_history = []

for gen in range(GENERATIONS):

    fitness = [simulate_strategy(ind, prices) for ind in population]
    sorted_pop = [x for _, x in sorted(zip(fitness, population), reverse=True)]

    best = sorted_pop[0]
    best_capital = simulate_strategy(best, prices)
    best_history.append(best_capital)

    print(f"Gen {gen}: Best capital = {best_capital:.2f}")

    # Selection: top 50%
    survivors = sorted_pop[:POP_SIZE // 2]

    # Reproduction
    new_population = survivors.copy()
    while len(new_population) < POP_SIZE:
        p1, p2 = np.random.choice(len(survivors), 2, replace=False)
        child = crossover(survivors[p1], survivors[p2])
        child = mutate(child)
        new_population.append(child)

    population = new_population

# ----------------------------------
# 4. Plot evolution
# ----------------------------------

plt.plot(best_history)
plt.title("Evolution of Best Capital")
plt.xlabel("Generation")
plt.ylabel("Capital")
plt.show()
