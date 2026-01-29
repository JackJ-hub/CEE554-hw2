import numpy as np

def run_once(N=1000, noise_rate=0.1, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    X = rng.uniform(-1.0, 1.0, size=(N, 2))
    x1, x2 = X[:, 0], X[:, 1]

    y = sign(x1**2 + x2**2 - 0.6)

    n_flip = int(noise_rate * N)
    flip_idx = rng.choice(N, size=n_flip)
    y[flip_idx] *= -1

    xb = np.c_[np.ones(N), x1, x2]
    w = np.linalg.pinv(xb) @ y 
    y_pred = sign(xb @ w)
    Ein = np.mean(y_pred != y)

    return w, Ein

def sign(z):
    return np.where(z >= 0, 1, -1)

def experiment(n_runs=1000, N=1000, noise_rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Eins = np.empty(n_runs)

    for r in range(n_runs):
        w, Ein = run_once(N=N, noise_rate=noise_rate, rng=rng)
        Eins[r] = Ein

    return Eins.mean()

"""
# Different each run (good): one RNG, reused inside the loop
rng = np.random.default_rng(0)
for r in range(n_runs):
    x = rng.uniform()
    idx = rng.choice(10)

# Same every run (bad): RNG is re-created each loop iteration
for r in range(n_runs):
    rng = np.random.default_rng(0)
    x = rng.uniform()
"""
if __name__ == "__main__":
    avg_Ein = experiment(n_runs=1000, N=1000, noise_rate=0.1, seed=0)
    print("Average Ein over 1000 runs =", avg_Ein)
