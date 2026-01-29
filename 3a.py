import numpy as np

def experiment(n_runs=100000, n_coins=1000, n_flips=10):
    rng = np.random.default_rng()

    v1 = np.empty(n_runs)
    vrand = np.empty(n_runs)
    vmin = np.empty(n_runs)

    for r in range(n_runs):
        flips = rng.integers(0, 2, size=(n_coins, n_flips))
        heads = flips.sum(axis=1)
        v = heads / n_flips

        v1[r] = v[0]

        crand = rng.integers(0, n_coins)
        vrand[r] = v[crand]

        cmin = int(np.argmin(v))
        vmin[r] = v[cmin]

    return v1.mean(), vrand.mean(), vmin.mean()

if __name__ == "__main__":
    mean_v1, mean_vrand, mean_vmin = experiment(n_runs=100000, n_coins=1000, n_flips=10)
    print("Average v1    =", mean_v1)
    print("Average vrand =", mean_vrand)
    print("Average vmin  =", mean_vmin)
