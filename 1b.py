import numpy as np
from pathlib import Path

def pla_with_init(x, y, w_init, max_iter=1000):
    N = x.shape[0]
    xb = np.c_[np.ones(N), x]
    w = np.array(w_init, dtype=float).copy()
    updates = 0
    for _ in range(max_iter):
        changed = False
        for i in range(N):
            if y[i] * (w @ xb[i]) <= 0:
                w += y[i] * xb[i]
                updates += 1
                changed = True
        if not changed:
            break
    return w, updates

def main():
    path = Path("/Users/jack/Desktop/CEE554/HW1/Bridge_Condition.txt")
    data = np.loadtxt(path)
    datasetA = data[:20, :]
    x1 = datasetA[:, 0:2]
    y1 = datasetA[:, 2].astype(int)

    w_lr = np.array([-2.06799143, 0.17494643, 0.20105202], dtype=float)
    w_final_lr, updates_lr = pla_with_init(x1, y1, w_lr)

    rng = np.random.default_rng(0)
    w_rand = rng.standard_normal(3)
    w_final_rand, updates_rand = pla_with_init(x1, y1, w_rand)
    print("Init = Linear Regression:", w_lr)
    print("PLA updates to converge:", updates_lr)
    print("Final w:", w_final_lr)
    print()
    print("Init = Random (seed=0):", w_rand)
    print("PLA updates to converge:", updates_rand)
    print("Final w:", w_final_rand)

if __name__ == "__main__":
    main()
