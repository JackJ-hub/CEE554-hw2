import numpy as np
import matplotlib.pyplot as plt

def linear_regression(x, y):
    N = x.shape[0]
    xb = np.c_[np.ones(N), x]
    w = np.linalg.pinv(xb) @ y 
    y_pred = np.where((xb @ w) >= 0, 1, -1)
    Ein = np.mean(y_pred != y)
    return w, Ein

def plot_scatter_with_boundary(x, y, w, title="Linear Regression Classification (Dataset A)"):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x[y == 1, 0], x[y == 1, 1], c="red", marker="o", label="y=+1")
    ax.scatter(x[y == -1, 0], x[y == -1, 1], c="green", marker="x", label="y=-1")

    w0, w1, w2 = w
    xs = np.linspace(x[:, 0].min(), x[:, 0].max(), 200)

    if abs(w2) > 1e-12:
        ys = -(w0 + w1 * xs) / w2
        ax.plot(xs, ys, linewidth=2, label="decision boundary")
    elif abs(w1) > 1e-12:
        ax.axvline(-w0 / w1, linewidth=2, label="decision boundary")
    else:
        ax.text(0.5, 0.5, "Degenerate boundary (w1≈0 and w2≈0)", transform=ax.transAxes,
                ha="center", va="center")

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    plt.show()
def main():
    PATH = "/Users/jack/Desktop/CEE554/HW1/Bridge_Condition.txt"
    data = np.loadtxt(PATH)
    dataset1 = data[0:20, :]
    x1 = dataset1[:, 0:2]
    y1 = dataset1[:, 2]
    w, Ein = linear_regression(x1, y1)
    print(f"Linear Regression Weights: {w}, In-sample Error: {Ein}")

    plot_scatter_with_boundary(x1, y1, w)

if __name__ == "__main__":
        main()
