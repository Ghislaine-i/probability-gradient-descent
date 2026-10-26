"""
Part 4: Convert manual gradient descent calculations into Python code.

Model:  y_hat = X . m + b
Cost:   J = (1/n) * sum((y_hat - target)^2)      (MSE)
Update: m = m - lr * grad_m,  b = b - lr * grad_b

The analytical gradient is verified at every step with SciPy's
numerical derivative (scipy.optimize.approx_fprime).
"""

import numpy as np
from scipy.optimize import approx_fprime
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# Step 1: Define the dataset and starting values
# ---------------------------------------------------------------
X = np.array([[1, 3],
              [4, 10]], dtype=float)
target = np.array([5, 6], dtype=float)

m = np.array([-1.0, 2.0])   # initial weights
b = 1.0                     # initial bias
lr = 0.01                   # learning rate
n = X.shape[0]              # number of samples
n_iterations = 3


# ---------------------------------------------------------------
# Step 2: Prediction, cost and gradient functions
# ---------------------------------------------------------------
def predict(m, b, X):
    """y_hat = X . m + b   (matrix multiplication, not element-by-element)."""
    return X.dot(m) + b


def cost_J(params, X, target):
    """MSE cost: J = (1/n) * sum((y_hat - target)^2)."""
    m_ = params[:2]
    b_ = params[2]
    error = predict(m_, b_, X) - target
    return np.mean(error ** 2)


def analytical_gradient(m, b, X, target):
    """Gradient derived by hand in Part 3."""
    y_hat = predict(m, b, X)
    error = y_hat - target
    grad_m = (2 / n) * X.T.dot(error)
    grad_b = (2 / n) * np.sum(error)
    return grad_m, grad_b, error, y_hat


def scipy_gradient(m, b, X, target, epsilon=1e-6):
    """Numerical gradient from SciPy, used to check the analytical one."""
    params = np.array([m[0], m[1], b])
    grad = approx_fprime(params, cost_J, epsilon, X, target)
    return grad[:2], grad[2]


# ---------------------------------------------------------------
# Step 3: Gradient descent loop (Predict -> Error -> MSE -> Gradient -> Update)
# ---------------------------------------------------------------
history_m = [m.copy()]
history_b = [b]
history_error = []

print("=" * 70)
print("GRADIENT DESCENT - PART 4 (Python reproduction of manual Part 3)")
print("=" * 70)

for i in range(1, n_iterations + 1):
    print(f"\n--- Iteration {i} ---")
    print(f"Current m = {m}, b = {b}")

    # 1. Predict
    y_hat = predict(m, b, X)
    print(f"Prediction y_hat = X . m + b = {y_hat}")

    # 2. Error
    error = y_hat - target
    print(f"Error e = y_hat - target = {error}")

    # 3. MSE cost (kept for the error-vs-iteration plot)
    mse = np.mean(error ** 2)
    history_error.append(mse)
    print(f"MSE cost J = {mse:.4f}")

    # 4. Analytical gradient
    grad_m, grad_b, _, _ = analytical_gradient(m, b, X, target)
    print(f"Analytical gradient  -> grad_m = {grad_m}, grad_b = {grad_b}")

    # 5. SciPy numerical gradient (sanity check)
    grad_m_scipy, grad_b_scipy = scipy_gradient(m, b, X, target)
    print(f"SciPy numerical grad -> grad_m = {np.round(grad_m_scipy, 4)}, "
          f"grad_b = {grad_b_scipy:.4f}  (should match analytical)")

    # 6. Update m and b
    m = m - lr * grad_m
    b = b - lr * grad_b
    print(f"Updated  -> m = {np.round(m, 4)}, b = {round(b, 4)}")

    history_m.append(m.copy())
    history_b.append(b)

# Final MSE after the last update (completes the error curve)
final_pred = predict(m, b, X)
final_error = final_pred - target
history_error.append(np.mean(final_error ** 2))

print("\n" + "=" * 70)
print(f"FINAL PARAMETERS after {n_iterations} iterations:")
print(f"m = {np.round(m, 4)}")
print(f"b = {round(b, 4)}")
print(f"Final predictions y_hat = {np.round(final_pred, 4)}")
print("=" * 70)


# ---------------------------------------------------------------
# Step 4: Visualizations
# ---------------------------------------------------------------
history_m = np.array(history_m)
iterations = np.arange(len(history_b))   # 0, 1, 2, ..., n_iterations

# Plot 1: how m1, m2 and b change
plt.figure(figsize=(8, 5))
plt.plot(iterations, history_m[:, 0], marker="o", label="m1")
plt.plot(iterations, history_m[:, 1], marker="o", label="m2")
plt.plot(iterations, history_b, marker="o", label="b")
plt.xlabel("Iteration")
plt.ylabel("Parameter value")
plt.title("Change in m and b over Gradient Descent Iterations")
plt.xticks(iterations)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("params_over_iterations.png", dpi=150)
plt.show()
plt.close()

# Plot 2: MSE over iterations
plt.figure(figsize=(8, 5))
plt.plot(iterations, history_error, marker="o", color="crimson")
plt.xlabel("Iteration")
plt.ylabel("Mean Squared Error (J)")
plt.title("Error (MSE) over Gradient Descent Iterations")
plt.xticks(iterations)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("error_over_iterations.png", dpi=150)
plt.show()
plt.close()
