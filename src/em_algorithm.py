# -*- coding: utf-8 -*-

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
PARENT_COLUMN = "mother"                          # "mother" or "father"
PARENT_LABEL = PARENT_COLUMN.capitalize() + "s"   # "Mothers" or "Fathers", used in output
TEST_HEIGHT = 66.0                                # the height to classify in Step 9
TOLERANCE = 1e-6                                  # stop when log-likelihood changes less than this
MAX_ITERATIONS = 500
MIN_VARIANCE = 1e-6                               # keeps a variance from collapsing to zero

# On average, mothers are shorter than the children in this dataset and fathers
# are taller. This only decides which starting mean is called "Children";
# EM itself never sees the labels.
CHILDREN_ARE_TALLER = PARENT_COLUMN == "mother"


def find_data_file(name="GaltonFamilies.csv"):
    """Find the CSV whether this runs as a script (data/ folder) or in a notebook."""
    try:
        base = Path(__file__).resolve().parent
        candidates = [base.parent / "data" / name, base / "data" / name, base / name]
    except NameError:                                  # __file__ does not exist in notebooks
        candidates = []
    candidates += [Path("data") / name, Path(name)]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {name}. Put it in a data/ folder or next to this file.")


DATA_PATH = find_data_file()


# ---------------------------------------------------------------------------
# Step 1: Load the dataset
# ---------------------------------------------------------------------------
def load_heights(path, parent_column):
    """Return pooled heights plus the hidden true labels (1 = child, 0 = parent)."""
    df = pd.read_csv(path)
    parents = df.drop_duplicates("family")[parent_column].to_numpy()   # one parent per family
    children = df["childHeight"].to_numpy()
    heights = np.concatenate([children, parents])
    true_labels = np.concatenate([np.ones(len(children)), np.zeros(len(parents))])
    return heights, true_labels, children, parents


heights, true_labels, children, parents = load_heights(DATA_PATH, PARENT_COLUMN)
print(f"Children: {len(children)}  mean={children.mean():.2f}  var={children.var():.2f}")
print(f"{PARENT_LABEL:<8}: {len(parents)}  mean={parents.mean():.2f}  var={parents.var():.2f}")
print(f"Pooled  : {len(heights)} heights (labels hidden from EM)")

# ---------------------------------------------------------------------------
# Step 2: Look at the pooled data
# ---------------------------------------------------------------------------
plt.hist(heights, bins=40, color="grey", edgecolor="white")
plt.title("Pooled heights – what EM sees (no labels)")
plt.xlabel("Height (inches)")
plt.ylabel("Count")
plt.show()


# ---------------------------------------------------------------------------
# Step 3: Core EM functions
# ---------------------------------------------------------------------------
def gaussian_pdf(x, mean, variance):
    return np.exp(-(x - mean) ** 2 / (2 * variance)) / np.sqrt(2 * np.pi * variance)


def weighted_densities(x, params):
    """Row k holds pi_k * N(x | mu_k, var_k) for every point."""
    return np.vstack([
        params["pi"][k] * gaussian_pdf(x, params["mu"][k], params["var"][k])
        for k in range(len(params["mu"]))
    ])


def log_likelihood(x, params):
    return np.sum(np.log(weighted_densities(x, params).sum(axis=0)))


def e_step(x, params):
    """Responsibilities: probability each point belongs to each component."""
    weighted = weighted_densities(x, params)
    return weighted / weighted.sum(axis=0)


def m_step(x, resp):
    """Re-estimate parameters using responsibilities as soft weights."""
    Nk = resp.sum(axis=1)
    mu = (resp * x).sum(axis=1) / Nk
    var = (resp * (x - mu[:, None]) ** 2).sum(axis=1) / Nk
    var = np.maximum(var, MIN_VARIANCE)
    pi = Nk / len(x)
    return {"mu": mu, "var": var, "pi": pi}


# ---------------------------------------------------------------------------
# Step 4: Initialise the parameters
# ---------------------------------------------------------------------------
def initialize(x):
    """Component 0 = children, component 1 = parents."""
    high, low = np.percentile(x, 75), np.percentile(x, 25)
    mu = [high, low] if CHILDREN_ARE_TALLER else [low, high]
    return {
        "mu": np.array(mu),
        "var": np.array([x.var(), x.var()]),
        "pi": np.array([0.5, 0.5]),
    }


# ---------------------------------------------------------------------------
# Step 5: Run EM until convergence
# ---------------------------------------------------------------------------
def record(iteration, params, x):
    return {
        "Iteration": iteration,
        "μ1 (Children)": params["mu"][0], f"μ2 ({PARENT_LABEL})": params["mu"][1],
        "σ1²": params["var"][0], "σ2²": params["var"][1],
        "π1": params["pi"][0], "π2": params["pi"][1],
        "Log-Likelihood": log_likelihood(x, params),
    }


def run_em(x, tol=TOLERANCE, max_iter=MAX_ITERATIONS):
    params = initialize(x)
    history = [record(0, params, x)]
    for it in range(1, max_iter + 1):
        resp = e_step(x, params)
        params = m_step(x, resp)
        history.append(record(it, params, x))
        if abs(history[-1]["Log-Likelihood"] - history[-2]["Log-Likelihood"]) < tol:
            break
    return params, pd.DataFrame(history)


final_params, history = run_em(heights)
print(f"Converged after {history['Iteration'].iloc[-1]} iterations")

# ---------------------------------------------------------------------------
# Step 6: Tracking table for iterations 0-2
# ---------------------------------------------------------------------------
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.width", 200)
print("\nTracking table (iterations 0-2):")
print(history.head(3).to_string(index=False))

# ---------------------------------------------------------------------------
# Step 7: Final parameters
# ---------------------------------------------------------------------------
print("\nFinal parameters:")
print(history.tail(1).to_string(index=False))

# ---------------------------------------------------------------------------
# Step 8: Check convergence
# ---------------------------------------------------------------------------
plt.plot(history["Iteration"], history["Log-Likelihood"], marker=".")
plt.title("Log-likelihood never decreases and flattens at convergence")
plt.xlabel("Iteration")
plt.ylabel("Log-likelihood")
plt.show()


# ---------------------------------------------------------------------------
# Step 9: Classify a new height
# ---------------------------------------------------------------------------
def classify(height, params):
    resp = e_step(np.array([height], dtype=float), params)[:, 0]
    print(f"\nHeight {height} in")
    print(f"  P({'Child':<6} | height) = {resp[0]:.4f}")
    print(f"  P({PARENT_COLUMN.capitalize():<6} | height) = {resp[1]:.4f}")
    return resp


classify(TEST_HEIGHT, final_params)

# ---------------------------------------------------------------------------
# Step 10: Visualise the fitted model
# ---------------------------------------------------------------------------
grid = np.linspace(heights.min() - 2, heights.max() + 2, 400)
plt.hist(heights, bins=40, density=True, color="lightgrey", edgecolor="white")
for k, name in enumerate(["Children", PARENT_LABEL]):
    curve = final_params["pi"][k] * gaussian_pdf(grid, final_params["mu"][k], final_params["var"][k])
    plt.plot(grid, curve, label=name)
plt.plot(grid, weighted_densities(grid, final_params).sum(axis=0), "k--", label="Mixture")
plt.legend()
plt.xlabel("Height (inches)")
plt.ylabel("Density")
plt.title("EM fit")
plt.show()


# ---------------------------------------------------------------------------
# Step 11: Compare with a simple global-mean split
# ---------------------------------------------------------------------------
def mean_split(x):
    cut = x.mean()
    upper, lower = x[x >= cut], x[x < cut]
    return cut, upper.mean(), lower.mean()


def accuracy(predicted_child, truth):
    return np.mean(predicted_child == truth)


cut, upper_mean, lower_mean = mean_split(heights)
print(f"\nCut at {cut:.2f}: upper pile mean = {upper_mean:.2f}, lower pile mean = {lower_mean:.2f}")
print(f"True means: children = {children.mean():.2f}, {PARENT_LABEL.lower()} = {parents.mean():.2f}")

# The naive split calls the taller pile "children" only if children are the taller group
above_cut = heights >= cut
naive_pred = (above_cut if CHILDREN_ARE_TALLER else ~above_cut).astype(float)
em_pred = (e_step(heights, final_params)[0] >= 0.5).astype(float)
print(f"Accuracy – mean split: {accuracy(naive_pred, true_labels):.3f}")
print(f"Accuracy – EM        : {accuracy(em_pred, true_labels):.3f}")
