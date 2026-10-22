# Probability and Gradient Descent

A group project exploring probability, statistical modeling, and optimization through practical Python implementations. The project covers the Expectation-Maximization (EM) algorithm, Bayesian probability, and gradient descent, combining mathematical calculations with code-based verification and visualizations.

## Team Members

- **Annie Bwiza**
- **Ange Muhawenimana**
- **Ineza Ghislaine**

## Project Objectives

This project applies mathematical concepts to real datasets and simple prediction problems. It aims to:

- Implement the Expectation-Maximization algorithm to model two height populations.
- Apply Bayes' Theorem to estimate movie-review sentiment probabilities.
- Manually calculate gradient descent updates using matrix multiplication.
- Implement gradient descent in Python, verify gradients using SciPy, and visualize parameter and cost changes using Matplotlib.

## Project Structure

```text
probability-gradient-descent/
├── data/
│   ├── GaltonFamilies.csv
│   └── imdb_reviews.csv
├── src/
│   ├── em_algorithm.py
│   ├── bayesian_sentiment.py
│   └── gradient_descent.py
├── notebooks/
│   └── analysis.ipynb
├── outputs/
│   ├── error_over_iterations.png.pdf
│   └── parameter_changes.png
├── docs/
│   ├── manual_calculations_group_17.pdf
│   └── formative_3_team_tracker_group_17.pdf
└── README.md
```

---

## Part 1: Expectation-Maximization — Separating Height Populations

### Objective

Implement the Expectation-Maximization (EM) algorithm from scratch to fit a two-component Gaussian mixture model to pooled adult children's and mothers' heights, without using the group labels during training.

The goal is to estimate the parameters of both populations and calculate the probability that a given height belongs to each group.

### Dataset

The analysis uses Galton family height data. We combine:

- 934 adult children's heights.
- 205 mothers' heights, with one observation per family.

This produces **1,139 unlabelled height observations**. The true group labels are used only to evaluate the results, not to train the model.

### Model and Formulas

### Step 3: Core EM functions

- **`gaussian_pdf`** is the normal (bell-curve) density for a given mean and variance.

$$
\boxed{\text{density} = \frac{1}{\sqrt{2\pi\sigma^2}} \; e^{-\frac{(x - \text{mean})^2}{2\sigma^2}}}
$$

- **`weighted_densities`** computes the weighted density of each group for every height.

$$
\boxed{\text{weighted density} = \pi \times \text{density}}
$$

- **`log_likelihood`** measures how well the current model explains the data (higher is better).

$$
\boxed{\text{log-likelihood} = \sum_{\text{heights}} \log(\text{sum of weighted densities of all groups})}
$$

- **`e_step`** (**Expectation**) gives each height a *responsibility*: the probability it belongs to each group, using Bayes' theorem.

$$
\boxed{\text{responsibility} = \frac{\text{weighted density of x in this group}}{\text{sum of weighted densities that group}}}
$$

- **`m_step`** (**Maximization**) re-estimates each group's mean, variance and mixing proportion, using the responsibilities as weights.

$$
\boxed{\text{new mean} = \frac{\sum (\text{responsibility} \times x)}{\sum \text{responsibility}}}
$$

$$
\boxed{\text{new } \sigma^2 = \frac{\sum \big(\text{responsibility} \times (x - \text{new mean})^2\big)}{\sum \text{responsibility}}}
$$

$$
\boxed{\text{new } \pi = \frac{\sum \text{responsibility}}{\text{number of heights}}}
$$

Here $x$ is a height, $\sigma^2$ is a group's variance, and $\pi$ is a group's mixing proportion (the fraction of all heights that belong to it).
### Method

The algorithm starts with initial estimates for the means, variances, and mixing probabilities of the two Gaussian components. It then repeats two steps:

1. **E-step:** Calculate the responsibility of each component for every height observation.
2. **M-step:** Update the component means, variances, and mixing probabilities using the responsibilities as weights.

The iterations continue until the change in log-likelihood falls below $10^{-6}$.

The implementation includes functions for the E-step, M-step, log-likelihood calculation, and classification.

### Selected Results

| Iteration | μ₁ (Children) | μ₂ (Mothers) | σ₁² | σ₂² | π₁ | π₂ | Log-likelihood |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 — Initial | 69.0000 | 63.5000 | 12.5802 | 12.5802 | 0.5000 | 0.5000 | -3115.4014 |
| 1 | 68.2536 | 64.3037 | 9.8201 | 7.5712 | 0.4933 | 0.5067 | -3046.1412 |
| 2 | 68.3165 | 64.2587 | 10.0323 | 6.9517 | 0.4913 | 0.5087 | -3044.0230 |
| 386 — Converged | 69.7856 | 64.2993 | 5.4747 | 5.7935 | 0.3559 | 0.6441 | -3035.7484 |

The log-likelihood increases during the iterations, indicating that the algorithm is finding a better-fitting Gaussian mixture under the chosen model.

### Why Not Split at the Global Mean?

A simple split at the global mean assigns every observation to one group or the other. This ignores the overlap between the populations and does not provide probabilities or estimates of each group's variance.

EM instead estimates two overlapping distributions and assigns each observation a probability of belonging to either group.

However, the height distributions overlap substantially. In this dataset, the mean-split method achieved 57.6% accuracy, while EM achieved 51.3%. This illustrates an important limitation: maximizing likelihood does not guarantee the most accurate classification when the underlying groups are difficult to distinguish.

### Classifying a New Height

The classification function returns the posterior probabilities $\gamma_{k}$ for the two groups, using the same formula as the E-step.

For a height of 66.0 inches, the reported probabilities are:

- **P(Child) = 0.1647**
- **P(Mother) = 0.8353**

These probabilities show how the fitted model evaluates a new observation rather than assigning it to a group with absolute certainty.

### Implementation

File: `src/em_algorithm.py`

Run the implementation with:

```bash
python src/em_algorithm.py
```

---

## Part 2: Bayesian Probability — IMDb Sentiment Analysis

### Objective

Apply Bayes' Theorem to estimate the probability that a movie review is positive given that it contains a selected keyword.

We calculate:

$$
P(\text{Positive} \mid \text{Keyword})
$$

The implementation uses basic Python operations rather than machine-learning classification libraries.

### Dataset

The project uses the [IMDb Movie Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

The dataset contains 50,000 reviews:

- 25,000 positive reviews.
- 25,000 negative reviews.

The dataset is stored locally as `data/imdb_reviews.csv`.

### Selected Keywords

**Positive keywords**
- `excellent`
- `brilliant`
- `loved`

**Negative keywords**
- `terrible`
- `boring`
- `worst`

### Bayes' Theorem

The formula used is:

$$
P(\text{Positive} \mid \text{Keyword}) =
\frac{P(\text{Keyword} \mid \text{Positive}) \times P(\text{Positive})}
{P(\text{Keyword})}
$$

Where:

- **Prior:** $P(\text{Positive})$ — the probability that a randomly selected review is positive.
- **Likelihood:** $P(\text{Keyword} \mid \text{Positive})$ — the probability that a positive review contains the keyword.
- **Marginal:** $P(\text{Keyword})$ — the probability that a randomly selected review contains the keyword.
- **Posterior:** $P(\text{Positive} \mid \text{Keyword})$ — the probability that a review is positive given that it contains the keyword.

Because half of the dataset is positive:

$$
P(\text{Positive}) = \frac{25{,}000}{50{,}000} = 0.5
$$

The likelihood and marginal are estimated from counts:

$$
P(\text{Keyword} \mid \text{Positive}) = \frac{\text{positive reviews containing the keyword}}{\text{total positive reviews}}
$$

$$
P(\text{Keyword}) = \frac{\text{all reviews containing the keyword}}{\text{total reviews}}
$$

### Results

| Keyword | Prior | Likelihood | Marginal | Posterior |
|---|---:|---:|---:|---:|
| excellent | 0.5000 | 0.1147 | 0.0710 | 80.74% |
| brilliant | 0.5000 | 0.0635 | 0.0418 | 76.01% |
| loved | 0.5000 | 0.0765 | 0.0507 | 75.36% |
| terrible | 0.5000 | 0.0153 | 0.0540 | 14.18% |
| boring | 0.5000 | 0.0236 | 0.0610 | 19.37% |
| worst | 0.5000 | 0.0164 | 0.0887 | 9.27% |

*The values are rounded. Posterior percentages are calculated using the underlying counts and full-precision probabilities.*

### Worked Example: `excellent`

The program reports the following values:

- Prior: 0.5000
- Likelihood: 0.1147
- Marginal: 0.0710
- Posterior: 80.74%

Substituting into Bayes' Theorem:

$$
P(\text{Positive} \mid \text{excellent}) =
\frac{0.1147 \times 0.5}{0.0710} \approx 0.81
$$

The small difference from the reported 80.74% comes from rounding the inputs; the program uses the full-precision values.

This means that, within this dataset, reviews containing the word `excellent` are more likely to be positive than negative.

### Interpretation

Among the selected positive keywords, `excellent` has the highest posterior probability, followed by `brilliant` and `loved`.

The negative keywords have lower probabilities of a positive review, with `worst` producing the lowest posterior probability at 9.27%.

These results show that keyword occurrence provides useful evidence about sentiment. However, a single word cannot always determine the sentiment of a whole review because context, negation, and other words can change its meaning.

### Implementation

File: `src/bayesian_sentiment.py`

The program:

1. Loads the CSV dataset using `csv.DictReader()`.
2. Cleans review text by converting it to lowercase, removing HTML tags, and decoding HTML entities.
3. Identifies the selected keywords in each review.
4. Counts total reviews, positive reviews, reviews containing each keyword, and positive reviews containing each keyword.
5. Calculates the prior, likelihood, marginal, and posterior probabilities.
6. Displays the results for all six keywords.

### How to Run

Download the dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews), place it in the `data/` directory, and name it `imdb_reviews.csv`.

Run:

```bash
python src/bayesian_sentiment.py
```

---

## Part 3: Gradient Descent Manual Calculation

### Objective
Manually compute three updates of the gradient descent algorithm for the parameters `m` and `b` in a linear regression model, using matrix multiplication (not scalar operations).

### Model
y = m1x1 + m2x2 + b

### Given
- X = [(1, 3), (4, 10)]
- targets = [5, 6]
- Initial m = [-1, 2], Initial b = 1
- Learning rate (α) = 0.01
- n = 2 (number of samples)

### Gradient of the MSE Cost Function
J = (1/n) * ||y_hat - target||²
e = y_hat - target = X.m + b - target
∂J/∂m = (2/n) * Xᵀ.e
∂J/∂b = (2/n) * Σe
Since n = 2, 2/n = 1, so:
∇m = Xᵀ.e
∇b = Σe
### Iterations Summary

| Iteration | m | b | Error (e) |
|-----------|---|---|-----------|
| Start | [-1, 2] | 1 | — |
| 1 | [-1.45, 0.87] | 0.88 | [1, 11] |
| 2 | [-1.3316, 1.1808] | 0.9318 | [-2.96, -2.22] |
| 3 | [-1.3696, 1.0952] | 0.9362 | [-1.8574, 1.4134] |

Each iteration followed the same 4 steps: **Predict → Compute Error → Compute Gradient → Update m and b.** Full handwritten steps for each iteration are included in the manual calculation document.

### Observed Trend
Across the three iterations, the error steadily decreases in magnitude (from [1, 11] down to [-1.86, 1.41]), and `m` and `b` are converging toward values that reduce the MSE. This confirms gradient descent is moving the parameters in the correct direction to minimize the cost function.

---

## Part 4: Gradient Descent in Code

### Objective
Convert the Part 3 manual calculations into Python code, using SciPy to verify the gradient and Matplotlib to visualize the results.

### Approach
- **Analytical gradient**: implemented directly from the formulas derived in Part 3 (`∇m = Xᵀ.e`, `∇b = Σe`).
- **SciPy verification**: `scipy.optimize.approx_fprime` is used to numerically compute the derivative of the MSE cost function, confirming the analytical gradient is correct at each step.
- **Update loop**: each iteration explicitly shows Predict → Error → MSE → Gradient (analytical & SciPy) → Update, matching the manual calculation step-by-step (no hidden/abstracted "train" function).

### Output
Running the script reproduces the exact same values as the manual calculations:

| Iteration | m | b |
|-----------|---|---|
| 1 | [-1.45, 0.87] | 0.88 |
| 2 | [-1.3316, 1.1808] | 0.9318 |
| 3 | [-1.3696, 1.0952] | 0.9362 |

### Visualizations
- **`params_over_iterations.png`** — shows how m1, m2, and b change across the 3 iterations.
- **`error_over_iterations.png`** — shows the MSE dropping from 61.0 down to ~2.7 across the iterations, visually confirming convergence.


### Implementation

File: `src/gradient_descent.py`

Run:

```bash
python src/gradient_descent.py
```

The script prints the intermediate calculations and saves the visualizations to the output directory.

---

## Installation and Requirements

### Prerequisites

- Python 3
- pip
- The project datasets in the `data/` directory

### Install Dependencies

From the project root, run:

```bash
pip install -r requirements.txt
```

The gradient descent implementation uses SciPy for numerical gradient verification and Matplotlib for visualization. The EM and Bayesian implementations use basic Python and numerical calculations without relying on scikit-learn.

### Run the Individual Parts

```bash
python src/em_algorithm.py
python src/bayesian_sentiment.py
python src/gradient_descent.py
```

The manual gradient descent calculations are documented in `docs/manual_calculations_group_17.pdf`. The notebook in `notebooks/analysis.ipynb` can be used to explore the project interactively, provided the notebook contains the corresponding implementations.

---

## Conclusion

This project demonstrates how probability and optimization concepts can be applied through practical programming exercises.

- **Part 1:** EM fits a two-component Gaussian mixture and estimates group membership probabilities, while illustrating the challenges of overlapping distributions.
- **Part 2:** Bayes' Theorem estimates the probability of positive sentiment from selected keywords in IMDb reviews.
The manual calculations in Part 3 and the Python implementation in Part 4 produced identical results, confirming the correctness of our gradient descent derivation. The steadily decreasing error across iterations demonstrates that the model is successfully learning better parameters with each update.

Together, the four parts connect mathematical theory with implementation, numerical verification, and interpretation of results.

## Team Contributions

This project was completed collaboratively by:

- Annie Bwiza
- Ange Muhawenimana
- Ineza Ghislaine

See `docs/formative_3_team_tracker_group_17.pdf` for the detailed breakdown of each member's contributions.

## Dataset Attribution

- **Galton family heights:** Galton family height data used for the Gaussian mixture analysis. [Galton family heights Dataset on kaggle](https://www.kaggle.com/datasets/jacopoferretti/parents-heights-vs-children-heights-galton-data)

- **IMDb Movie Reviews:** [IMDb Movie Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

Please refer to the original dataset sources for licensing and usage conditions.
