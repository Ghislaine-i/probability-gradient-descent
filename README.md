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
│   └── formative_3_tam_tracker_group_17.pdf
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

## Part 3: Manual Gradient Descent Calculations

### Objective

Manually calculate three gradient descent updates for a simple linear regression model, using matrix multiplication to compute the gradients.

### Model and Initial Values

The model is:

$$
\hat{y} = m_1 x_1 + m_2 x_2 + b
$$

Given:

$$
X = \begin{bmatrix} 1 & 3 \\ 4 & 10 \end{bmatrix},
\qquad
y = \begin{bmatrix} 5 \\ 6 \end{bmatrix}
$$

Initial parameters:

- $m = [-1, 2]$
- $b = 1$
- Learning rate: $\alpha = 0.01$
- Number of observations: $n = 2$

### Cost Function and Gradients

The Mean Squared Error (MSE) is:

$$
J = \frac{1}{n} \lVert \hat{y} - y \rVert^2
$$

Define the error vector:

$$
e = \hat{y} - y = Xm + b\mathbf{1} - y
$$

The gradients are:

$$
\nabla_m J = \frac{2}{n} X^T e
$$

$$
\frac{\partial J}{\partial b} = \frac{2}{n} \sum_i e_i
$$

Since $n = 2$, the factor $2/n = 1$, giving:

$$
\nabla_m J = X^T e,
\qquad
\frac{\partial J}{\partial b} = \sum_i e_i
$$

The parameters are updated using:

$$
m_{\text{new}} = m_{\text{old}} - \alpha \, \nabla_m J
$$

$$
b_{\text{new}} = b_{\text{old}} - \alpha \, \frac{\partial J}{\partial b}
$$

### Worked Example: Update 1

**Predict:**

$$
\hat{y} = Xm + b\mathbf{1} =
\begin{bmatrix} 1 & 3 \\ 4 & 10 \end{bmatrix}
\begin{bmatrix} -1 \\ 2 \end{bmatrix} + 1
= \begin{bmatrix} 6 \\ 17 \end{bmatrix}
$$

**Error:**

$$
e = \hat{y} - y = \begin{bmatrix} 6 - 5 \\ 17 - 6 \end{bmatrix} = \begin{bmatrix} 1 \\ 11 \end{bmatrix}
$$

**MSE:**

$$
J = \frac{1^2 + 11^2}{2} = 61
$$

**Gradients:**

$$
\nabla_m J = X^T e =
\begin{bmatrix} 1 & 4 \\ 3 & 10 \end{bmatrix}
\begin{bmatrix} 1 \\ 11 \end{bmatrix}
= \begin{bmatrix} 45 \\ 113 \end{bmatrix},
\qquad
\frac{\partial J}{\partial b} = 1 + 11 = 12
$$

**Update:**

$$
m_{\text{new}} = \begin{bmatrix} -1 \\ 2 \end{bmatrix} - 0.01 \begin{bmatrix} 45 \\ 113 \end{bmatrix} = \begin{bmatrix} -1.45 \\ 0.87 \end{bmatrix},
\qquad
b_{\text{new}} = 1 - 0.01(12) = 0.88
$$

### Iteration Summary

| Stage | m | b |
|---|---|---:|
| Initial | [-1, 2] | 1.0000 |
| Update 1 | [-1.4500, 0.8700] | 0.8800 |
| Update 2 | [-1.3316, 1.1808] | 0.9318 |
| Update 3 | [-1.3696, 1.0952] | 0.9362 |

Each update follows the same sequence:

**Predict → Compute Error → Calculate MSE → Compute Gradients → Update Parameters**

The full step-by-step calculations are documented in `docs/manual_calculations_group_17.pdf`.

### Observed Trend

The cost decreases substantially after the first update and continues to decrease over the next updates. The parameter values change according to the errors and gradients calculated at each step.

This demonstrates how gradient descent adjusts model parameters to reduce prediction error. Three updates illustrate the process but do not, by themselves, establish that the algorithm has reached its minimum.

---

## Part 4: Gradient Descent in Python

### Objective

Implement the manual gradient descent calculations in Python, verify the analytical gradients using SciPy, and visualize how the parameters and cost change over three updates.

### Approach

**1. Analytical gradient**

The gradients are calculated directly from the formulas derived in Part 3:

$$
\nabla_m J = X^T e,
\qquad
\frac{\partial J}{\partial b} = \sum_i e_i
$$

**2. Numerical verification with SciPy**

The implementation uses `scipy.optimize.approx_fprime` to approximate the gradient numerically using finite differences:

$$
\frac{\partial J}{\partial \theta_j} \approx \frac{J(\theta + \varepsilon \, \mathbf{e}_j) - J(\theta)}{\varepsilon}
$$

The numerical results are compared with the analytical gradients to check their correctness, allowing for small numerical differences.

**3. Explicit update loop**

Each iteration displays the prediction, error, MSE, analytical gradient, SciPy gradient, and updated parameters. This makes it possible to follow the algorithm step by step without hiding the calculations inside a training function.

### Parameter Results

| Iteration | m | b |
|---|---|---:|
| 1 | [-1.4500, 0.8700] | 0.8800 |
| 2 | [-1.3316, 1.1808] | 0.9318 |
| 3 | [-1.3696, 1.0952] | 0.9362 |

The Python implementation reproduces the manual parameter updates to the displayed precision.

### Visualizations

The script generates the following figures in `outputs/figures/`:

- **`parameter_changes.png`** — shows how $m_1$, $m_2$, and $b$ change over the three updates.
- **`error_over_iterations.png`** — shows the change in MSE during gradient descent.

The initial MSE is 61.0. It decreases to approximately 6.845 after the first update, 2.724 after the second update, and 2.391 after the third update.

These results show a decrease in the cost over the three updates, indicating that the chosen parameter updates are improving the model's fit on this dataset.

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
- **Part 3:** Manual gradient descent calculations demonstrate how matrix multiplication is used to compute gradients and update parameters.
- **Part 4:** Python reproduces the gradient descent updates, SciPy provides a numerical check of the analytical gradients, and Matplotlib visualizes changes in the parameters and cost.

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
