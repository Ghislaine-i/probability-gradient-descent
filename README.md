# Probability and Gradient Descent

## Project Overview

This project explores probability and optimization through practical Python implementations. It covers the Expectation-Maximization (EM) algorithm, Bayesian probability, and gradient descent.

## Project Structure

```text
probability-gradient-descent/
├── data/
│   ├── heights.csv
│   └── imdb_reviews.csv
├── src/
│   ├── em_algorithm.py
│   ├── bayesian_sentiment.py
│   ├── manual_calculations.py
│   └── gradient_descent.py
├── notebooks/
│   └── analysis.ipynb
├── outputs/
│   └── figures/
├── docs/
│   ├── manual_calculations.pdf
│   └── contributions.pdf
├── README.md
└── requirements.txt
```

*Note: The structure above shows the intended project layout. Some files may still be under development.*

---

## Part 2: Bayesian Probability — IMDb Sentiment Analysis

### 1. Objective

Apply Bayes' Theorem to estimate the probability that an IMDb movie review is positive given that it contains a selected keyword.

### 2. Dataset

We use the [IMDb Movie Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).

The dataset contains 50,000 movie reviews:
- 25,000 positive reviews
- 25,000 negative reviews

The dataset is stored locally as `data/imdb_reviews.csv`.

### 3. Keyword Selection

We selected six keywords based on their expected association with positive or negative sentiment.

**Positive keywords**
- `excellent`
- `brilliant`
- `loved`

**Negative keywords**
- `terrible`
- `boring`
- `worst`

### 4. Choice of Conditional Probability

We calculate only the posterior probability:

**P(Positive | keyword)**

This represents the probability that a review is positive given that it contains a particular keyword. We do not calculate the negative posterior probability.

### 5. Bayes' Theorem

Bayes' Theorem is expressed as:

P(Positive | keyword) =  
[P(keyword | Positive) × P(Positive)] / P(keyword)

The probabilities used are:

- **Prior:** P(Positive), the probability that a randomly selected review is positive.
- **Likelihood:** P(keyword | Positive), the probability that a positive review contains the selected keyword.
- **Marginal:** P(keyword), the probability that a randomly selected review contains the selected keyword.
- **Posterior:** P(Positive | keyword), the probability that a review is positive given that it contains the selected keyword.

Because half of the dataset consists of positive reviews, the prior probability is:

P(Positive) = 25,000 / 50,000 = 0.5

### 6. Results

The following probabilities were produced by running `src/bayesian_sentiment.py` on the dataset.

| Keyword | Prior P(Positive) | Likelihood P(keyword \| Positive) | Marginal P(keyword) | Posterior P(Positive \| keyword) |
|---|---:|---:|---:|---:|
| excellent | 0.5000 | 0.1147 | 0.0710 | 80.74% |
| brilliant | 0.5000 | 0.0635 | 0.0418 | 76.01% |
| loved | 0.5000 | 0.0765 | 0.0507 | 75.36% |
| terrible | 0.5000 | 0.0153 | 0.0540 | 14.18% |
| boring | 0.5000 | 0.0236 | 0.0610 | 19.37% |
| worst | 0.5000 | 0.0164 | 0.0887 | 9.27% |

The table displays rounded values. The posterior percentages are calculated by the Python program using the underlying counts and full-precision probabilities.

### 7. Worked Example: The Keyword `excellent`

From the program's output:

- Prior: P(Positive) = 0.5000
- Likelihood: P(excellent | Positive) = 0.1147
- Marginal: P(excellent) = 0.0710
- Posterior reported by the program: 80.74%

Applying Bayes' Theorem:

P(Positive | excellent)  
= [P(excellent | Positive) × P(Positive)] / P(excellent)

Using the unrounded probabilities calculated by the program, the posterior is approximately **80.74%**.

This means that a review containing the word `excellent` has an estimated 80.74% probability of being positive in this dataset.

### 8. Interpretation

The three selected positive keywords have relatively high posterior probabilities of a positive review. The keyword `excellent` has the highest posterior probability at 80.74%, followed by `brilliant` at 76.01% and `loved` at 75.36%.

The negative keywords have lower posterior probabilities of a positive review. Among them, `worst` has the lowest posterior probability at 9.27%, followed by `terrible` at 14.18% and `boring` at 19.37%.

These results suggest that keyword occurrence provides useful evidence about review sentiment. However, a single keyword does not necessarily determine the sentiment of an entire review because context, negation, and other words can affect meaning.

### 9. Python Implementation

The implementation is located in:

`src/bayesian_sentiment.py`

The program uses Python's standard library and basic operations rather than external machine learning libraries.

The main steps are:

1. **Load the dataset:** Read the review text and sentiment labels from the CSV file using `csv.DictReader()`.
2. **Clean the text:** Remove HTML tags, convert text to lowercase, and decode HTML entities.
3. **Identify keywords:** Extract words from each review and check whether each selected keyword occurs.
4. **Count occurrences:** Count the total reviews, positive reviews, reviews containing each keyword, and positive reviews containing each keyword.
5. **Calculate probabilities:** Compute the prior, likelihood, marginal, and posterior using basic arithmetic.
6. **Display results:** Print the probability values and posterior percentages for all six keywords.

### 10. How to Run the Bayesian Analysis

**Prerequisites:** Python 3 installed on your computer.

1. Clone or download the project repository.
2. Download the IMDb dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews).
3. Place the CSV file in the `data/` folder and name it `imdb_reviews.csv`.
4. Open a terminal at the project root.
5. Run:

```bash
python src/bayesian_sentiment.py
```

The program displays the dataset counts and probability calculations for all six keywords.

---

## Other Project Components

The repository also includes modules for the EM algorithm, manual gradient descent calculations, and gradient descent implementation. Refer to the corresponding Python files, notebook, and documentation for those parts of the project.

## Team Contributions

Refer to `docs/contributions.pdf` for the group's contribution breakdown.

## Dataset Attribution

The IMDb Movie Reviews dataset is provided through Kaggle. Refer to the dataset page for its license and usage conditions:

https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
