
## Part 2: Bayesian Probability — IMDb Sentiment Analysis

### Objective
Apply Bayes' Theorem to estimate the probability that an IMDb movie review is positive given that it contains a selected keyword.

### Dataset
IMDb Movie Reviews Dataset: https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

The analysis uses 50,000 reviews: 25,000 positive and 25,000 negative.

### Selected Keywords
- Positive keywords: excellent, brilliant, loved
- Negative keywords: terrible, boring, worst

### Method
We calculate the posterior probability P(Positive | keyword) for each keyword using Bayes' Theorem:

P(Positive | keyword) =
[P(keyword | Positive) × P(Positive)] / P(keyword)

The prior probability of a positive review is 0.5 because half of the dataset contains positive reviews.

### Results

| Keyword | Prior | Likelihood | Marginal | Posterior |
|---|---:|---:|---:|---:|
| excellent | 0.5000 | 0.1147 | 0.0710 | 80.74% |
| brilliant | 0.5000 | 0.0635 | 0.0418 | 76.01% |
| loved | 0.5000 | 0.0765 | 0.0507 | 75.36% |
| terrible | 0.5000 | 0.0153 | 0.0540 | 14.18% |
| boring | 0.5000 | 0.0236 | 0.0610 | 19.37% |
| worst | 0.5000 | 0.0164 | 0.0887 | 9.27% |

### Interpretation
The positive keywords have higher posterior probabilities of a positive review, while the negative keywords have lower probabilities. The keyword "excellent" has the highest posterior probability at 80.74%, whereas "worst" has the lowest at 9.27%.

These results show that keywords provide useful evidence about sentiment, but a single keyword does not determine the sentiment of an entire review.

### How to Run
1. Download the IMDb dataset from the link above.
2. Place the CSV file at `data/imdb_reviews.csv`.
3. Run the following command from the project root:

   `python src/bayesian_sentiment.py`
