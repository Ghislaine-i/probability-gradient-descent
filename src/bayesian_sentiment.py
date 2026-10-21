
import csv
import re
from html import unescape
from pathlib import Path

# Find the project root and dataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "imdb_reviews.csv"

# Keywords to analyze
POSITIVE_KEYWORDS = ["excellent", "brilliant", "loved"]
NEGATIVE_KEYWORDS = ["terrible", "boring", "worst"]


def clean_review(review):
    """Remove HTML tags and convert review text to lowercase."""
    review = re.sub(r"<br\s*/?\s*>", " ", review, flags=re.IGNORECASE)
    review = re.sub(r"<[^>]+>", " ", review)
    return unescape(review).lower()


def load_reviews(filename):
    """Load IMDb reviews and their sentiment labels from the CSV file."""
    reviews = []

    if not filename.exists():
        raise FileNotFoundError(
            f"\nDataset not found at: {filename}\n"
            "Make sure your IMDb CSV file is named "
            "'imdb_reviews.csv' and is inside the data folder."
        )

    with open(filename, "r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        if not reader.fieldnames:
            raise ValueError("The CSV file is empty or has no column headers.")

        # Make column names case-insensitive
        column_names = {
            name.strip().lower(): name for name in reader.fieldnames
        }

        if "review" not in column_names or "sentiment" not in column_names:
            raise ValueError(
                "The CSV must contain 'review' and 'sentiment' columns. "
                f"Columns found: {reader.fieldnames}"
            )

        review_column = column_names["review"]
        sentiment_column = column_names["sentiment"]

        for row in reader:
            review_text = row.get(review_column, "")
            sentiment = row.get(sentiment_column, "").strip().lower()

            if not review_text or sentiment not in ("positive", "negative"):
                continue

            cleaned_text = clean_review(review_text)
            words = set(re.findall(r"\b[a-z]+\b", cleaned_text))

            reviews.append({
                "words": words,
                "sentiment": sentiment
            })

    if not reviews:
        raise ValueError(
            "No valid reviews were loaded. Check the dataset and its labels."
        )

    return reviews


def calculate_probabilities(reviews, keyword):
    """Calculate prior, likelihood, marginal, and posterior probabilities."""
    total_reviews = len(reviews)

    positive_reviews = [
        review for review in reviews
        if review["sentiment"] == "positive"
    ]

    matching_reviews = [
        review for review in reviews
        if keyword in review["words"]
    ]

    positive_matching_reviews = [
        review for review in positive_reviews
        if keyword in review["words"]
    ]

    if not matching_reviews:
        print(f"\nKeyword: {keyword}")
        print("This keyword was not found in the dataset.")
        return

    # P(Positive)
    prior = len(positive_reviews) / total_reviews

    # P(Keyword | Positive)
    likelihood = (
        len(positive_matching_reviews) / len(positive_reviews)
        if positive_reviews else 0
    )

    # P(Keyword)
    marginal = len(matching_reviews) / total_reviews

    # Bayes' theorem: P(Positive | Keyword)
    posterior = (likelihood * prior) / marginal

    print(f"\nKeyword: {keyword}")
    print("-" * 40)
    print(f"Reviews containing keyword: {len(matching_reviews)}")
    print(
        "Positive reviews containing keyword: "
        f"{len(positive_matching_reviews)}"
    )
    print(f"Prior P(Positive): {prior:.4f}")
    print(f"Likelihood P({keyword} | Positive): {likelihood:.4f}")
    print(f"Marginal P({keyword}): {marginal:.4f}")
    print(f"Posterior P(Positive | {keyword}): {posterior:.4f}")
    print(f"Posterior percentage: {posterior * 100:.2f}%")


def main():
    """Run the Bayesian sentiment analysis."""
    print("=" * 50)
    print("IMDb BAYESIAN SENTIMENT ANALYSIS")
    print("=" * 50)
    print(f"Dataset: {DATASET_PATH}")

    reviews = load_reviews(DATASET_PATH)

    positive_count = sum(
        1 for review in reviews
        if review["sentiment"] == "positive"
    )
    negative_count = sum(
        1 for review in reviews
        if review["sentiment"] == "negative"
    )

    print(f"\nTotal reviews loaded: {len(reviews)}")
    print(f"Positive reviews: {positive_count}")
    print(f"Negative reviews: {negative_count}")

    print("\n\n========== POSITIVE KEYWORDS ==========")
    for keyword in POSITIVE_KEYWORDS:
        calculate_probabilities(reviews, keyword)

    print("\n\n========== NEGATIVE KEYWORDS ==========")
    for keyword in NEGATIVE_KEYWORDS:
        calculate_probabilities(reviews, keyword)

    print("\n\nAnalysis completed successfully.")


# This line makes the program run when you execute this file.
if __name__ == "__main__":
    main()
