
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def build_count_vectorizer():
    """Represent text using word and two-word phrase counts."""
    return CountVectorizer(
        max_features=1500,
        ngram_range=(1, 2)
    )


def build_tfidf_vectorizer():
    """Represent text using TF-IDF features."""
    return TfidfVectorizer(
        max_features=1500,
        ngram_range=(1, 2)
    )
