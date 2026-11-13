
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def initialize_nltk():
    resources = {
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }

    for resource_path, package_name in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            if not nltk.download(package_name, quiet=True):
                raise RuntimeError(
                    f"Could not download NLTK resource: {package_name}"
                )


initialize_nltk()

LEMMATIZER = WordNetLemmatizer()

# Keep negation words because they can reverse sentiment.
NEGATIONS = {"no", "nor", "not"}

STOP_WORDS = set(stopwords.words("english")) - NEGATIONS


def clean_text(text: str) -> str:
    """Clean a review without removing important negation."""
    text = str(text).lower()

    # Expand common negative contractions before removing punctuation.
    text = re.sub(r"n't\b", " not", text)
    text = re.sub(r"\bcan't\b", "cannot", text)
    text = re.sub(r"\bwon't\b", "will not", text)

    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()

    words = [
        LEMMATIZER.lemmatize(word)
        for word in words
        if word not in STOP_WORDS
    ]

    return " ".join(words)
