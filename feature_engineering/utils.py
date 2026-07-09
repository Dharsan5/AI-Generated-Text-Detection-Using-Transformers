"""
Handcrafted Feature Extraction Utilities
=========================================
Computes 11 linguistic/statistical features per text sample:

  1.  word_count           – total tokens (whitespace-split)
  2.  char_count           – total characters (including spaces)
  3.  sentence_count       – sentences delimited by [.!?]
  4.  avg_word_length      – mean characters per token
  5.  avg_sentence_length  – mean words per sentence
  6.  vocab_size           – number of unique tokens
  7.  lexical_diversity    – vocab_size / word_count
  8.  stopword_ratio       – stopwords / word_count
  9.  punctuation_ratio    – punctuation chars / char_count
  10. digit_ratio          – digit chars / char_count
  11. uppercase_ratio      – uppercase chars / char_count (pre-lowercasing, use raw text)

All batch extraction runs in parallel via concurrent.futures for speed on
the ~552k sample dataset.
"""

import re
import string
import logging
from typing import List, Dict, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stopword set (NLTK English stopwords, no external dependency)
# ---------------------------------------------------------------------------
STOPWORDS: frozenset = frozenset({
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "you're", "you've", "you'll", "you'd", "your", "yours",
    "yourself", "yourselves", "he", "him", "his", "himself",
    "she", "she's", "her", "hers", "herself",
    "it", "it's", "its", "itself",
    "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "that'll",
    "these", "those", "am", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "having",
    "do", "does", "did", "doing",
    "a", "an", "the", "and", "but", "if", "or", "because",
    "as", "until", "while", "of", "at", "by", "for", "with",
    "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below",
    "to", "from", "up", "down", "in", "out", "on", "off",
    "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very",
    "s", "t", "can", "will", "just", "don", "don't",
    "should", "should've", "now", "d", "ll", "m", "o",
    "re", "ve", "y", "ain", "aren", "aren't", "couldn",
    "couldn't", "didn", "didn't", "doesn", "doesn't",
    "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't",
    "isn", "isn't", "ma", "mightn", "mightn't",
    "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
    "shouldn", "shouldn't", "wasn", "wasn't",
    "weren", "weren't", "won", "won't", "wouldn", "wouldn't",
})

_SENTENCE_SPLIT = re.compile(r"[.!?]+")
_PUNCT_SET: frozenset = frozenset(string.punctuation)


# ---------------------------------------------------------------------------
# Individual feature functions
# ---------------------------------------------------------------------------

def _tokens(text: str) -> List[str]:
    """Return whitespace-split token list (already lowercased by preprocessor)."""
    return text.split() if isinstance(text, str) else []


def _sentences(text: str) -> List[str]:
    """Split text into non-empty sentence fragments."""
    if not isinstance(text, str):
        return []
    parts = _SENTENCE_SPLIT.split(text)
    return [s.strip() for s in parts if s.strip()]


# ---------------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------------

def extract_features(text: str) -> Dict[str, Union[int, float]]:
    """
    Extract all 11 handcrafted features from a single (preprocessed) text.

    Args:
        text: Preprocessed (lowercased, cleaned) text string.

    Returns:
        Dict mapping feature name → numeric value.
    """
    if not isinstance(text, str) or not text:
        return {
            "word_count": 0, "char_count": 0, "sentence_count": 0,
            "avg_word_length": 0.0, "avg_sentence_length": 0.0,
            "vocab_size": 0, "lexical_diversity": 0.0,
            "stopword_ratio": 0.0, "punctuation_ratio": 0.0,
            "digit_ratio": 0.0, "uppercase_ratio": 0.0,
        }

    tokens = _tokens(text)
    sentences = _sentences(text)
    char_count = len(text)
    word_count = len(tokens)
    sent_count = len(sentences)

    # Averages
    avg_word_len = (
        sum(len(t) for t in tokens) / word_count if word_count else 0.0
    )
    avg_sent_len = word_count / sent_count if sent_count else 0.0

    # Vocabulary
    vocab = set(tokens)
    vocab_size = len(vocab)
    lex_div = vocab_size / word_count if word_count else 0.0

    # Ratios
    sw_count = sum(1 for t in tokens if t in STOPWORDS)
    sw_ratio = sw_count / word_count if word_count else 0.0

    punct_count = sum(1 for c in text if c in _PUNCT_SET)
    punct_ratio = punct_count / char_count if char_count else 0.0

    digit_count = sum(1 for c in text if c.isdigit())
    digit_ratio = digit_count / char_count if char_count else 0.0

    # Uppercase ratio uses pre-lowercased text so it will always be 0 after
    # preprocessing; kept for completeness / raw-text use cases.
    upper_count = sum(1 for c in text if c.isupper())
    upper_ratio = upper_count / char_count if char_count else 0.0

    return {
        "word_count":          word_count,
        "char_count":          char_count,
        "sentence_count":      sent_count,
        "avg_word_length":     round(avg_word_len, 6),
        "avg_sentence_length": round(avg_sent_len, 6),
        "vocab_size":          vocab_size,
        "lexical_diversity":   round(lex_div, 6),
        "stopword_ratio":      round(sw_ratio, 6),
        "punctuation_ratio":   round(punct_ratio, 6),
        "digit_ratio":         round(digit_ratio, 6),
        "uppercase_ratio":     round(upper_ratio, 6),
    }


def extract_features_batch(
    texts: List[str],
) -> List[Dict[str, Union[int, float]]]:
    """
    Extract features from a list of texts (sequential).

    Args:
        texts: List of preprocessed strings.

    Returns:
        List of feature dicts (same order as *texts*).
    """
    return [extract_features(t) for t in texts]


def features_to_dataframe(
    features_list: List[Dict[str, Union[int, float]]]
) -> "pd.DataFrame":  # type: ignore[name-defined]
    """Convert list of feature dicts to a pandas DataFrame."""
    import pandas as pd  # lazy import to avoid circular issues

    return pd.DataFrame(features_list)
