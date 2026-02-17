"""
Text preprocessing module for ReturnX project.
Replicates EXACT logic from notebook 03_nlp_feature_engineering.ipynb
"""

import re
from nltk.corpus import stopwords

# Poison phrases to remove
POISON_PHRASES: list[str] = [
    "true to size",
    "true size",
    "love dress",
    "love love",
    "fit perfectly",
    "fits perfectly",
    "fit great",
    "just right",
    "looks like",
    "looks great",
    "looked like",
    "look like",
    "look great",
    "highly recommend",
    "super cute",
    "absolutely love",
    "really wanted",
    "received compliments",
    "soft comfortable",
    "material soft",
    "fabric soft",
    "super soft",
    "usually wear",
    "normally wear",
    "usual size",
    "wait wear",
    "easy wear",
    "fit well",
    "fits well",
    "well made",
]

# Individual positive sentiment words to remove
INDIVIDUAL_STOP_WORDS: list[str] = [
    "love",
    "great",
    "perfect",
    "cute",
    "recommend",
    "compliments",
    "comfortable",
    "want",
    "wanted",
    "feel",
    "like",
    "nice",
    "good",
]

# Negations to KEEP - don't remove these
NEGATIONS: set[str] = {"no", "not", "nor", "neither", "never", "none"}


def clean_review_text(text: str) -> str:
    """
    Clean review text using EXACT logic from notebook 03.

    Steps (from notebook lines 410-446):
    1. Convert to lowercase
    2. Remove poison phrases
    3. Remove stopwords except negations
    4. Remove special characters but keep numbers
    5. Collapse multiple spaces

    Args:
        text: Raw customer complaint text

    Returns:
        Cleaned text ready for TF-IDF vectorization
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Lowercase
    text = text.lower()

    # Step 2: Remove poison phrases
    for phrase in POISON_PHRASES:
        text = re.sub(re.escape(phrase), "", text)

    # Step 3: Remove stopwords except negations
    stop_words = set(stopwords.words("english"))
    final_stop_words = list(stop_words - NEGATIONS)
    final_stop_words.extend(INDIVIDUAL_STOP_WORDS)

    words: list[str] = text.split()
    words_cleaned = [
        word for word in words if (word not in final_stop_words) or (word in NEGATIONS)
    ]
    text = " ".join(words_cleaned)

    # Step 4: Remove special chars but KEEP numbers
    text = re.sub(r"[^a-z0-9\s]", "", text)

    # Step 5: Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def calculate_word_count(text: str) -> int:
    """
    Calculate word count from ORIGINAL text (before cleaning).
    This matches the 'word_count' feature from notebook 02.

    Args:
        text: Original customer complaint text

    Returns:
        Number of words in the text
    """
    if not isinstance(text, str):
        return 0
    return len(str(text).split())
