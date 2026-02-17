"""
Unit tests for the preprocessing module.
"""

from src.preprocessing import clean_review_text


def test_clean_review_text_basic():
    """Test basic cleaning: lowercase, special chars."""
    raw = "Hello World! 123"
    expected = "hello world 123"
    assert clean_review_text(raw) == expected


def test_poison_phrases_removal():
    """Test removal of poison phrases like 'true to size'."""
    raw = "The dress is true to size and fits perfectly."
    assert clean_review_text(raw) == "dress"


def test_stopwords_and_negations():
    """Test that 'not' is kept but 'the' is removed."""
    raw = "The material is not bad."
    expected = "material not bad"  # "the"/"is" removed, "not" kept
    assert clean_review_text(raw) == expected


def test_individual_stop_words():
    """Test removal of custom stop words like 'love', 'cute'."""
    raw = "I love this cute dress."
    expected = "dress"  # "love"/"cute" removed
    assert clean_review_text(raw) == expected


def test_empty_input():
    """Test handling of empty or non-string input."""
    assert clean_review_text("") == ""
    assert clean_review_text(None) == ""
