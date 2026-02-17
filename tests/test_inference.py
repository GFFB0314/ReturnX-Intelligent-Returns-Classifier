"""
Unit tests for the inference module.
"""

from unittest.mock import MagicMock
import numpy as np
from src.inference import calculate_word_count, predict_category


def test_calculate_word_count():
    """Test word count calculation logic."""
    assert calculate_word_count("hello world") == 2
    assert calculate_word_count("") == 0
    assert calculate_word_count("  three   words here  ") == 3


def test_predict_category_structure():
    """Test that predict_category returns the correct dictionary structure."""
    # Mock artifacts
    mock_model = MagicMock()
    mock_model.predict_proba.return_value = np.array(
        [[0.1, 0.1, 0.7, 0.1]]
    )  # 4 classes, Sizing at index 2

    mock_le = MagicMock()
    mock_le.inverse_transform.return_value = ["Sizing"]
    mock_le.classes_ = ["Defect", "Other", "Sizing", "Style"]

    mock_tfidf = MagicMock()
    mock_tfidf.transform.return_value = np.zeros((1, 10))  # Dummy sparse matrix

    result = predict_category(
        complaint_text="Too small",
        age=30,
        rating=3,
        model=mock_model,
        label_encoder=mock_le,
        tfidf=mock_tfidf,
    )

    assert result["category"] == "Sizing"
    assert result["confidence"] == 0.7
    assert "probabilities" in result
    assert result["probabilities"]["Sizing"] == 0.7
    assert result["probabilities"]["Defect"] == 0.1
