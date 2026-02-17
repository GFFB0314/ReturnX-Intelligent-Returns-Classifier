"""
ETL Module for ReturnX Project.

This module handles data extraction from the source database (or CSV fallback)
and performs initial preprocessing cleaning steps.
It aligns with the logic in 'notebooks/01_extraction.ipynb'.
"""

import os

import pandas as pd
from sqlalchemy import create_engine, exc
from sqlalchemy.engine.base import Engine
from src.config import DB_CONNECTION_STRING
from src.preprocessing import clean_review_text, calculate_word_count
from src.logging_utils import setup_logger

# Configure logging using shared utility
logger = setup_logger(__name__)


def extract_data(csv_path: str = "data/interim/labeled_reviews.csv") -> pd.DataFrame:
    """
    Extracts data from the SQL database using the specific view 'retail_returns.labeled_reviews_v'.
    If the database is inaccessible, falls back to a local CSV file.

    Args:
        csv_path (str): Path to the fallback CSV file. Default matches notebook 01 output.

    Returns:
        pd.DataFrame: extracted dataframe.
    """
    logger.info("Starting data extraction...")

    try:
        logger.info("Attempting to connect to database using DB_CONNECTION_STRING...")
        engine: Engine = create_engine(DB_CONNECTION_STRING)

        # Exact query from notebook 01
        query = "SELECT * FROM retail_returns.labeled_reviews_v"

        df = pd.read_sql(query, engine)
        logger.info("Data extracted successfully from database. Shape: %s", df.shape)
        return df

    except exc.SQLAlchemyError as e:
        logger.warning("Database extraction failed: %s", e)
        logger.info("Falling back to local CSV file...")

    # Fallback to CSV
    # Check for both labeled_reviews.csv
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            logger.info("Data loaded from %s. Shape: %s", csv_path, df.shape)
            return df
        except (pd.errors.ParserError, IOError) as e:
            logger.error("Failed to read CSV at %s: %s", csv_path, e)

    error_msg = (
        f"Data extraction failed. Neither database nor CSV files found in {csv_path}."
    )
    logger.error(error_msg)
    raise FileNotFoundError(error_msg)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies text cleaning and feature engineering (word count).

    Args:
        df (pd.DataFrame): Raw dataframe containing 'review_text'.

    Returns:
        pd.DataFrame: Dataframe with 'clean_text' and 'word_count' columns added.
    """
    logger.info("Starting preprocessing...")

    if "review_text" not in df.columns:
        error_msg = "Input dataframe missing required column 'review_text'"
        logger.error(error_msg)
        raise ValueError(error_msg)

    try:
        # Text cleaning
        logger.info("Applying text cleaning (clean_review_text)...")
        df["clean_text"] = df["review_text"].apply(clean_review_text)

        # Feature engineering
        logger.info("Calculating word counts...")
        df["word_count"] = df["review_text"].apply(calculate_word_count)

        # Handle NaNs in text - ensure no empty strings cause issues later
        df["clean_text"] = df["clean_text"].fillna("")

        # Optional: Remove empty rows after cleaning if strictly needed,
        # but notebook 03 just uses them. We'll keep them but warn if many.
        empty_count = len(df[df["clean_text"] == ""])
        if empty_count > 0:
            logger.info("Found %d rows with empty text after cleaning.", empty_count)

        logger.info("Preprocessing complete. Final Shape: %s", df.shape)
        return df

    except (ValueError, TypeError) as e:
        logger.error("Error during preprocessing: %s", e)
        raise e
