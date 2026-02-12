"""
ETL Module for ReturnX Project.

This module handles data extraction from the source database (or CSV fallback)
and performs initial preprocessing cleaning steps.
It aligns with the logic in 'notebooks/01_extraction.ipynb'.
"""

import pandas as pd
import os
import logging
import sys
from sqlalchemy import create_engine
from src.config import DB_CONNECTION_STRING
from src.preprocessing import clean_review_text, calculate_word_count

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

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
        engine = create_engine(DB_CONNECTION_STRING)
        
        # Exact query from notebook 01
        query = "SELECT * FROM retail_returns.labeled_reviews_v"
        
        df = pd.read_sql(query, engine)
        logger.info(f"Data extracted successfully from database. Shape: {df.shape}")
        return df
        
    except Exception as e:
        logger.warning(f"Database extraction failed: {e}")
        logger.info("Falling back to local CSV file...")

    # Fallback to CSV
    # Check for both labeled_reviews.csv (notebook 01 output) and labeled2_reviews.csv (used in other contexts)
    potential_paths = [csv_path, "data/interim/labeled2_reviews.csv"]
    
    for path in potential_paths:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                logger.info(f"Data loaded from {path}. Shape: {df.shape}")
                return df
            except Exception as e:
                logger.error(f"Failed to read CSV at {path}: {e}")

    error_msg = f"Data extraction failed. Neither database nor CSV files found in {potential_paths}."
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
            logger.info(f"Found {empty_count} rows with empty text after cleaning.")

        logger.info(f"Preprocessing complete. Final Shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error during preprocessing: {e}")
        raise e