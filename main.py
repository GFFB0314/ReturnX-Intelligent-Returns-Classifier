"""
ReturnX Intelligent Returns Classifier - CLI Entry Point.

This script orchestrates the MLOps pipeline:
1. 'train': Runs the ETL and Model Training pipeline.
2. 'dashboard': Launches the Streamlit prediction dashboard.

Usage:
    python main.py train
    python main.py dashboard
"""

import argparse
import logging
import os
import sys

from src import etl, model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def run_training_pipeline():
    """
    Executes the full training pipeline:
    1. Extract and preprocess data from DB or CSV.
    2. Train model and save best artifacts for deployment.
    """
    logger.info("Initializing Training Pipeline...")

    # Step 1: ETL
    logger.info("[Step 1/2] Running ETL Process...")
    try:
        df_raw = etl.extract_data()
        df_processed = etl.preprocess_data(df_raw)
        logger.info("ETL Complete. Data Shape: %s", df_processed.shape)
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        logger.critical("ETL Pipeline Failed: %s", e)
        sys.exit(1)

    # Step 2: Model Training
    logger.info("[Step 2/2] Training Model...")
    try:
        model.train_model(df_processed)
        logger.info(
            "Training pipeline completed successfully. Artifacts ready in 'src/'."
        )
    except (ValueError, TypeError, RuntimeError) as e:
        logger.critical("Model Training Failed: %s", e)
        sys.exit(1)


def run_dashboard():
    """
    Launches the Streamlit dashboard application.
    """
    logger.info("Launching Streamlit Dashboard...")
    try:
        # Use os.system to run the streamlit command
        exit_code = os.system("streamlit run dashboard.py")
        if exit_code != 0:
            logger.error("Dashboard exited with errors. Exit code: %d", exit_code)
    except (OSError, RuntimeError) as e:
        logger.error("Failed to launch dashboard: %s", e)


def main():
    """
    Main entry point for the ReturnX CLI.
    Parses arguments and dispatches to the appropriate pipeline task.
    """
    parser = argparse.ArgumentParser(description="ReturnX Classifier CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Train command
    subparsers.add_parser(
        "train", help="Run the retraining pipeline (ETL + Model Training)"
    )

    # Dashboard command
    subparsers.add_parser("dashboard", help="Launch the Streamlit Prediction Dashboard")

    args = parser.parse_args()

    if args.command == "train":
        run_training_pipeline()
    elif args.command == "dashboard":
        run_dashboard()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
