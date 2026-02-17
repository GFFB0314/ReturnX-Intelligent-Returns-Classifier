"""
Configuration management for the ReturnX project.
Handles environment variables and database connection strings.
"""

import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


# Database Credentials from environment variables
_db_user = os.getenv("DB_USER", "").strip()
_db_pass = os.getenv("DB_PASS", "").strip()
_db_host = os.getenv("DB_HOST", "").strip()
_db_port = os.getenv("DB_PORT", "").strip()
_db_name = os.getenv("DB_NAME", "").strip()


def get_db_connection_string() -> str:
    """Constructs the PostgreSQL connection string from environment variables."""
    return f"postgresql://{_db_user}:{_db_pass}@{_db_host}:{_db_port}/{_db_name}"


# Constant for easy import
DB_CONNECTION_STRING = get_db_connection_string()
