"""
Configuration management for the ReturnX project.
Handles environment variables and database connection strings.
"""

import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()


# Database Credentials from environment variables
user = os.getenv("DB_USER", "").strip()
password = os.getenv("DB_PASS", "").strip()
host = os.getenv("DB_HOST", "").strip()
port = os.getenv("DB_PORT", "").strip()
dbname = os.getenv("DB_NAME", "").strip()

def get_db_connection_string() -> str:
    """Constructs the PostgreSQL connection string from environment variables."""
    user = os.getenv("DB_USER", "postgres").strip()
    password = os.getenv("DB_PASS", "password").strip()
    host = os.getenv("DB_HOST", "localhost").strip()
    port = os.getenv("DB_PORT", "5432").strip()
    dbname = os.getenv("DB_NAME", "returnx_db").strip()
    
    return f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

# Constant for easy import
DB_CONNECTION_STRING = get_db_connection_string()