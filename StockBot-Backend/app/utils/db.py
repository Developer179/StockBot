# app/utils/db.py
"""
Database utility functions.
"""
import logging
import psycopg2

logger = logging.getLogger(__name__)

# Database connection parameters
# In a real application, these should be in a config file or environment variables
DB_NAME = "univest"
DB_USER = "apple"
DB_PASSWORD = "univest123"
DB_HOST = "localhost"
DB_PORT = "5432"

def get_db_connection():
    """
    Create and return a connection to the database.
    """
    try:
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise