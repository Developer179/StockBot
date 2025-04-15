"""
Database utility functions.
"""
import logging
import psycopg2
import os

# --- Logging Setup ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "..", "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "app.log")

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# --- Database Connection Parameters ---
# Consider replacing with environment variables for security
DB_NAME = "univest"
DB_USER = "postgres"
DB_PASSWORD = "univest@123"
DB_HOST = "localhost"
DB_PORT = "5432"

def get_db_connection():
    """
    Create and return a connection to the database.
    """
    try:
        logger.debug("Attempting to connect to the database...")
        conn = psycopg2.connect(
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            host=DB_HOST,
            port=DB_PORT
        )
        logger.info("Database connection successful.")
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise
