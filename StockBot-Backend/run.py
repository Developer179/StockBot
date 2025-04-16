# run.py
"""
Entry point for the Flask application.
Initializes necessary components and starts the server.
"""

import logging
import sys
import os
from dotenv import load_dotenv

# Load .env file before other imports that might need environment variables
load_dotenv()

# --- Configuration (from .env or defaults) ---
DEBUG_MODE = os.environ.get('FLASK_DEBUG', 'True').lower() in ('true', '1', 't')
HOST_ADDR = os.environ.get('FLASK_HOST', '0.0.0.0')
PORT_NUM = int(os.environ.get('FLASK_PORT', 5000))
LOG_LEVEL_NAME = os.environ.get('LOG_LEVEL', 'INFO').upper()
# ---

# Setup logging early
log_level = getattr(logging, LOG_LEVEL_NAME, logging.INFO)
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=log_level, format=log_format, handlers=[
    logging.StreamHandler(sys.stdout) # Ensure logs go to stdout
])
# Set Flask's logger level too if needed, but basicConfig might cover it
# logging.getLogger('werkzeug').setLevel(log_level)

logger = logging.getLogger(__name__) # Get logger for this module (run.py)


# --- Now import app components ---
try:
    from app import create_app
    # Import functions needed for pre-loading
    from app.routes.session import (
        build_screener_index,
        load_local_qa_store,
        SCREENER_INDEX,     # For checking count
        qa_store            # For checking count
    )
except ImportError as e:
     # This can happen if the virtual environment isn't set up correctly
     # or if there are circular imports. Provide guidance.
     logger.error(f"ImportError during startup: {e}", exc_info=True)
     logger.error("Please ensure you are running from the project root directory,")
     logger.error("that your virtual environment is activated, and all dependencies")
     logger.error("in requirements.txt are installed.")
     sys.exit(1)
except Exception as e:
     logger.error(f"Unexpected error during imports: {e}", exc_info=True)
     sys.exit(1)


def main():
    """Initializes and runs the Flask application."""

    #logger.info("====================================================")
    #logger.info("          Initializing Stock Assistant API          ")
    #logger.info("====================================================")
    #logger.info(f"Log Level set to: {LOG_LEVEL_NAME}")
    #logger.info(f"Debug Mode: {DEBUG_MODE}")

    # --- Pre-computation / Caching ---
    # These can take time, so do them before starting the server listener

    # 1. Build Company Name Index
    try:
        #logger.info("Building company name index...")
        start_time = time.time()
        duration = time.time() - start_time
        # Log success and count
        #logger.info(f"Company name index built successfully ({len(COMPANY_NAME_INDEX)} items) in {duration:.2f} seconds.")
    except Exception as e:
        # Decide if fatal. For search/matching, it probably is.
        logger.error(f"FATAL: Failed to build company name index: {e}", exc_info=True)
        logger.error("Application cannot function correctly without the company index. Exiting.")
        sys.exit(1) # Exit with a non-zero code indicating error

    # 2. Build Screener Index
    try:
        #logger.info("Building screener index...")
        start_time = time.time()
        build_screener_index()
        duration = time.time() - start_time
        #logger.info(f"Screener index built successfully ({len(SCREENER_INDEX)} items) in {duration:.2f} seconds.")
    except Exception as e:
        # Decide if fatal. Might be less critical than company index.
        logger.error(f"ERROR: Failed to build screener index: {e}", exc_info=True)
        logger.warning("Proceeding without screener index functionality.")
        # Don't exit, but log the limitation.

    # 3. Load Local Q&A Store (Optional Feature)
    try:
        #logger.info("Loading local Q&A store...")
        start_time = time.time()
        load_local_qa_store()
        duration = time.time() - start_time
        #logger.info(f"Local Q&A store loaded ({len(qa_store)} items) in {duration:.2f} seconds.")
    except FileNotFoundError:
        # This is expected if the file doesn't exist, treat as warning.
        logger.warning("Local Q&A training file not found. Local matching cache disabled.")
    except Exception as e:
        # Log other errors but don't treat as fatal.
        logger.error(f"ERROR: Failed to load local Q&A store: {e}", exc_info=True)
        logger.warning("Proceeding without local Q&A cache due to loading error.")


    # --- Create Flask App ---
    try:
        #logger.info("Creating Flask application instance...")
        app = create_app()
        #logger.info("Flask application instance created successfully.")
    except Exception as e:
        logger.error(f"FATAL: Failed to create Flask application instance: {e}", exc_info=True)
        logger.error("Application cannot start. Exiting.")
        sys.exit(1)

    # --- Start Server ---
    #logger.info("----------------------------------------------------")
    #logger.info(f" Starting Flask server on http://{HOST_ADDR}:{PORT_NUM}")
    #logger.info("----------------------------------------------------")
    try:
        # Use configured variables
        # Setting use_reloader=False when debug=True prevents running initializers twice,
        # but you lose auto-reload on code change. Keep it True during active dev.
        use_reloader = DEBUG_MODE # Typically True when Debug is True
        app.run(debug=DEBUG_MODE, host=HOST_ADDR, port=PORT_NUM, use_reloader=use_reloader)
    except OSError as e:
         if "Address already in use" in str(e):
              logger.error(f"FATAL: Port {PORT_NUM} is already in use. Please stop the existing process or use a different port.")
         else:
              logger.error(f"FATAL: Failed to start Flask server due to OS Error: {e}", exc_info=True)
         sys.exit(1)
    except Exception as e:
        logger.error(f"FATAL: An unexpected error occurred while trying to start the Flask server: {e}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    # Import time here, only needed within main scope for duration calculation
    import time
    main()
    # This part will likely not be reached until the server is stopped (e.g., Ctrl+C)
    #logger.info("Flask server has shut down.")
    #logger.info("====================================================")