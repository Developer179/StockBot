# # app/__init__.py
# from flask import Flask
# from flask_cors import CORS
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)

# def create_app():
#     """
#     Create and configure the Flask application
#     """
#     # Create Flask app
#     app = Flask(__name__)
#     app.secret_key = 'your_secret_key_here'  # Required for sessions

#     # Configure CORS properly - allowing specific origins
#     CORS(app,
#          resources={r"/*": {
#             "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
#             "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
#             "allow_headers": ["Content-Type", "Authorization", "Accept"],
#             "supports_credentials": True  # Make sure this is set to True
#          }},)

#     logger.info("Flask app created with CORS enabled for http://localhost:3000")

#     # Import and register routes
#     from app.routes import register_routes
#     register_routes(app)

#     return app
# app/__init__.py
import os
import logging
from flask import Flask
from dotenv import load_dotenv
from flask_cors import CORS # Import CORS

def create_app():
    """Creates and configures the Flask application instance."""
    load_dotenv() # Load .env variables

    app = Flask(__name__)

    # --- Initialize CORS ---
    # Basic configuration: Allow all origins for development
    # For production, restrict origins: origins=["http://your-frontend-domain.com", "https://..."]
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    # Explanation:
    # - CORS(app): Initialize CORS for the entire app.
    # - resources={r"/api/*": ...}: Apply CORS specifically to routes starting with /api/.
    # - {"origins": "*"}: Allow requests from ANY origin.
    #   Replace "*" with "http://localhost:3000" for more specific local dev,
    #   and with your actual frontend domain(s) in production.
    # ---

    # Configuration (can be expanded)
    app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_default_secret_key_for_dev')

    # Set up logging level from environment variable or default to INFO
    log_level_name = os.environ.get('LOG_LEVEL', 'INFO').upper()
    log_level = getattr(logging, log_level_name, logging.INFO)
    # Ensure logging is configured before first use
    if not app.debug or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        # Avoid duplicate handlers when Werkzeug reloader is active
         logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
         logging.getLogger().setLevel(log_level) # Ensure root logger level is set

    logger = logging.getLogger(__name__)
    logger.info(f"Flask application configured with log level: {log_level_name}")
    logger.info(f"CORS configured for /api/* routes with origins: *") # Log CORS config

    # Import and register blueprints
    try:
        # Assuming you removed search_bp based on previous advice
        from .routes.session import session_bp
        app.register_blueprint(session_bp, url_prefix='/api/session')
        logger.info("Registered 'session' blueprint under /api/session")
    except Exception as e:
        logger.error(f"Failed to import or register blueprints: {e}", exc_info=True)
        raise # Fatal if blueprints can't be registered

    # Add a simple root route for health check or info
    @app.route('/')
    def index():
        logger.debug("Root path '/' accessed.")
        return "API Service is running."

    return app