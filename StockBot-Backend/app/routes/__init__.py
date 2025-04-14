# app/routes/__init__.py
from .search import search_bp
from .session import session_bp

def register_routes(app):
    app.register_blueprint(search_bp)
    app.register_blueprint(session_bp)
    

# app/__init__.py
# import os
# import logging
# from flask import Flask
# from dotenv import load_dotenv

# def create_app():
#     """Creates and configures the Flask application instance."""
#     load_dotenv() # Load .env variables

#     app = Flask(__name__)

#     # Configuration (can be expanded)
#     app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'a_default_secret_key_for_dev')
#     # Add other configurations if needed

#     # Set up logging level from environment variable or default to INFO
#     log_level_name = os.environ.get('LOG_LEVEL', 'INFO').upper()
#     log_level = getattr(logging, log_level_name, logging.INFO)
#     logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#     logging.getLogger().setLevel(log_level) # Ensure root logger level is set

#     logger = logging.getLogger(__name__)
#     logger.info(f"Flask application configured with log level: {log_level_name}")


#     # Import and register blueprints
#     try:
#         from .routes.session import session_bp
#         # Register blueprint with API prefix
#         app.register_blueprint(session_bp, url_prefix='/api/session')
#         logger.info("Registered 'session' blueprint under /api/session")
#     except Exception as e:
#         logger.error(f"Failed to import or register blueprints: {e}", exc_info=True)
#         raise # Fatal if blueprints can't be registered

#     # Add a simple root route for health check or info
#     @app.route('/')
#     def index():
#         logger.debug("Root path '/' accessed.")
#         return "API Service is running."

#     return app