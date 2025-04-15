import os
import re
import json
import time
import uuid
import hashlib
import logging
import functools
from typing import Callable, Any, Dict, List, Optional, Union

import torch
import requests
import psycopg2  # core driver
from rapidfuzz import fuzz
from flask import Blueprint, request, jsonify
from sentence_transformers import SentenceTransformer, util

# ==============================================================================
# Logging Configuration
# ==============================================================================

LOG_LEVEL = os.getenv("UNIVEST_LOG_LEVEL", "INFO").upper()

# Configure root logger - basicConfig should ideally be called once at application entry point
# If this blueprint is part of a larger Flask app, configure logging there.
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__) # Get logger for this specific module
logger.info("Logging initialised – level=%s", LOG_LEVEL)

# ==============================================================================
# Decorators
# ==============================================================================

def log_call(level: int = logging.DEBUG):
    """Decorator to automatically log entry, exit and runtime of a function."""
    def _decorator(func: Callable):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            child_logger = logging.getLogger(func.__module__) # Use module logger
            # Truncate args/kwargs representation for cleaner logs
            arg_preview = ", ".join([repr(a)[:80] for a in args]) # Increased limit slightly
            kw_preview = ", ".join([f"{k}={repr(v)[:80]}" for k, v in kwargs.items()]) # Increased limit slightly
            log_string = f"→ {func.__name__}({arg_preview}{', ' if kw_preview else ''}{kw_preview})"
            child_logger.log(level, log_string)
            start_ts = time.time()
            try:
                result = func(*args, **kwargs)
                elapsed_ms = (time.time() - start_ts) * 1000
                child_logger.log(level, f"← {func.__name__} [%.1f ms]", elapsed_ms)
                return result
            except Exception:
                child_logger.exception(f"✖ Exception in {func.__name__}")
                raise
        return _wrapper
    return _decorator

# ==============================================================================
# Conditional Imports & Utility Imports
# ==============================================================================

# --- Conditional Import for psycopg2.extras ---
psycopg2_extras = None
try:
    if hasattr(psycopg2, "extras"):
        psycopg2_extras = psycopg2.extras
    else:
        # Attempt import if psycopg2 doesn't expose extras directly
        from psycopg2 import extras as psycopg2_extras_import # type: ignore
        psycopg2_extras = psycopg2_extras_import
except ImportError:
    logger.error("Failed to import psycopg2.extras – make sure 'psycopg2‑binary' is installed.")
    # Depending on requirements, you might want to raise an error or exit here
    # raise ImportError("psycopg2.extras is required but could not be imported.")

# --- Relative Imports for Utils (or adjust path as needed) ---
try:
    # Assumes structure like: your_project/app/session/routes.py
    # and your_project/app/utils/db.py, helpers.py
    from ..utils.db import get_db_connection
    from ..utils.helpers import make_json_safe, compute_data_hash
except (ImportError, ValueError): # ValueError for relative imports beyond top-level
    logger.warning("Relative imports failed. Attempting direct/sys.path import for utils...")
    try:
        import sys
        # Adjust path based on where this script actually lives relative to 'app'
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
        from app.utils.db import get_db_connection
        from app.utils.helpers import make_json_safe, compute_data_hash
    except ImportError as e:
        logger.critical(f"Failed to import utility functions (db, helpers): {e}. Check PYTHONPATH or project structure.", exc_info=True)
        # Critical error, might be necessary to exit if these utils are essential
        raise ImportError(f"Could not import essential utility functions: {e}")


# ==============================================================================
# Blueprint Setup
# ==============================================================================
session_bp = Blueprint('session', __name__)

# ==============================================================================
# Configuration & Constants
# ==============================================================================
ESSENTIAL_COMPANY_KEYS: List[str] = [
    # This list serves as a fallback if Rasa table detection fails
    "fin_code", "comp_name", "symbol", "isin", "sector", "industry", "scrip_name",
    "bse_symbol", "nse_last_closed_price", "nse_todays_open", "nse_todays_high",
    "nse_todays_low", "bse_last_closed_price", "bse_todays_open", "bse_todays_high",
    "bse_todays_low", "nse_lower_limit", "nse_upper_limit", "bse_lower_limit",
    "bse_upper_limit", "market_capital", "pe_ratio", "pb_ratio", "book_value",
    "face_value", "eps", "short_term_verdict", "long_term_verdict", "segment",
    "instrument_name", "type",
]

# Mapping from Rasa canonical IDs (entity values) to DB table names
# Ensure keys match exactly what Rasa's EntitySynonymMapper will output
CANONICAL_ID_TO_SOURCE_MAP = {
    # company_master Columns
    "comp_name": "company_master", "scrip_name": "company_master", "symbol": "company_master",
    "bse_symbol": "company_master", "mcx_symbol": "company_master", "lot_size": "company_master",
    "isin": "company_master", "industry": "company_master", "sector": "company_master",
    "strike_price": "company_master", "nse_lower_limit": "company_master", "nse_upper_limit": "company_master",
    "bse_lower_limit": "company_master", "bse_upper_limit": "company_master", "segment": "company_master",
    "instrument_name": "company_master",
    # company_additional_details Columns
    "nse_todays_low": "company_additional_details", "nse_todays_high": "company_additional_details",
    "nse_todays_open": "company_additional_details", "nse_last_closed_price": "company_additional_details",
    "bse_todays_low": "company_additional_details", "bse_todays_high": "company_additional_details",
    "bse_todays_open": "company_additional_details", "bse_last_closed_price": "company_additional_details",
    "oi": "company_additional_details", "short_term_verdict": "company_additional_details",
    "long_term_verdict": "company_additional_details",
    # consolidated_company_equity Columns
    "market_capital": "consolidated_company_equity", "pe_ratio": "consolidated_company_equity",
    # "bool_value": "consolidated_company_equity", # Consider renaming if possible
    "pb_ratio": "consolidated_company_equity",
    "face_value": "consolidated_company_equity", "eps": "consolidated_company_equity",
    "type": "consolidated_company_equity", # Consider renaming if possible
    "book_value": "consolidated_company_equity", # Added book_value mapping
    # Screeners (Keyword -> Table Name) - Add ALL your screener keywords here
    "FUTURES_TOP_PRICE_GAINERS": "screeners", "NIFTY50": "screeners", "LONG_TERM_VERDICT_BUY": "screeners",
    "VOLUME_SHOCKERS": "screeners", "HIGH_DIVIDEND_STOCKS": "screeners", "GOLDEN_CROSSOVER": "screeners",
    "LONG_BUILD_UP": "screeners", "SHORT_BUILD_UP": "screeners", "FUTURES_TOP_VOLUME_GAINERS": "screeners",
    # ... Add ALL other screener keywords from your DB/Rasa data ...
    "FUNDAMENTAL_STRONG_STOCKS": "screeners", "FII_HOLDING": "screeners", # Example additions
}

# Mapping from Table Name to the set of fields belonging to that table
TABLE_TO_FIELDS_MAP = {
    "company_master": {
        "fin_code", "comp_name", "scrip_name", "symbol", "bse_symbol", "mcx_symbol",
        "lot_size", "isin", "industry", "sector", "strike_price", "nse_lower_limit",
        "nse_upper_limit", "bse_lower_limit", "bse_upper_limit", "segment", "instrument_name"
    },
    "company_additional_details": {
        "fin_code", "nse_todays_low", "nse_todays_high", "nse_todays_open", "nse_last_closed_price",
        "bse_todays_low", "bse_todays_high", "bse_todays_open", "bse_last_closed_price",
        "oi", "short_term_verdict", "long_term_verdict"
    },
    "consolidated_company_equity": {
        "fin_code", "market_capital", "pe_ratio", "book_value", "pb_ratio",
        "face_value", "eps", "type" # Added bool_value here too
    },
    "screeners": { # Representing screeners conceptually, data fetch is different
         # No specific company fields, uses keyword and fin_codes list
    }
}


# ==============================================================================
# Global Variables & Cache
# ==============================================================================
COMPANY_DATA_CACHE: Dict[str, Dict[str, Any]] = {} # Consider using a more robust cache (e.g., Flask-Caching)
CACHE_EXPIRY = 3600  # 1 hour

AMBIGUITY_CACHE: Dict[str, Dict[str, Any]] = {} # Cache for ambiguity resolution context
AMBIGUITY_CACHE_EXPIRY = 300  # 5 minutes

embedding_model: Optional[SentenceTransformer] = None
COMPANY_NAME_INDEX: List[Dict[str, Any]] = []
COMPANY_EMBEDDING_MATRIX: Optional[torch.Tensor] = None
SCREENER_INDEX: List[Dict[str, Any]] = []
SCREENER_EMBEDDING_MATRIX: Optional[torch.Tensor] = None
qa_store: List[Dict[str, str]] = []
qa_embeddings: Optional[torch.Tensor] = None

# ==============================================================================
# Initialization Functions
# ==============================================================================

@log_call()
def _initialize_embedding_model() -> None:
    """Lazy-load the SentenceTransformer model."""
    global embedding_model
    if embedding_model is not None:
        return

    cache_folder = os.environ.get("SENTENCE_TRANSFORMERS_HOME") # Optional: specify cache dir
    model_name = "all-MiniLM-L6-v2" # Or your chosen model
    logger.info("Loading SentenceTransformer '%s'...", model_name)
    try:
        embedding_model = SentenceTransformer(model_name, cache_folder=cache_folder)
        # You might want to run a dummy encode here to ensure CUDA init (if using GPU)
        # embedding_model.encode("test")
        logger.info("SentenceTransformer loaded successfully.")
    except Exception as exc:
        logger.exception("Failed to load SentenceTransformer: %s", exc)
        raise # Fail fast if model loading fails

@log_call()
def build_company_name_index() -> None:
    """Populate COMPANY_NAME_INDEX & COMPANY_EMBEDDING_MATRIX from DB."""
    global COMPANY_NAME_INDEX, COMPANY_EMBEDDING_MATRIX

    # Check if already built (and model loaded)
    _initialize_embedding_model() # Ensure model is loaded
    if COMPANY_NAME_INDEX and COMPANY_EMBEDDING_MATRIX is not None and embedding_model:
        logger.debug("Company index already cached (size=%d)", len(COMPANY_NAME_INDEX))
        return
    elif not embedding_model:
        logger.error("Embedding model not loaded, cannot build company index.")
        # Clear potentially partially built state
        COMPANY_NAME_INDEX = []
        COMPANY_EMBEDDING_MATRIX = None
        return

    logger.info("Starting to build company name index...") # Indicate start

    conn = None
    embeddings: List[torch.Tensor] = []
    index: List[Dict[str, Any]] = []
    try:
        conn = get_db_connection()
        if not conn:
            logger.error("Failed to get DB connection for building company index.")
            return # Cannot proceed without DB

        cur = None
        is_dict_cursor = False
        if psycopg2_extras:
             try:
                 cur = conn.cursor(cursor_factory=psycopg2_extras.DictCursor)
                 is_dict_cursor = True
                 logger.debug("Using DictCursor for company index build.")
             except Exception as e:
                 logger.warning(f"Failed to create DictCursor, falling back to standard cursor: {e}")
                 cur = conn.cursor()
        else:
            cur = conn.cursor()
            logger.debug("Using standard cursor for company index build.")

        # Fetch only the necessary columns
        cur.execute(
            """
            SELECT fin_code, comp_name, symbol
            FROM company_master
            WHERE comp_name IS NOT NULL AND symbol IS NOT NULL AND fin_code IS NOT NULL
            ORDER BY fin_code -- Order for consistency if needed
            """
        )
        rows = cur.fetchall()
        total_rows = len(rows) # Get total count for logging
        cur.close() # Close cursor promptly
        logger.info("Fetched %d companies for embedding index", total_rows)

        if not embedding_model:
            # Double check model wasn't unloaded somehow
            logger.error("Embedding model became unavailable during index build.")
            return

        # --- *** CORRECTED LOOP *** ---
        start_build_time = time.time()
        for idx, row in enumerate(rows):
            # Log progress periodically
            if idx > 0 and (idx % 50000 == 0 or idx == total_rows - 1): # Log every 50k or at the end
                percent_done = ((idx + 1) / total_rows) * 100
                elapsed_time = time.time() - start_build_time
                logger.info("Building company index: Processed %d/%d companies (%.1f%%) in %.1fs",
                            idx + 1, total_rows, percent_done, elapsed_time)

            try:
                # Access row elements correctly based on cursor type
                fin_code = row["fin_code"] if is_dict_cursor else row[0]
                name = row["comp_name"] if is_dict_cursor else row[1]
                symbol = row["symbol"] if is_dict_cursor else row[2]

                # Defensive checks for None/empty values just in case DB query allows them
                if not fin_code or not name or not symbol:
                    logger.warning(f"Skipping row index {idx} due to missing data: fin_code={fin_code}, name={name}, symbol={symbol}")
                    continue

                # Create text representation for embedding
                text = f"{name} ({symbol})".lower()
                if len(text) < 3: # Skip very short/potentially junk entries
                    logger.debug(f"Skipping short text entry: '{text}' for fin_code={fin_code}")
                    continue

                # Encode and store
                emb = embedding_model.encode(text, convert_to_tensor=True)
                embeddings.append(emb)
                index.append({"fin_code": fin_code, "name": name, "symbol": symbol, "index_text": text})

            except Exception as inner_exc:
                # Log error for the specific row but continue building the index
                row_identifier = f"fin_code={fin_code}" if 'fin_code' in locals() else f"row index {idx}"
                logger.exception(f"Encoding or processing failed for {row_identifier}. Error: {inner_exc}. Row data (partial): {str(row)[:100]}")
                # Optionally: continue? Or break if too many errors? For now, continue.

        # --- *** END CORRECTED LOOP *** ---

        if embeddings and index:
            # Stack tensors only if embeddings were generated
            COMPANY_EMBEDDING_MATRIX = torch.stack(embeddings)
            COMPANY_NAME_INDEX = index
            end_build_time = time.time()
            logger.info("✅ Company index built successfully – %d entries in %.1f seconds.",
                        len(COMPANY_NAME_INDEX), end_build_time - start_build_time)
        else:
            logger.warning("No company embeddings generated or index populated – index left empty.")
            COMPANY_EMBEDDING_MATRIX = None
            COMPANY_NAME_INDEX = []

    except psycopg2.Error as db_err:
        logger.exception("Database error building company index: %s", db_err)
        COMPANY_EMBEDDING_MATRIX = None
        COMPANY_NAME_INDEX = []
    except Exception as e:
        # Catch unexpected errors during the build process
        logger.exception("Unexpected error building company index: %s", e)
        COMPANY_EMBEDDING_MATRIX = None
        COMPANY_NAME_INDEX = []
    finally:
        # Log completion regardless of success/failure
        logger.info("Finished attempt to build company name index.")
        if conn:
            conn.close()
            logger.debug("Database connection closed after company index build.")
            
            
@log_call()
def build_screener_index() -> None:
    """Populate SCREENER_INDEX & SCREENER_EMBEDDING_MATRIX from DB."""
    global SCREENER_INDEX, SCREENER_EMBEDDING_MATRIX
    _initialize_embedding_model()
    if SCREENER_INDEX and SCREENER_EMBEDDING_MATRIX is not None:
        logger.debug("Screener index already cached (size=%d)", len(SCREENER_INDEX))
        return

    conn = None
    embeddings: List[torch.Tensor] = []
    index: List[Dict[str, str]] = []
    try:
        conn = get_db_connection()
        cur = conn.cursor() # DictCursor not strictly necessary here
        cur.execute("SELECT keyword, title, description FROM screeners ORDER BY keyword")
        rows = cur.fetchall()
        cur.close()
        logger.info("Fetched %d screeners for embedding index", len(rows))

        if not embedding_model:
            logger.error("Embedding model not loaded, cannot build screener index.")
            return

        for keyword, title, desc in rows:
            # Create text representation for embedding
            text = f"Title: {title or keyword}. Description: {desc or ''}. Keyword: {keyword}".lower()
            if len(text) < 5: # Skip very short descriptions
                continue
            try:
                emb = embedding_model.encode(text, convert_to_tensor=True)
                embeddings.append(emb)
                # Store keyword and original text used for embedding
                index.append({"keyword": keyword, "text": text})
            except Exception:
                logger.exception("Encoding failed for screener '%s'", keyword)

        if embeddings:
            SCREENER_EMBEDDING_MATRIX = torch.stack(embeddings)
            SCREENER_INDEX = index
            logger.info("Screener index ready – %d embeddings", len(SCREENER_INDEX))
        else:
            logger.warning("No screener embeddings generated – index left empty")
            SCREENER_EMBEDDING_MATRIX = None
            SCREENER_INDEX = []

    except psycopg2.Error as db_err:
        logger.exception("Database error building screener index: %s", db_err)
        SCREENER_EMBEDDING_MATRIX = None
        SCREENER_INDEX = []
    except Exception as e:
        logger.exception("Unexpected error building screener index: %s", e)
        SCREENER_EMBEDDING_MATRIX = None
        SCREENER_INDEX = []
    finally:
        if conn:
            conn.close()

# ==============================================================================
# Core Logic Functions
# ==============================================================================

@log_call(logging.DEBUG)
def find_company_by_name_or_symbol(query: str) -> Optional[Dict[str, Any]]:
    """Finds best company match using embeddings and fuzzy matching."""
    build_company_name_index() # Ensure index is built
    if not COMPANY_NAME_INDEX or COMPANY_EMBEDDING_MATRIX is None or not embedding_model:
        logger.error("Company index or embedding model not available for matching.")
        return None

    query_clean = query.lower().strip()
    if not query_clean: return None

    try:
        query_emb = embedding_model.encode(query_clean, convert_to_tensor=True)
        # Ensure tensor dimensions match for cosine similarity
        if query_emb.ndim == 1: query_emb = query_emb.unsqueeze(0) # Add batch dim if missing
        if COMPANY_EMBEDDING_MATRIX.ndim != 2 or query_emb.ndim != 2:
            logger.error(f"Dimension mismatch: Query {query_emb.shape}, Matrix {COMPANY_EMBEDDING_MATRIX.shape}")
            return None

        cosine_scores = util.pytorch_cos_sim(query_emb, COMPANY_EMBEDDING_MATRIX)[0]
    except Exception as e:
        logger.exception(f"Error during embedding or similarity calculation for query '{query}': {e}")
        return None


    best_match: Optional[Dict[str, Any]] = None
    best_score = -1.0
    emb_thresh, fuzz_thresh = 0.60, 75 # Matching thresholds

    for i, entry in enumerate(COMPANY_NAME_INDEX):
        emb_score = cosine_scores[i].item()
        # Calculate fuzzy score against both name and symbol
        fuzz_score = max(
            fuzz.WRatio(query_clean, entry.get("name", "").lower()),
            fuzz.WRatio(query_clean, entry.get("symbol", "").lower()),
            fuzz.WRatio(query_clean, entry.get("index_text", "").lower()) # Compare against indexed text too
        )
        # Combine scores (heuristic, adjust weights as needed)
        combined = (
            emb_score * 0.6 + (fuzz_score / 100.0) * 0.4 # Weight embedding higher if decent score
            if emb_score > 0.5
            else emb_score * 0.4 + (fuzz_score / 100.0) * 0.6 # Weight fuzzy higher otherwise
        )

        # Check if this is the best match so far AND meets thresholds
        if combined > best_score and (emb_score >= emb_thresh or fuzz_score >= fuzz_thresh):
            best_score = combined
            # Create match dict, copy entry to avoid modifying index
            best_match = {**entry, "match_score": round(combined, 4)}

    if best_match:
        logger.info("Company matched – query='%s' → %s (%s) | Score: %.3f", query, best_match.get("name"), best_match.get("symbol"), best_score)
    else:
        logger.warning("No confident company match found for query '%s' (Best Score: %.3f)", query, best_score)
    return best_match


def filter_dict(data_dict: Dict, keys_to_keep: List[str]) -> Dict:
    """Filters a dictionary to keep only specified keys."""
    if not isinstance(data_dict, dict): return {}
    return {k: data_dict.get(k) for k in keys_to_keep if k in data_dict}


@log_call(logging.DEBUG)
def get_specific_company_data(fin_code: str, requested_fields: List[str]) -> Optional[Dict[str, Any]]:
    """Fetches only specific fields for a company from relevant tables."""
    conn = None
    logger.debug(f"Fetching specific fields {requested_fields} for fin_code: {fin_code}")
    if not requested_fields or not fin_code:
        logger.warning("Missing fin_code or requested_fields for specific data fetch.")
        return None

    # Field definitions mapping to table aliases
    fields_in_master = TABLE_TO_FIELDS_MAP["company_master"]
    fields_in_additional = TABLE_TO_FIELDS_MAP["company_additional_details"]
    fields_in_equity = TABLE_TO_FIELDS_MAP["consolidated_company_equity"]

    # Table alias mapping for query building
    table_aliases = {'m': fields_in_master, 'ad': fields_in_additional, 'eq': fields_in_equity}

    select_parts = set() # Using set to avoid duplicates like m.fin_code
    joins = []
    join_needed = {'ad': False, 'eq': False}
    actual_fields_to_select = set() # Store db_field name for SELECT clause

    # Always include fin_code from master table
    select_parts.add("m.fin_code")
    actual_fields_to_select.add("fin_code")

    # Determine required fields and joins
    for field in requested_fields:
        found = False
        for alias, field_set in table_aliases.items():
            if field in field_set:
                # Skip adding fin_code again if requested explicitly
                if field != 'fin_code':
                    db_field_name = f"{alias}.{field}"
                    select_parts.add(db_field_name)
                    actual_fields_to_select.add(field) # We need the final key name

                # Mark joins as needed
                if alias == 'ad': join_needed['ad'] = True
                if alias == 'eq': join_needed['eq'] = True
                found = True
                break # Field found in one table set
        if not found and field != 'fin_code':
             logger.warning(f"Requested field '{field}' not mapped to a known table alias (m, ad, eq).")

    if not actual_fields_to_select: # Should at least have 'fin_code'
        logger.warning(f"No valid fields to select for fin_code {fin_code} based on requested: {requested_fields}")
        return None

    # Add JOIN clauses if necessary
    if join_needed['ad']: joins.append("LEFT JOIN company_additional_details ad ON m.fin_code = ad.fin_code")
    if join_needed['eq']: joins.append("LEFT JOIN consolidated_company_equity eq ON m.fin_code = eq.fin_code")

    # Construct the SELECT clause from unique parts
    select_clause = ', '.join(sorted(list(select_parts)))

    # Build the final SQL query
    sql = f"SELECT {select_clause} FROM company_master m {' '.join(joins)} WHERE m.fin_code = %s"
    logger.debug("Executing SQL:\n%s\nparams=(%s,)", sql, fin_code)

    try:
        conn = get_db_connection()
        cur = None
        is_dict_cursor = False
        # Use DictCursor if available
        if psycopg2_extras:
             try: cur = conn.cursor(cursor_factory=psycopg2_extras.DictCursor); is_dict_cursor = True
             except Exception as e: logger.warning(...); cur = conn.cursor()
        else: cur = conn.cursor()

        cur.execute(sql, (fin_code,))
        result_row = cur.fetchone()

        if not result_row:
            logger.warning(f"No specific data found for fin_code {fin_code} with SQL: {sql}")
            cur.close()
            return None

        # Convert row to dictionary robustly
        if is_dict_cursor:
            data_dict_raw = dict(result_row)
        else:
            # Manual mapping if not using DictCursor
            cols = [desc[0] for desc in cur.description]
            data_dict_raw = dict(zip(cols, result_row))

        cur.close()
        logger.info(f"Fetched raw specific data for fin_code {fin_code}")

        # Map raw results back to originally requested field names
        final_data = {}
        # DictCursor directly gives keys matching SELECT clause (e.g., 'comp_name', 'nse_todays_high')
        # Standard cursor gives positional values, mapped via `cols` above.
        # We need to ensure the final dict has keys matching `requested_fields`.

        for req_field in requested_fields:
            # DictCursor should have the correct key directly
            if req_field in data_dict_raw:
                final_data[req_field] = data_dict_raw[req_field]
            else:
                # Handle potential prefix if not using DictCursor perfectly, or fallback
                potential_key_m = f"m.{req_field}"
                potential_key_ad = f"ad.{req_field}"
                potential_key_eq = f"eq.{req_field}"
                if potential_key_m in data_dict_raw: final_data[req_field] = data_dict_raw[potential_key_m]
                elif potential_key_ad in data_dict_raw: final_data[req_field] = data_dict_raw[potential_key_ad]
                elif potential_key_eq in data_dict_raw: final_data[req_field] = data_dict_raw[potential_key_eq]
                else:
                    final_data[req_field] = None # Field was requested but not found in result
                    logger.debug(f"Requested field '{req_field}' not present in fetched DB data for {fin_code}.")

        return make_json_safe(final_data) # Ensure safe types for JSON serialization

    except psycopg2.Error as db_err:
        logger.exception(f"DB error during specific fetch for fin_code {fin_code}: {db_err}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error during specific fetch for fin_code {fin_code}: {e}")
        return None
    finally:
        if conn:
            conn.close()


@log_call(logging.DEBUG)
def get_screener_data(keyword: str) -> Dict[str, Any]:
    """Fetches data for a specific screener keyword."""
    conn = None
    logger.debug(f"Fetching screener data for keyword: {keyword}")
    # Default return structure
    default_response = {"keyword": keyword, "title": keyword, "description": "Screener not found or error.", "total_companies": 0, "companies": []}
    try:
        conn = get_db_connection()
        cur = None
        is_dict_cursor = False
        # Use DictCursor for easier access to screener info
        if psycopg2_extras:
             try: cur = conn.cursor(cursor_factory=psycopg2_extras.DictCursor); is_dict_cursor = True
             except Exception as e: logger.warning(...); cur = conn.cursor()
        else: cur = conn.cursor()

        cur.execute('SELECT keyword, title, description, fin_codes FROM screeners WHERE keyword = %s', (keyword,))
        screener_info_row = cur.fetchone()

        if not screener_info_row:
            logger.warning(f"Screener keyword '{keyword}' not found in database.")
            cur.close()
            return default_response

        # Convert screener info row to dict
        if is_dict_cursor:
            screener_info = dict(screener_info_row)
        else:
            cols = [desc[0] for desc in cur.description]
            screener_info = dict(zip(cols, screener_info_row))

        fin_codes_str = screener_info.get('fin_codes')
        if not fin_codes_str:
            logger.warning(f"Screener '{keyword}' found but has no associated fin_codes.")
            cur.close()
            return make_json_safe({
                "keyword": keyword,
                "title": screener_info.get('title', keyword),
                "description": screener_info.get('description', ''),
                "total_companies": 0,
                "companies": []
            })

        # Process fin_codes list
        fin_codes = [code.strip() for code in fin_codes_str.split(',') if code.strip()]
        total_companies = len(fin_codes)
        logger.info(f"Screener '{keyword}' has {total_companies} companies.")

        # Fetch example company details (limit to 10)
        companies = []
        limit = 10
        if fin_codes:
            fin_codes_to_fetch = tuple(fin_codes[:limit])
            # Use IN operator for multiple codes, handle single code case
            if len(fin_codes_to_fetch) > 0:
                placeholders = '%s' if len(fin_codes_to_fetch) == 1 else ', '.join(['%s'] * len(fin_codes_to_fetch))
                sql = f'SELECT fin_code, comp_name, symbol, sector FROM company_master WHERE fin_code IN ({placeholders})'
                cur.execute(sql, fin_codes_to_fetch)
                company_cols = [desc[0] for desc in cur.description]
                companies = [dict(zip(company_cols, row)) for row in cur.fetchall()]
                logger.debug(f"Fetched {len(companies)} example companies for screener '{keyword}'.")

        cur.close()
        return make_json_safe({
            "keyword": keyword,
            "title": screener_info.get('title', keyword),
            "description": screener_info.get('description', ''),
            "total_companies": total_companies,
            "companies": companies
        })

    except psycopg2.Error as db_err:
        logger.exception(f"Error getting screener data for keyword '{keyword}': {db_err}")
        return default_response # Return default on error
    except Exception as e:
        logger.exception(f"Unexpected error getting screener data for keyword '{keyword}': {e}")
        return default_response # Return default on error
    finally:
        if conn:
            conn.close()


def generate_direct_answer(question: str, company_data: Dict[str, Any], company_name: str) -> Optional[str]:
    """Attempts to generate a direct answer from fetched DB data based on keywords."""
    question_lower = question.lower()
    answer = None
    logger.debug(f"Attempting direct answer generation for '{question}' with data keys: {list(company_data.keys())}")

    # Define specific patterns and the EXACT field they map to
    # Order matters - most specific first!
    specific_patterns = [
        # Prices
        (r'\bnse\s+open(ing)?\s+price\b', 'nse_todays_open', "Today's opening price for {name} on NSE was ₹{val:.2f}."),
        (r'\bbse\s+open(ing)?\s+price\b', 'bse_todays_open', "Today's opening price for {name} on BSE was ₹{val:.2f}."),
        (r'\bnse\s+(clos(e|ing)|last)\s+price\b', 'nse_last_closed_price', "Last closing price of {name} on NSE: ₹{val:.2f}."),
        (r'\bbse\s+(clos(e|ing)|last)\s+price\b', 'bse_last_closed_price', "Last closing price of {name} on BSE: ₹{val:.2f}."),
        (r'\bnse\s+(day|todays)?\s*high\b', 'nse_todays_high', "Today's high for {name} on NSE: ₹{val:.2f}."),
        (r'\bbse\s+(day|todays)?\s*high\b', 'bse_todays_high', "Today's high for {name} on BSE: ₹{val:.2f}."),
        (r'\bnse\s+(day|todays)?\s*low\b', 'nse_todays_low', "Today's low for {name} on NSE: ₹{val:.2f}."),
        (r'\bbse\s+(day|todays)?\s*low\b', 'bse_todays_low', "Today's low for {name} on BSE: ₹{val:.2f}."),
        (r'\b(price|ltp|cmp)\b.*\b(on|for)\s+nse\b', 'nse_last_closed_price', "Last closing price of {name} on NSE: ₹{val:.2f}."), # Price on NSE
        (r'\b(price|ltp|cmp)\b.*\b(on|for)\s+bse\b', 'bse_last_closed_price', "Last closing price of {name} on BSE: ₹{val:.2f}."), # Price on BSE

        # Limits
        (r'\bnse\s+upper\s+(limit|band|circuit)\b', 'nse_upper_limit', "NSE upper circuit limit for {name}: ₹{val:.2f}."),
        (r'\bbse\s+upper\s+(limit|band|circuit)\b', 'bse_upper_limit', "BSE upper circuit limit for {name}: ₹{val:.2f}."),
        (r'\bnse\s+lower\s+(limit|band|circuit)\b', 'nse_lower_limit', "NSE lower circuit limit for {name}: ₹{val:.2f}."),
        (r'\bbse\s+lower\s+(limit|band|circuit)\b', 'bse_lower_limit', "BSE lower circuit limit for {name}: ₹{val:.2f}."),
        (r'\bcircuit\s+limit(s)?\b', ['nse_lower_limit', 'nse_upper_limit', 'bse_lower_limit', 'bse_upper_limit'], "Circuit limits..."), # Special multi-field handling

        # Ratios & Financials
        (r'\bp(.)?e\s+ratio\b', 'pe_ratio', "P/E Ratio for {name}: {val:.2f}."),
        (r'\b(market cap|mcap|market capitalization)\b', 'market_capital', "Market Cap of {name}: {val}."), # Special formatting needed
        (r'\bbook value\b', 'book_value', "Book Value per share for {name}: ₹{val:.2f}."),
        (r'\beps\b|\bearnings per share\b', 'eps', "EPS (Earnings Per Share) for {name}: ₹{val:.2f}."),
        (r'\bface value\b', 'face_value', "Face value for {name}: ₹{val:.2f}."),
        (r'\bp(.)?b\s+ratio\b|\bprice to book\b', 'pb_ratio', "P/B (Price to Book) ratio for {name}: {val:.2f}."),

        # Company Info
        (r'\bsector\b', 'sector', "{name} belongs to the {val} sector."),
        (r'\bindustry\b', 'industry', "{name} operates in the {val} industry."),
        (r'\b(symbol|ticker)\b', 'symbol', "The primary ticker symbol for {name} is {val}."),
        (r'\bisin\b', 'isin', "The ISIN for {name} is {val}."),
        (r'\bbse\s+symbol\b', 'bse_symbol', "The BSE symbol for {name} is {val}."),

        # Verdicts
        (r'\bshort term verdict\b|\bshort term view\b', 'short_term_verdict', "{name} Short-term Verdict: '{val}'."),
        (r'\blong term verdict\b|\blong term view\b', 'long_term_verdict', "{name} Long-term Verdict: '{val}'."),
        (r'\bverdict\b|\boutlook\b|\brecommendation\b', ['short_term_verdict', 'long_term_verdict'], "Verdicts..."), # Special multi-field handling

        # Other
        (r'\blot size\b', 'lot_size', "Lot size for {name}: {val}."),
        (r'\boi\b|open interest\b', 'oi', "Open Interest for {name}: {val}."), # Assuming OI is formatted correctly
    ]

    matched_value = None
    formatting_string = None
    matched_pattern_keyword = None
    data_keys_available = set(company_data.keys()) # Keys available in the fetched data

    for pattern, db_field_or_list, fmt_string in specific_patterns:
        if re.search(pattern, question_lower):
            matched_pattern_keyword = pattern # Log which pattern matched
            logger.debug(f"Direct answer pattern matched: '{pattern}'")

            if isinstance(db_field_or_list, list):
                # Handle multi-field case (e.g., verdicts, limits)
                required_fields = set(db_field_or_list)
                if required_fields.issubset(data_keys_available): # Check if ALL required fields are available
                    temp_data = {f: company_data.get(f) for f in db_field_or_list}
                    # Check if at least one value is not None
                    if any(v is not None for v in temp_data.values()):
                         matched_value = temp_data # Pass the dict of values
                         formatting_string = fmt_string
                         logger.debug(f"Multi-field pattern '{pattern}' matched with available data.")
                         break # Found match, stop searching
                    else:
                         logger.debug(f"Multi-field pattern '{pattern}' matched, but all required data fields are None.")
                         answer = f"The specific data ({', '.join(required_fields)}) for {company_name} is currently unavailable."
                         return answer # Return unavailable message
                else:
                     missing_fields = required_fields - data_keys_available
                     logger.info(f"Multi-field pattern '{pattern}' matched, but required data fields {missing_fields} were not fetched or available.")
                     # Decide if partial answer is okay or state unavailable
                     answer = f"Some information ({', '.join(missing_fields)}) needed for this query about {company_name} was not available."
                     return answer # Return partial/unavailable message

            else:
                # Handle single-field case
                db_field = db_field_or_list
                if db_field in data_keys_available:
                    value = company_data[db_field]
                    if value is not None:
                        matched_value = value
                        formatting_string = fmt_string
                        logger.debug(f"Single-field pattern '{pattern}' matched with available data for field '{db_field}'.")
                        break # Found match, stop searching
                    else:
                        logger.debug(f"Pattern '{pattern}' matched, but data field '{db_field}' is None.")
                        answer = f"The specific data ({db_field.replace('_', ' ')}) for {company_name} is currently unavailable."
                        return answer # Return unavailable message
                else:
                    logger.info(f"Pattern '{pattern}' matched, but data field '{db_field}' was not fetched or available.")
                    # This might happen if Rasa's table suggestion was wrong, or DB query failed partially
                    answer = f"The information for '{db_field.replace('_', ' ')}' for {company_name} was not available in the retrieved data."
                    return answer # Return unavailable message

    # --- Format the Answer if a match was found ---
    if matched_value is not None and formatting_string is not None:
        try:
            # Handle special multi-field formatting
            if formatting_string == "Circuit limits...":
                 nse_low = matched_value.get('nse_lower_limit'); nse_high = matched_value.get('nse_upper_limit')
                 bse_low = matched_value.get('bse_lower_limit'); bse_high = matched_value.get('bse_upper_limit')
                 parts = []
                 if nse_low is not None and nse_high is not None: parts.append(f"NSE Limits: ₹{nse_low:.2f}-₹{nse_high:.2f}")
                 if bse_low is not None and bse_high is not None: parts.append(f"BSE Limits: ₹{bse_low:.2f}-₹{bse_high:.2f}")
                 answer = f"Circuit limits for {company_name}: {'; '.join(parts)}." if parts else f"Circuit limit information for {company_name} is unavailable."
            elif formatting_string == "Verdicts...":
                st = matched_value.get('short_term_verdict'); lt = matched_value.get('long_term_verdict')
                if st and lt: answer = f"{company_name}: Short-term: '{st}', Long-term: '{lt}'."
                elif st: answer = f"{company_name} Short-term: '{st}'."
                elif lt: answer = f"{company_name} Long-term: '{lt}'."
                else: answer = f"Verdicts for {company_name} are unavailable."
            # Handle Market Cap formatting
            elif "Market Cap" in formatting_string:
                 mcap = matched_value
                 mcap_str = f"{mcap}" # Default string representation
                 try:
                     mcap_f = float(mcap)
                     if mcap_f >= 1e7: mcap_str = f"₹{mcap_f/1e7:.2f} Cr" # Crores
                     elif mcap_f >= 1e5: mcap_str = f"₹{mcap_f/1e5:.2f} Lac" # Lakhs (Optional)
                     else: mcap_str = f"₹{mcap_f:.2f}" # Direct value if small
                 except (ValueError, TypeError):
                     pass # Keep original string if conversion fails
                 answer = formatting_string.format(name=company_name, val=mcap_str)
            # General single-value formatting
            else:
                 # Basic type checking for formatting
                 if isinstance(matched_value, (int, float)) and "{val:.2f}" in formatting_string:
                     answer = formatting_string.format(name=company_name, val=matched_value)
                 elif isinstance(matched_value, str):
                      answer = formatting_string.format(name=company_name, val=matched_value)
                 else: # Fallback for other types or format mismatches
                     answer = formatting_string.format(name=company_name, val=str(matched_value))


            if answer:
                logger.info(f"Generated DIRECT answer using pattern '{matched_pattern_keyword}'.")
                return answer
            else:
                 logger.warning(f"Formatting resulted in empty answer for pattern '{matched_pattern_keyword}'.")

        except (TypeError, ValueError, KeyError, AttributeError, IndexError) as fmt_err:
            logger.error(f"Formatting error for pattern '{matched_pattern_keyword}' with value '{matched_value}': {fmt_err}", exc_info=True)
            # Fall through, maybe LLM can handle it

    # --- Fallback (Optional - kept from original code, review if needed) ---
    # Consider removing this block if Rasa intent + specific field fetch is reliable enough
    # This block checks for broader keywords if NO specific pattern matched above.
    if answer is None:
        field_map_fallback = {
            "price": ["nse_last_closed_price", "bse_last_closed_price"],
            "verdict": ["short_term_verdict", "long_term_verdict"],
            "limit": ["nse_lower_limit", "nse_upper_limit", "bse_lower_limit", "bse_upper_limit"],
        }
        found_data_fallback = {}; target_keyword_fallback = None
        for keyword, db_fields in field_map_fallback.items():
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, question_lower):
                target_keyword_fallback = keyword
                for db_field in db_fields:
                    # Check if fallback field exists in the *already fetched data*
                    if db_field in company_data and company_data[db_field] is not None:
                        found_data_fallback[db_field] = company_data[db_field]
                if found_data_fallback: break # Found data for this keyword

        if found_data_fallback and target_keyword_fallback:
             kw = target_keyword_fallback; data = found_data_fallback
             logger.debug(f"Attempting fallback direct answer generation for keyword '{kw}'")
             # Re-use formatting logic from specific patterns if possible
             if kw == "price":
                 p_nse = data.get('nse_last_closed_price'); p_bse = data.get('bse_last_closed_price')
                 if p_nse is not None: answer = f"Last closing price of {company_name} on NSE: ₹{p_nse:.2f}."
                 elif p_bse is not None: answer = f"Last closing price of {company_name} on BSE: ₹{p_bse:.2f}."
             elif kw == "verdict":
                 st = data.get('short_term_verdict'); lt = data.get('long_term_verdict')
                 if st and lt: answer = f"{company_name}: Short-term: '{st}', Long-term: '{lt}'."
                 elif st: answer = f"{company_name} Short-term: '{st}'."
                 elif lt: answer = f"{company_name} Long-term: '{lt}'."
             elif kw == "limit":
                 nse_low=data.get('nse_lower_limit'); nse_high=data.get('nse_upper_limit')
                 bse_low=data.get('bse_lower_limit'); bse_high=data.get('bse_upper_limit')
                 parts = []
                 if nse_low is not None and nse_high is not None: parts.append(f"NSE Limits: ₹{nse_low:.2f}-₹{nse_high:.2f}")
                 if bse_low is not None and bse_high is not None: parts.append(f"BSE Limits: ₹{bse_low:.2f}-₹{bse_high:.2f}")
                 if parts: answer = f"Circuit limits for {company_name}: {'; '.join(parts)}."

             if answer: logger.info(f"Generated DIRECT answer (fallback) for keyword '{kw}'."); return answer
             else: logger.warning(f"Data found for fallback keyword '{kw}' but no specific formatting applied.")

    # If still no answer
    logger.debug(f"No specific direct answer rule applied or formatted for: '{question}'")
    return None


@log_call(logging.DEBUG)
def get_rasa_response_payload(question: str, rasa_url: str = "http://localhost:5005") -> Optional[Dict[str, Any]]: #TODO:check
    """Sends question to Rasa REST webhook, returns 'custom' JSON payload."""
    if not rasa_url:
        logger.error("Rasa URL is not configured. Cannot call Rasa.")
        return None
    webhook_endpoint = f"{rasa_url.rstrip('/')}/webhooks/rest/webhook"
    # Use a unique but potentially predictable sender ID for debugging Rasa sessions if needed
    sender_id = f"backend_caller_{hashlib.sha1(question.encode()).hexdigest()[:10]}"
    payload = {"sender": sender_id, "message": question}

    try:
        logger.debug(f"Sending request to Rasa: {webhook_endpoint} | Sender: {sender_id}")
        # Increased timeout slightly for NLU processing
        res = requests.post(webhook_endpoint, json=payload, timeout=15)
        res.raise_for_status() # Raises HTTPError for bad responses (4xx or 5xx)
        response_data = res.json()

        logger.debug(f"Raw response from Rasa: {json.dumps(response_data, indent=2)}")

        if not isinstance(response_data, list):
            logger.warning(f"Rasa response format unexpected (not a list): {response_data}")
            return None

        # Find the first message with a non-empty 'custom' dictionary payload
        for msg in reversed(response_data): # Check last messages first, often action results
            if isinstance(msg, dict) and "custom" in msg and isinstance(msg["custom"], dict) and msg["custom"]:
                custom_payload = msg["custom"]
                logger.info(f"Extracted 'custom' payload from Rasa for question: '{question}'")
                # Basic validation of expected keys (optional)
                if custom_payload.get("query_intent"):
                     return custom_payload # Return the whole dictionary
                else:
                     logger.warning(f"Rasa 'custom' payload missing 'query_intent': {custom_payload}")
            elif isinstance(msg, dict) and msg.get("text"):
                 logger.debug(f"Rasa response contains text message: '{msg['text'][:100]}...'")


        logger.warning(f"No message with a valid 'custom' dictionary found in Rasa response for: '{question}'")
        return None

    except requests.exceptions.Timeout:
        logger.error(f"Connection to Rasa webhook ({webhook_endpoint}) timed out.")
        return None
    except requests.exceptions.ConnectionError as e:
        logger.error(f"Connection error connecting to Rasa ({webhook_endpoint}): {e}")
        return None
    except requests.exceptions.RequestException as e:
        # Catches other HTTP errors (4xx, 5xx) after raise_for_status()
        logger.error(f"HTTP error communicating with Rasa ({webhook_endpoint}): {e} | Response: {getattr(e.response, 'text', 'N/A')[:500]}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Failed to decode JSON response from Rasa ({webhook_endpoint}): {e}. Response text: {res.text[:500]}...")
        return None
    except Exception as e:
        # Catch any other unexpected errors during the call
        logger.exception(f"Unexpected error while calling Rasa webhook ({webhook_endpoint}): {e}")
        return None

# ==============================================================================
# LLM Interaction Functions
# ==============================================================================

@log_call(logging.INFO)
def call_gemini_api(prompt_text: str, *, is_json_output: bool = False) -> Union[str, Dict]:
    """Low-level wrapper for calling Google Gemini API."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.error("GEMINI_API_KEY environment variable not set.")
        return {"error": "API key missing"} if is_json_output else "Configuration Error: Gemini API key is missing."

    # Use the specified model, default to flash if needed
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
    # Construct URL based on Google AI documentation
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    payload = {
        "contents": [{"parts": [{"text": prompt_text}]}],
        # Add safety settings if desired
        # "safetySettings": [
        #     {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        #     {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"}
        # ],
        "generationConfig": {
            "temperature": 0.6, # Adjust temperature for creativity vs predictability
            "topP": 0.9,        # Adjust top-p sampling if needed
            "topK": 40,         # Adjust top-k sampling if needed
            "maxOutputTokens": 1024, # Limit response length
        }
    }
    if is_json_output:
        payload["generationConfig"]["response_mime_type"] = "application/json"

    headers = {"Content-Type": "application/json"}
    timeout_seconds = 90 # Generous timeout for potentially complex generation

    logger.info("Calling Gemini API → Model=%s | JSON Output=%s | Prompt Length=%d", model, is_json_output, len(prompt_text))
    logger.debug("Gemini Request Payload (Generation Config): %s", payload.get("generationConfig"))

    try:
        res = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
        res.raise_for_status() # Check for HTTP errors
        data = res.json()
        logger.debug("Gemini Raw Response Snippet: %s", str(data)[:500])

        # Check for explicit blocks or errors in the response structure
        if not data.get("candidates") and data.get("promptFeedback"):
            block_reason = data["promptFeedback"].get("blockReason")
            safety_ratings = data["promptFeedback"].get("safetyRatings")
            logger.error(f"Gemini request blocked. Reason: {block_reason}, Ratings: {safety_ratings}")
            error_msg = f"Request blocked due to content policy ({block_reason})."
            return {"error": "Blocked Content", "details": block_reason} if is_json_output else error_msg

        # Extract text content safely
        raw_text = data["candidates"][0]["content"]["parts"][0].get("text", "")

    except requests.exceptions.Timeout:
        logger.exception("Gemini API request timed out after %d seconds.", timeout_seconds)
        return {"error": "API Timeout"} if is_json_output else "Error: The request to the language model timed out."
    except requests.exceptions.RequestException as exc:
        error_detail = str(exc)
        if exc.response is not None:
             try: error_detail = f"{exc} | Response: {exc.response.json()}"
             except: error_detail = f"{exc} | Response: {exc.response.text[:200]}" # Fallback if not JSON
        logger.exception("Gemini API request failed: %s", error_detail)
        return {"error": "API Request Failed", "details": str(exc)} if is_json_output else f"Error: Could not communicate with the language model ({exc})."
    except (KeyError, IndexError, TypeError) as e:
        logger.exception("Failed to parse Gemini response structure: %s. Response: %s", e, str(data)[:500])
        return {"error": "Malformed API Response", "details": str(e)} if is_json_output else "Error: Received an invalid response from the language model."
    except Exception as e: # Catch any other unexpected errors
         logger.exception("Unexpected error during Gemini API call: %s", e)
         return {"error": "Unexpected Error", "details": str(e)} if is_json_output else "An unexpected error occurred while processing the request."


    # Process the extracted text
    if is_json_output:
        # Attempt to parse the potentially JSON formatted string
        cleaned = raw_text.strip()
        # Remove markdown code block fences if present
        if cleaned.startswith("```json"): cleaned = cleaned[7:-3].strip()
        elif cleaned.startswith("```"): cleaned = cleaned[3:-3].strip()
        try:
            parsed_json = json.loads(cleaned)
            logger.info("Gemini JSON output parsed successfully.")
            return parsed_json
        except json.JSONDecodeError as json_err:
            logger.error(f"Gemini response failed JSON parsing: {json_err}. Raw text: '{cleaned[:200]}...'")
            # Return error dict, maybe include raw text for debugging
            return {"error": "JSON Parse Error", "details": str(json_err), "raw_text": raw_text}
    else:
        # Return plain text response
        logger.info("Gemini text output received successfully (Length: %d).", len(raw_text))
        return raw_text


def get_llm_answer(prompt: str, is_json_output: bool = False, original_question: Optional[str] = None, company_data: Optional[Dict] = None, screener_data: Optional[Dict] = None) -> Union[str, Dict]:
    """Gets answer from LLM, logs interaction, handles errors."""
    llm_response = call_gemini_api(prompt, is_json_output=is_json_output)

    # Handle Error Responses from call_gemini_api
    if isinstance(llm_response, dict) and "error" in llm_response:
        error_type = llm_response.get("error", "Unknown Error")
        logger.error(f"LLM call failed with error: {error_type}. Details: {llm_response.get('details')}")
        # Return a user-friendly error message or the error dict depending on context
        return llm_response if is_json_output else f"Sorry, I encountered an error while processing that: {error_type}."

    # Log successful interactions (only for text responses for now)
    if not is_json_output and isinstance(llm_response, str) and original_question:
        data_hash = None
        log_data_ref = None
        if company_data:
            data_hash = compute_data_hash(company_data) # Assumes compute_data_hash exists
            log_data_ref = "company"
        elif screener_data:
            data_hash = compute_data_hash(screener_data)
            log_data_ref = "screener"

        # Log entry structure
        log_entry = {
            "ts": time.time(),
            "q": original_question,
            "p_snippet": prompt[:500] + ("..." if len(prompt) > 500 else ""), # Log snippet of prompt
            "a": llm_response,
            "h": data_hash,
            "ref": log_data_ref,
            "int": "final_answer" # Indicate this is the final answer generation step
        }
        try:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            log_filepath = os.path.join(log_dir,"qa_llm_interactions.jsonl")
            with open(log_filepath, "a", encoding='utf-8') as f:
                f.write(json.dumps(log_entry) + "\n")
            logger.debug(f"LLM interaction logged to {log_filepath}")
        except Exception as log_e:
            logger.error(f"Failed writing LLM interaction log: {log_e}")

    # Return the successful response (string or dictionary)
    return llm_response


@log_call(logging.DEBUG)
def find_screener_by_keywords(keywords: List[str]) -> Optional[str]:
    """Finds the best matching screener keyword using embeddings."""
    build_screener_index() # Ensure index is ready
    if not SCREENER_INDEX or SCREENER_EMBEDDING_MATRIX is None or not embedding_model or not keywords:
        logger.error("Screener index/model not available or no keywords provided.")
        return None

    query = " ".join(keywords).lower().strip()
    if not query: return None

    try:
        query_emb = embedding_model.encode(query, convert_to_tensor=True)
        if query_emb.ndim == 1: query_emb = query_emb.unsqueeze(0)
        if SCREENER_EMBEDDING_MATRIX.ndim != 2 or query_emb.ndim != 2:
            logger.error(f"Screener Dimension mismatch: Query {query_emb.shape}, Matrix {SCREENER_EMBEDDING_MATRIX.shape}")
            return None
        cosine_scores = util.pytorch_cos_sim(query_emb, SCREENER_EMBEDDING_MATRIX)[0]
    except Exception as e:
        logger.exception(f"Error during screener embedding/similarity for query '{query}': {e}")
        return None

    best_idx = int(torch.argmax(cosine_scores))
    best_score = cosine_scores[best_idx].item()

    # Adjust threshold based on testing - may need to be higher/lower
    match_threshold = 0.60 # Example threshold

    if best_score >= match_threshold:
        matched_keyword = SCREENER_INDEX[best_idx]["keyword"]
        logger.info("Screener matched – Keywords='%s' → Matched Keyword: %s (Score: %.3f)", keywords, matched_keyword, best_score)
        return matched_keyword
    else:
        logger.warning("No confident screener match for keywords '%s' (Best Score: %.3f < %.2f)", keywords, best_score, match_threshold)
        return None


def get_full_company_data(fin_code: str) -> Optional[Dict[str, Any]]:
    """Fetches comprehensive data (all tables) for a company. Use Sparingly."""
    conn = None
    logger.debug(f"Fetching FULL data for fin_code: {fin_code}")
    if not fin_code: return None

    full_data = {}
    try:
        conn = get_db_connection()
        dict_cur = None
        is_dict_cursor = False
        if psycopg2_extras:
             try: dict_cur = conn.cursor(cursor_factory=psycopg2_extras.DictCursor); is_dict_cursor = True
             except Exception as e: logger.warning(...); dict_cur = conn.cursor()
        else: dict_cur = conn.cursor()

        # Fetch from company_master
        dict_cur.execute('SELECT * FROM company_master WHERE fin_code = %s', (fin_code,))
        master_data_row = dict_cur.fetchone()
        if not master_data_row:
            logger.warning(f"No master data found for fin_code {fin_code}. Cannot fetch full data.")
            dict_cur.close()
            return None
        master_cols = list(master_data_row.keys()) if is_dict_cursor else [desc[0] for desc in dict_cur.description]
        full_data.update(dict(master_data_row) if is_dict_cursor else dict(zip(master_cols, master_data_row)))

        # Fetch from company_additional_details
        dict_cur.execute('SELECT * FROM company_additional_details WHERE fin_code = %s', (fin_code,))
        additional_data_row = dict_cur.fetchone()
        if additional_data_row:
            is_add_dict = isinstance(additional_data_row, dict) or (psycopg2_extras and isinstance(additional_data_row, psycopg2_extras.DictRow))
            add_cols = list(additional_data_row.keys()) if is_add_dict else [desc[0] for desc in dict_cur.description]
            add_data_dict = dict(additional_data_row) if is_add_dict else dict(zip(add_cols, additional_data_row))
            full_data.update({k: v for k, v in add_data_dict.items() if k != 'fin_code'}) # Avoid overwriting fin_code

        # Fetch from consolidated_company_equity
        dict_cur.execute('SELECT * FROM consolidated_company_equity WHERE fin_code = %s', (fin_code,))
        equity_data_row = dict_cur.fetchone()
        if equity_data_row:
            is_eq_dict = isinstance(equity_data_row, dict) or (psycopg2_extras and isinstance(equity_data_row, psycopg2_extras.DictRow))
            eq_cols = list(equity_data_row.keys()) if is_eq_dict else [desc[0] for desc in dict_cur.description]
            eq_data_dict = dict(equity_data_row) if is_eq_dict else dict(zip(eq_cols, equity_data_row))
            full_data.update({k: v for k, v in eq_data_dict.items() if k != 'fin_code'}) # Avoid overwriting fin_code

        dict_cur.close()
        logger.info(f"Successfully fetched full data for fin_code {fin_code}")
        return make_json_safe(full_data)

    except psycopg2.Error as db_err:
        logger.exception(f"DB error fetching full data for fin_code {fin_code}: {db_err}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error fetching full data for fin_code {fin_code}: {e}")
        return None
    finally:
        if conn:
            conn.close()


# ==============================================================================
# Prompt Templates
# ==============================================================================
# Use f-strings or .format() method for insertion
COMPANY_QUERY_PROMPT_TEMPLATE = """
You are a highly knowledgeable and concise stock market assistant replying on behalf of Univest Stock Advisory. Your tone should be professional, confident, and direct.
**CRITICAL INSTRUCTION:** Use ONLY the provided "Key Company Data Points" below to answer the user's question. Absolutely DO NOT use any external knowledge or information not present in the provided data block. Your entire response MUST be derived solely from the information given here.
- Focus solely on the question asked and the data provided.
- Do NOT mention the source of the data (e.g., "According to the provided data...").
- Avoid technical jargon unless essential or part of the requested data point.
- Be brief and to the point. Minimize filler text.
- If the specific information needed to answer the question is **NOT** present or is `null`/`None` in the "Key Company Data Points", you MUST respond with ONLY ONE of the following phrases, exactly as written:
    - "The specific information requested is not available in the key data points provided."
    - "The key data points provided do not contain the details needed to answer this question."
- DO NOT guess, infer, make calculations (unless explicitly asked and data allows), or retrieve external information. Just state it's unavailable using one of the exact phrases above if the data isn't there.

--- Key Company Data Points ({company_name}) ---
{essential_company_data}
---

User Question: {question}
Answer:"""

COMPARISON_PROMPT_TEMPLATE = """
You are a highly knowledgeable and concise stock market assistant replying on behalf of Univest Stock Advisory. Your tone should be professional, confident, and direct.
**CRITICAL INSTRUCTION:** Use ONLY the provided "Key Company Data Points" for the specified companies to answer the user's comparison question. Absolutely DO NOT use any external knowledge or information not present in the provided data.
- Directly compare the companies based ONLY on the user's question and the available "Key Company Data Points".
- Highlight relevant differences or similarities found ONLY in the provided data.
- Do NOT mention the source of the data.
- Be brief and focus on the comparison requested.
- If a necessary data point for the comparison is **MISSING** or `null`/`None` for one or more companies, you MUST state clearly which information is missing (e.g., "P/E ratio data is not available in the provided key data for Company X.") and DO NOT attempt the comparison for that specific metric. Only compare metrics where data is present for all relevant companies.

--- Key Company Data Points ---
{essential_comparison_data}
---

User Question: {question}
Answer:"""

GENERAL_FINANCE_PROMPT_TEMPLATE = """
You are a knowledgeable and helpful stock market assistant replying on behalf of Univest Stock Advisory. Your tone should be professional, clear, and informative.
Answer the user's general finance or stock market question accurately and concisely. If the question asks for an opinion or prediction, politely state that you provide information based on available data but do not offer financial advice or forecasts. Explain concepts clearly if necessary. Keep the answer focused on the question and relatively brief.

User Question: {question}
Answer:"""

SCREENER_PROMPT_TEMPLATE = """
You are a helpful stock market assistant replying on behalf of Univest Stock Advisory. Your tone should be professional and informative.
The user's question has matched a pre-defined stock screener. Present the results clearly.
- Briefly acknowledge the screener criteria using its title or description.
- State the total number of companies matching the criteria.
- List up to 10 example companies found by the screener, including their name and sector if available. Format as a list or table for readability.
- Do NOT provide analysis, opinions, or investment advice about the stocks listed unless specifically asked in the original question (which is unlikely for a screener match). Just present the screener results.
- Be concise and well-formatted.

--- Screener Data ({screener_title}) ---
Keyword: {keyword}
Description: {description}
Total Companies Found: {total_companies}
Example Companies:
{companies_list_str}
---

Original User Question (for context only): {question}

Answer acknowledging the screener results:"""

# ==============================================================================
# Flask Routes
# ==============================================================================

@session_bp.route('/smart-ask', methods=['POST'])
# @log_call(logging.INFO) # Add decorator if needed for overall route timing/logging
def smart_ask():
    """ Handles user questions using Rasa for intent/table detection and DB-first approach. """
    start_time = time.time()
    gemini_call_counter = 0 # Track LLM calls per request
    session_id = None # Initialize session_id
    response_data = {"type": "error_initialization"} # Default error type

    try:
        data = request.get_json()
        if not data: raise ValueError("Request body is empty or not valid JSON.")
        question = data.get("question", "").strip()
        session_id = data.get("session_id") # Capture session ID if provided
        resolve_context_id = data.get("resolve_ambiguity_context_id")
        selected_fin_code = data.get("selected_fin_code")

        if not question and not (resolve_context_id and selected_fin_code):
            logger.error("Received empty question and no ambiguity context.")
            return jsonify({"error": "Empty question received."}), 400

    except Exception as req_err:
        logger.error(f"Failed to parse request body: {req_err}", exc_info=True)
        return jsonify({"error": "Malformed request body or missing question."}), 400

    # --- Initialize variables ---
    company_match = None
    fin_code = None
    company_official_name = None
    is_ambiguity_resolved = False
    original_question = question # Store original question text

    # --- Ambiguity Resolution Logic ---
    if resolve_context_id and selected_fin_code:
        logger.info(f"Attempting ambiguity resolution: ContextID={resolve_context_id}, SelectedFinCode={selected_fin_code} | Session: {session_id}")
        # Use a thread-safe way to access/modify cache if running multi-threaded
        context_data = AMBIGUITY_CACHE.get(resolve_context_id)
        if context_data and (time.time() - context_data.get('timestamp', 0) <= AMBIGUITY_CACHE_EXPIRY):
            original_question = context_data['question'] # Use original question from stored context
            logger.info(f"Found ambiguity context. Original Question: '{original_question}'")
            build_company_name_index() # Ensure index is ready
            # Find the selected company details from index
            company_match_resolved = next((c for c in COMPANY_NAME_INDEX if str(c.get("fin_code")) == str(selected_fin_code)), None)

            if not company_match_resolved:
                logger.error(f"Selected fin_code {selected_fin_code} not found in company index during ambiguity resolution.")
                return jsonify({
                    "type": "error_processing_selection",
                    "message": "Sorry, I couldn't find the details for the company you selected. Please try asking again.",
                    "source": "system_error"
                }), 200 # 200 OK, but with error payload

            # Successfully resolved
            fin_code = selected_fin_code
            company_official_name = company_match_resolved.get("name", "Selected Company")
            company_match = company_match_resolved # Store full match info
            is_ambiguity_resolved = True
            logger.info(f"Ambiguity resolved to company: {company_official_name} ({fin_code})")
            # Remove context from cache after successful resolution
            try:
                del AMBIGUITY_CACHE[resolve_context_id]
            except KeyError:
                logger.warning(f"Attempted to delete ambiguity context {resolve_context_id}, but it was already gone.")
        else:
            # Context not found or expired
            logger.warning(f"Ambiguity context {resolve_context_id} expired or not found. Session: {session_id}")
            return jsonify({
                "type": "error_context_lost",
                "message": "Sorry, the selection context has expired. Could you please ask your original question again?",
                "source": "system_error"
            }), 200 # 200 OK, but with error payload
    # --- End Ambiguity Resolution ---

    # --- DB Connection Check (Essential before proceeding) ---
    try:
        # Use context manager for safety if get_db_connection supports it
        # Or manually manage connection closing in a finally block if not
        conn_test = get_db_connection()
        if not conn_test: raise ConnectionError("get_db_connection returned None.")
        conn_test.close() # Close the test connection
        logger.debug("Initial DB connection check successful.")
    except Exception as db_conn_err:
        logger.critical(f"Initial DB connection check failed: {db_conn_err}", exc_info=True)
        return jsonify({"type": "database_error", "message": "System error: Cannot connect to the data source at the moment. Please try again later.", "source": "system_error"}), 503 # Service Unavailable

    # --- Rasa Interaction ---
    rasa_payload = None
    rasa_url = os.getenv("RASA_WEBHOOK_URL", "http://localhost:5005") #TODO:check Get Rasa URL from env


    # Call Rasa unless we just resolved ambiguity and lack the original question somehow
    if (is_ambiguity_resolved and original_question) or (not is_ambiguity_resolved and original_question):
         log_msg = f"Calling Rasa for question: '{original_question}' | Session: {session_id}"
         if is_ambiguity_resolved: log_msg += " (Post-Ambiguity Resolution)"
         logger.info(log_msg)
         rasa_payload = get_rasa_response_payload(original_question, rasa_url=rasa_url)
    else:
         logger.warning(f"Skipping Rasa call: No question available. is_ambiguity_resolved={is_ambiguity_resolved}")


    # --- Main Logic based on Rasa Payload ---
    if rasa_payload:
        rasa_intent = rasa_payload.get("query_intent")

        # === NEW LOGIC: CORRECTED COMPANY NAME EXTRACTION ===
        # Prioritize resolved_company, fall back to extracted_company if resolved is null/empty
        company_name_from_rasa = rasa_payload.get("resolved_company")
        if not company_name_from_rasa:
            extracted_name = rasa_payload.get("extracted_company")
            if extracted_name: # Ensure it's not None or empty string
                company_name_from_rasa = extracted_name
                logger.debug(f"Rasa 'resolved_company' was null/empty, falling back to 'extracted_company': '{extracted_name}'")
            else:
                # If both resolved and extracted are null/empty, explicitly set to None
                company_name_from_rasa = None
                logger.debug("Both 'resolved_company' and 'extracted_company' are null or empty in Rasa payload.")
        # =====================================================

        rasa_extracted_concept = rasa_payload.get("extracted_concept") # Single concept from ask_data_source
        rasa_target_concepts = rasa_payload.get("target_concepts") # List from ask_investment_advice
        rasa_source_table = rasa_payload.get("source_table")

        # === CHANGED === Use the potentially corrected name in logging
        logger.info(f"Rasa Payload Parsed: Intent='{rasa_intent}', Company='{company_name_from_rasa}', Concept(s)='{rasa_target_concepts or rasa_extracted_concept}', Table='{rasa_source_table}' | Session: {session_id}")


        # Determine the current company context
        current_company_name = None
        current_fin_code = None
        current_company_match = None

        if is_ambiguity_resolved and fin_code:
            # Priority to the company selected during ambiguity resolution
            current_fin_code = fin_code
            current_company_name = company_official_name
            current_company_match = company_match
            logger.debug(f"Using company from ambiguity resolution: {current_company_name} ({current_fin_code})")
        # === CHANGED === Use the corrected variable here: company_name_from_rasa
        elif company_name_from_rasa:
            # If Rasa identified a company (either resolved or extracted), find its fin_code
            logger.debug(f"Rasa identified company: '{company_name_from_rasa}'. Finding match using backend index...")
            # === CHANGED === Pass the potentially extracted name string to the backend matcher
            match_from_rasa_name = find_company_by_name_or_symbol(company_name_from_rasa)
            if match_from_rasa_name:
                current_fin_code = match_from_rasa_name.get("fin_code")
                # Use official name from index, fallback to name from rasa if index lacks it somehow
                current_company_name = match_from_rasa_name.get("name", company_name_from_rasa)
                current_company_match = match_from_rasa_name
                logger.info(f"Backend matched Rasa company '{company_name_from_rasa}' to: {current_company_name} ({current_fin_code})")
            else:
                # === CHANGED === Store the name that failed to match for clarity in logs/errors
                current_company_name = company_name_from_rasa
                logger.warning(f"Backend could not find a match in index for company '{current_company_name}' identified by Rasa.")
                # fin_code remains None
        else:
            logger.debug("Neither ambiguity resolution nor Rasa identified a specific company.")


        # --- Handle Investment Advice Intent ---
        if rasa_intent == "ask_investment_advice":
            response_data["type"] = "investment_advice"
            logger.debug(f"Processing 'ask_investment_advice' intent for '{current_company_name or 'Unknown Company'}'")

            # === CHANGED === Check fin_code status after potential backend match attempt
            if not current_fin_code:
                logger.warning("Cannot process advice intent: Company fin_code is missing (could not match Rasa name or ambiguity resolution failed).")
                response_data.update({
                    "type": "error_missing_info",
                    # Provide the name Rasa gave if resolution failed to give context
                    "message": f"Could not identify the specific company ('{current_company_name}') for the advice request in my records." if current_company_name else "Please specify which company you'd like advice for."
                })
            elif not rasa_target_concepts or not rasa_source_table:
                logger.warning(f"Cannot process advice intent: Missing target_concepts ({rasa_target_concepts}) or source_table ({rasa_source_table}) in Rasa payload.")
                response_data.update({"type": "error_processing_advice", "message": "Could not fully understand the advice request details from the assistant's analysis."})
            else:
                logger.info(f"Fetching advice data for {current_company_name} ({current_fin_code}). Needs: {rasa_target_concepts} from {rasa_source_table}")
                advice_data = get_specific_company_data(current_fin_code, rasa_target_concepts)

                if advice_data:
                    # (Keep logic for calling LLM for advice interpretation)
                    verdict_prompt = f"""
                    Analyze the following investment verdicts for {current_company_name} and provide a brief summary interpretation.
                    Focus ONLY on the provided verdicts. Do not add external information or opinions. State if data is missing.
                    Short-term Verdict: {advice_data.get('short_term_verdict', 'Not Available')}
                    Long-term Verdict: {advice_data.get('long_term_verdict', 'Not Available')}
                    Summary:"""
                    gemini_call_counter += 1
                    logger.info(f"[Gemini Call #{gemini_call_counter}] for advice interpretation: {current_company_name}")
                    gemini_advice_answer = get_llm_answer(
                        verdict_prompt,
                        original_question=original_question,
                        company_data=advice_data # Log context
                    )
                    response_data.update({
                        "company": current_company_match or {"name": current_company_name, "fin_code": current_fin_code},
                        "answer": gemini_advice_answer,
                        "source": "llm_with_db_verdicts"
                    })
                else:
                    logger.error(f"Failed to fetch advice data ({rasa_target_concepts}) for {current_company_name} ({current_fin_code})")
                    response_data.update({"type": "data_fetch_failed", "message": f"Sorry, I couldn't retrieve the investment outlook data for '{current_company_name}' right now."})

        # --- Handle Standard Data Source Intent ---
        elif rasa_intent == "ask_data_source":
            response_data["type"] = "company_query" # Or "data_source_query"
            logger.debug(f"Processing 'ask_data_source' intent for concept '{rasa_extracted_concept}', table '{rasa_source_table}'")

             # Validate required info
            if not rasa_extracted_concept or not rasa_source_table:
                logger.warning(f"Cannot process data source intent: Missing concept ({rasa_extracted_concept}) or source_table ({rasa_source_table}).")
                response_data.update({"type": "error_processing_data_query", "message": "Could not fully understand which data point you're asking about."})
            # === CHANGED === Check fin_code status after potential backend match attempt
            elif not current_fin_code and rasa_source_table != 'screeners': # Screeners don't always need a company
                logger.warning(f"Cannot process data source intent for concept '{rasa_extracted_concept}': Company fin_code is missing (could not match Rasa name '{company_name_from_rasa}' or ambiguity resolution failed).")
                response_data.update({
                    "type": "error_missing_info",
                    # Provide the name Rasa gave if resolution failed to give context
                    "message": f"Please specify which company you're asking about for '{rasa_extracted_concept.replace('_',' ')}'. I couldn't definitively identify '{current_company_name}' in my records." if current_company_name else f"Please specify which company you're asking about for '{rasa_extracted_concept.replace('_',' ')}'."
                })
            # Handle Screeners Separately
            elif rasa_source_table == 'screeners':
                 # (Keep screener logic as before)
                 logger.info(f"Handling screener request for keyword: '{rasa_extracted_concept}'")
                 screener_data = get_screener_data(rasa_extracted_concept) # Use concept as keyword
                 if screener_data and screener_data.get("total_companies", 0) > 0:
                     companies_list = screener_data.get("companies", [])
                     companies_list_str = "\n".join([f"- {c.get('comp_name', 'N/A')} ({c.get('symbol', 'N/A')}) - {c.get('sector', 'N/A')}" for c in companies_list]) if companies_list else "No specific examples available."
                     prompt = SCREENER_PROMPT_TEMPLATE.format(
                         screener_title=screener_data.get('title', rasa_extracted_concept),
                         keyword=rasa_extracted_concept,
                         description=screener_data.get('description', 'N/A'),
                         total_companies=screener_data.get('total_companies', 0),
                         companies_list_str=companies_list_str,
                         question=original_question
                     )
                     gemini_call_counter += 1
                     logger.info(f"[Gemini Call #{gemini_call_counter}] for screener summary: {rasa_extracted_concept}")
                     answer = get_llm_answer(prompt, original_question=original_question, screener_data=screener_data)
                     response_data.update({
                         "type": "screener_result",
                         "screener_keyword": rasa_extracted_concept,
                         "screener_title": screener_data.get('title', rasa_extracted_concept),
                         "answer": answer,
                         "source": "llm_with_screener_data"
                     })
                 elif screener_data: # Found screener but no companies
                     logger.warning(f"Screener '{rasa_extracted_concept}' found but has no companies.")
                     response_data.update({
                         "type": "screener_result_empty",
                         "screener_keyword": rasa_extracted_concept,
                         "screener_title": screener_data.get('title', rasa_extracted_concept),
                         "answer": f"The screener '{screener_data.get('title', rasa_extracted_concept)}' currently has no companies matching its criteria.",
                         "source": "database"
                     })
                 else: # Screener keyword not found
                     logger.error(f"Screener keyword '{rasa_extracted_concept}' indicated by Rasa was not found in the database.")
                     response_data.update({
                         "type": "screener_not_found",
                         "answer": f"Sorry, I couldn't find a screener matching '{rasa_extracted_concept}'.",
                         "source": "database"
                     })

            # Handle Company-Specific Data Point Queries (Now requires valid current_fin_code)
            else:
                logger.info(f"Fetching specific data for {current_company_name} ({current_fin_code}). Concept: {rasa_extracted_concept}, Table: {rasa_source_table}")

                # Determine fields to fetch based ONLY on the identified source table
                fields_to_get_set = TABLE_TO_FIELDS_MAP.get(rasa_source_table, set())
                if not fields_to_get_set:
                    logger.warning(f"No fields mapped in TABLE_TO_FIELDS_MAP for Rasa table '{rasa_source_table}'. Falling back to ESSENTIAL keys.")
                    fields_to_get_set = set(ESSENTIAL_COMPANY_KEYS) # Fallback
                else:
                    fields_to_get_set = fields_to_get_set.copy() # Avoid modifying original map
                    fields_to_get_set.add(rasa_extracted_concept) # Add the specific concept
                    fields_to_get_set.add("fin_code") # Ensure primary key is always there
                    # Add core identifiers if not already present, helpful for prompts/context
                    fields_to_get_set.update(["comp_name", "symbol"])
                    logger.info(f"Fetching fields defined for table '{rasa_source_table}' (plus essentials): {fields_to_get_set}")

                fields_to_get = list(fields_to_get_set)
                specific_data = get_specific_company_data(current_fin_code, fields_to_get)

                if specific_data:
                    # Ensure company name from DB is used if fetched
                    db_company_name = specific_data.get("comp_name", current_company_name) # Prefer DB name if available
                    logger.debug(f"[Data Fetched from {rasa_source_table}]:\n{json.dumps(specific_data, indent=2, default=str)}")

                    # Attempt direct answer generation first
                    direct_answer = generate_direct_answer(original_question, specific_data, db_company_name)

                    if direct_answer:
                        logger.info(f"✅ DIRECT answer generated for {db_company_name} (Concept: {rasa_extracted_concept})")
                        response_data.update({
                            "company": current_company_match or {"name": db_company_name, "fin_code": current_fin_code},
                            "answer": direct_answer,
                            "source": "database"
                        })
                    else:
                        # Fallback to LLM using ONLY the fetched data
                        logger.info(f"No direct answer. Calling Gemini using data from '{rasa_source_table}'.")
                        essential_data_for_llm = specific_data
                        gemini_call_counter += 1
                        logger.info(f"[Gemini Call #{gemini_call_counter}] for: {db_company_name} | Concept: {rasa_extracted_concept} | Data Source: {rasa_source_table}")
                        logger.debug(f"[Data to Gemini]:\n{json.dumps(essential_data_for_llm, indent=2, default=str)}")
                        prompt = COMPANY_QUERY_PROMPT_TEMPLATE.format(
                            company_name=db_company_name, # Use name from DB
                            essential_company_data=json.dumps(essential_data_for_llm, indent=2),
                            question=original_question
                        )
                        logger.debug(f"[Prompt Sent to Gemini]:\n{prompt[:1000]}...")
                        answer = get_llm_answer(
                            prompt,
                            original_question=original_question,
                            company_data=essential_data_for_llm # Log context
                        )
                        response_data.update({
                            "company": current_company_match or {"name": db_company_name, "fin_code": current_fin_code},
                            "answer": answer,
                            "source": "llm_with_db_data"
                        })
                else:
                    # Failed to fetch even the specific data
                    logger.error(f"Failed to fetch specific data ({fields_to_get}) for {current_company_name} ({current_fin_code}) based on table '{rasa_source_table}'")
                    response_data.update({"type": "data_fetch_failed", "message": f"Sorry, I couldn't retrieve the requested data ({rasa_extracted_concept.replace('_',' ')}) for '{current_company_name}' right now."})

        # --- Handle Other Intents Recognized by Rasa (e.g., greet, goodbye) ---
        else:
            # (Keep logic for other intents as before)
            logger.info(f"Handling non-data/advice Rasa intent: '{rasa_intent}' | Session: {session_id}")
            response_data["type"] = f"general_fallback_rasa_{rasa_intent}"
            prompt = GENERAL_FINANCE_PROMPT_TEMPLATE.format(question=original_question)
            answer = get_llm_answer(prompt, original_question=original_question)
            response_data.update({"answer": answer, "source": "llm_only"})

    # --- Fallback if Rasa Interaction Failed or yielded no usable payload ---
    else:
        # (Keep fallback logic as before)
        logger.warning(f"Rasa did not return a usable payload. Falling back to general LLM processing. | Session: {session_id} | Question: '{original_question}'")
        response_data["type"] = "general_fallback_no_rasa"
        prompt = GENERAL_FINANCE_PROMPT_TEMPLATE.format(question=original_question)
        gemini_call_counter += 1
        logger.info(f"[Gemini Call #{gemini_call_counter}] General Fallback (No Rasa Payload)")
        # answer = get_llm_answer(prompt, original_question=original_question)
        # response_data.update({"answer": answer, "source": "llm_only"})


    # --- Final Response Assembly ---
    end_time = time.time()
    processing_time = round((end_time - start_time) * 1000)
    response_data["processing_time_ms"] = processing_time

    # Ensure there's always an answer field, provide default error if needed
    if "answer" not in response_data or not response_data.get("answer"):
        # === CHANGED === Add more context to final error if possible
        error_message = "Sorry, I encountered an issue processing your request. Please try rephrasing or ask again later."
        if response_data.get("type") == "error_missing_info" and response_data.get("message"):
             error_message = response_data["message"] # Use the specific error message if available
        logger.error(f"Processing finished but no answer generated for: '{original_question}' | Final Response Data: {response_data}")
        response_data["answer"] = error_message
        response_data["type"] = response_data.get("type", "error_unknown") # Keep error type if set, else add generic one
        response_data["source"] = response_data.get("source", "system_error")

    logger.info(f"Request processed in {processing_time} ms | Final Type: {response_data.get('type')} | Source: {response_data.get('source')} | Gemini Calls: {gemini_call_counter} | Session: {session_id}")

    # Before returning, clean up any expired ambiguity contexts
    cleanup_expired_ambiguity_cache() # Assuming this function exists

    return jsonify(response_data), 200


# ==============================================================================
# Other Helper Routes
# ==============================================================================

@session_bp.route('/llm-status', methods=['GET'])
def llm_status():
    """Checks connectivity and basic response from the configured LLM."""
    logger.info("Checking LLM status...")
    try:
        # Use a simple, safe prompt
        prompt = "Respond with 'Operational' if you are functioning correctly."
        response = call_gemini_api(prompt) # Expects text response

        if isinstance(response, str) and "operational" in response.lower():
            logger.info("LLM status check: OK")
            return jsonify({"status": "ok", "response": response}), 200
        elif isinstance(response, dict) and "error" in response:
            logger.error(f"LLM status check failed: {response}")
            return jsonify({"status": "error", "details": response}), 503 # Service Unavailable
        else:
            logger.warning(f"LLM status check returned unexpected response: {str(response)[:200]}")
            return jsonify({"status": "error", "message": "Unexpected response from LLM.", "response": str(response)[:200]}), 500
    except Exception as e:
        logger.exception("Exception during LLM status check.")
        return jsonify({"status": "error", "details": str(e)}), 500


@session_bp.route('/reload-indices', methods=['POST'])
def reload_indices():
    """Manually triggers reloading of embedding indices."""
    # TODO: Add authentication/authorization if this is exposed externally
    # Example: Check for a specific header or IP address
    # if not request.headers.get('X-Admin-Key') == os.environ.get('ADMIN_RELOAD_KEY'):
    #     logger.warning("Unauthorized attempt to reload indices.")
    #     return jsonify({"error": "Unauthorized"}), 403

    logger.info("Manual index reload triggered via API.")
    results = {}
    status_code = 200

    try:
        build_company_name_index()
        results["company_index"] = f"Reloaded {len(COMPANY_NAME_INDEX)} companies."
        logger.info(results["company_index"])
    except Exception as e:
        results["company_index"] = f"Error: {e}"
        logger.error(f"API Reload Error (Company Index): {e}", exc_info=True)
        status_code = 500

    try:
        build_screener_index()
        results["screener_index"] = f"Reloaded {len(SCREENER_INDEX)} screeners."
        logger.info(results["screener_index"])
    except Exception as e:
        results["screener_index"] = f"Error: {e}"
        logger.error(f"API Reload Error (Screener Index): {e}", exc_info=True)
        status_code = 500

    # Add QA store reload if implemented
    # try: load_local_qa_store(); results["qa_store"] = "..."; logger.info(...)
    # except Exception as e: results["qa_store"] = f"Error: {e}"; logger.error(...); status_code = 500


    logger.info(f"Index reload finished with status: {status_code}")
    return jsonify({"status": "Reload completed.", "results": results}), status_code

# ==============================================================================
# Helper Functions
# ==============================================================================

def load_local_qa_store():
    """Loads Q&A pairs from a local file (Optional)."""
    global qa_store, qa_embeddings
    _initialize_embedding_model() # Ensure model loaded first
    qa_store = []
    embeddings = []
    # Make filepath configurable or relative
    filepath = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "qa_training_data.jsonl") # Adjust path
    logger.info(f"Attempting to load local Q&A store from: {filepath}")
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try: item = json.loads(line)
                except json.JSONDecodeError: logger.warning(f"Skipping invalid JSON line {line_num} in QA store: {line.strip()}"); continue
                except Exception as e: logger.error(f"Error reading line {line_num} from QA store: {e}"); continue

                if item.get("prompt") and item.get("answer") and embedding_model:
                    qa_store.append(item)
                    try:
                        prompt_text = str(item["prompt"])
                        embedding = embedding_model.encode(prompt_text, convert_to_tensor=True)
                        embeddings.append(embedding)
                    except Exception as e:
                        logger.error(f"Error encoding QA prompt line {line_num}: {e}", exc_info=True)
                        # Remove corresponding item from qa_store if embedding failed?
                        # qa_store.pop()

        if embeddings:
            qa_embeddings = torch.stack(embeddings)
            logger.info(f"Loaded {len(qa_store)} Q&A pairs with embeddings from '{filepath}'.")
        else:
            logger.info(f"No valid Q&A pairs found or loaded with embeddings from '{filepath}'.")
            qa_embeddings = None
    except FileNotFoundError:
        logger.warning(f"Local Q&A file '{filepath}' not found. Skipping local QA store loading.")
    except Exception as e:
        logger.error(f"Failed loading local Q&A store: {e}", exc_info=True)


def cleanup_expired_ambiguity_cache():
    """Removes expired entries from the ambiguity cache."""
    # This should ideally run periodically (e.g., via a background scheduler like APScheduler)
    # or less frequently within requests if scheduler is not available.
    # For simplicity, calling it at the end of smart_ask is okay for low traffic.
    now = time.time()
    # Create a list of keys to delete to avoid modifying dict while iterating
    expired_keys = [
        key for key, data in list(AMBIGUITY_CACHE.items())
        if now - data.get('timestamp', 0) > AMBIGUITY_CACHE_EXPIRY
    ]
    if expired_keys:
        logger.info(f"Cleaning up {len(expired_keys)} expired ambiguity cache entries.")
        for key in expired_keys:
            try:
                del AMBIGUITY_CACHE[key]
            except KeyError:
                pass # Key already removed, ignore


# ==============================================================================
# Optional: Application Startup Tasks
# ==============================================================================
# These should ideally be called once when the Flask application starts,
# not every time the blueprint is imported. Use Flask's app context or startup hooks.

# Example (Place in your main app factory or run script):
# with app.app_context():
#     logger.info("Performing startup tasks...")
#     _initialize_embedding_model()
#     build_company_name_index()
#     build_screener_index()
#     load_local_qa_store() # If using local QA
#     logger.info("Startup tasks complete.")