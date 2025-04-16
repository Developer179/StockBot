import os
import re
import json
import time
import uuid
import hashlib
import logging
import functools
from typing import Callable, Any, Dict, List, Optional, Union

import torch # Still needed for screener embeddings
import requests
import psycopg2 # core driver
# Remove rapidfuzz as it's not used for company matching anymore
from flask import Blueprint, request, jsonify
from sentence_transformers import SentenceTransformer, util # Still needed for screeners

# ==============================================================================
# Logging Configuration
# ==============================================================================
LOG_LEVEL = os.getenv("UNIVEST_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(funcName)s:%(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__) # Get logger for this specific module
logger.info("Logging initialised – level=%s", LOG_LEVEL)

# ==============================================================================
# Decorators (Keep log_call as it's useful)
# ==============================================================================
def log_call(level: int = logging.DEBUG):
    """Decorator to automatically log entry, exit and runtime of a function."""
    def _decorator(func: Callable):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            child_logger = logging.getLogger(func.__module__)
            arg_preview = ", ".join([repr(a)[:80] for a in args])
            kw_preview = ", ".join([f"{k}={repr(v)[:80]}" for k, v in kwargs.items()])
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
psycopg2_extras = None
try:
    if hasattr(psycopg2, "extras"):
        psycopg2_extras = psycopg2.extras
    else:
        from psycopg2 import extras as psycopg2_extras_import # type: ignore
        psycopg2_extras = psycopg2_extras_import
except ImportError:
    logger.error("Failed to import psycopg2.extras – make sure 'psycopg2‑binary' is installed.")
    # Depending on requirements, you might want to raise an error or exit here

try:
    from ..utils.db import get_db_connection
    from ..utils.helpers import make_json_safe, compute_data_hash
except (ImportError, ValueError):
    logger.warning("Relative imports failed. Attempting direct/sys.path import for utils...")
    try:
        import sys
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
        from app.utils.db import get_db_connection
        from app.utils.helpers import make_json_safe, compute_data_hash
    except ImportError as e:
        logger.critical(f"Failed to import utility functions (db, helpers): {e}. Check PYTHONPATH or project structure.", exc_info=True)
        raise ImportError(f"Could not import essential utility functions: {e}")

# ==============================================================================
# Blueprint Setup
# ==============================================================================
session_bp = Blueprint('session', __name__)

# ==============================================================================
# Configuration & Constants
# ==============================================================================
# Keep ESSENTIAL_COMPANY_KEYS as a fallback, though less critical now
ESSENTIAL_COMPANY_KEYS: List[str] = [
    "fin_code", "comp_name", "symbol", "isin", "sector", "industry", "scrip_name",
    # ... (keep other keys if needed as fallback for LLM context)
    "nse_last_closed_price", "bse_last_closed_price", "market_capital", "pe_ratio",
]

# Keep CANONICAL_ID_TO_SOURCE_MAP and TABLE_TO_FIELDS_MAP for mapping concepts to tables/fields
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
    "pb_ratio": "consolidated_company_equity", "book_value": "consolidated_company_equity",
    "face_value": "consolidated_company_equity", "eps": "consolidated_company_equity",
    "type": "consolidated_company_equity",
    # Screeners (Keyword -> Table Name)
    "FUTURES_TOP_PRICE_GAINERS": "screeners", "NIFTY50": "screeners", "LONG_TERM_VERDICT_BUY": "screeners",
    # ... Add ALL other screener keywords ...
}

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
        "face_value", "eps", "type"
    },
    "screeners": {} # Keep screeners conceptual
}

# ==============================================================================
# Global Variables & Cache
# ==============================================================================
# Keep caches and screener-related globals
AMBIGUITY_CACHE: Dict[str, Dict[str, Any]] = {} # Cache for ambiguity resolution context
AMBIGUITY_CACHE_EXPIRY = 300  # 5 minutes

# Embedding Model (only needed for screeners now)
embedding_model: Optional[SentenceTransformer] = None
SCREENER_INDEX: List[Dict[str, Any]] = []
SCREENER_EMBEDDING_MATRIX: Optional[torch.Tensor] = None

# QA Store (Optional)
qa_store: List[Dict[str, str]] = []
qa_embeddings: Optional[torch.Tensor] = None

# ==============================================================================
# Initialization Functions (Keep for Screeners and potentially QA)
# ==============================================================================

@log_call()
def _initialize_embedding_model() -> None:
    """Lazy-load the SentenceTransformer model (if needed for screeners)."""
    global embedding_model
    if embedding_model is not None:
        return
    # Only load if we actually have screeners or QA that need it
    # This check could be more sophisticated (e.g., based on config)
    # For now, assume it might be needed if screener index isn't disabled.
    cache_folder = os.environ.get("SENTENCE_TRANSFORMERS_HOME")
    model_name = "all-MiniLM-L6-v2"
    logger.info("Attempting to load SentenceTransformer '%s' (for screeners/QA)...", model_name)
    try:
        embedding_model = SentenceTransformer(model_name, cache_folder=cache_folder)
        logger.info("SentenceTransformer loaded successfully.")
    except Exception as exc:
        logger.warning("Failed to load SentenceTransformer: %s. Screener matching might be affected.", exc)
        # Don't raise, allow app to continue without screener embeddings if necessary


# --- REMOVED build_company_name_index ---

@log_call()
def build_screener_index() -> None:
    """Populate SCREENER_INDEX & SCREENER_EMBEDDING_MATRIX from DB."""
    global SCREENER_INDEX, SCREENER_EMBEDDING_MATRIX
    # Ensure model is loaded *before* attempting to build embeddings
    _initialize_embedding_model() # Ensure model is loaded attempt happens first
    if not embedding_model:
         logger.warning("Embedding model not loaded, cannot build screener index embeddings.")
         # Optionally load screener text data even without embeddings?
         # For now, we skip embedding generation if model fails.
         # Load text data anyway:
         conn = None
         index: List[Dict[str, str]] = []
         try:
             conn = get_db_connection()
             if not conn: return
             cur = conn.cursor()
             cur.execute("SELECT keyword, title, description FROM screeners ORDER BY keyword")
             rows = cur.fetchall()
             cur.close()
             for keyword, title, desc in rows:
                 text = f"Title: {title or keyword}. Description: {desc or ''}. Keyword: {keyword}".lower()
                 index.append({"keyword": keyword, "text": text})
             SCREENER_INDEX = index
             SCREENER_EMBEDDING_MATRIX = None # Explicitly set matrix to None
             logger.info("Screener index text loaded (%d entries), but embeddings not generated (model unavailable).", len(SCREENER_INDEX))
         except psycopg2.Error as db_err:
             logger.exception("Database error building screener index text: %s", db_err)
             SCREENER_INDEX = []
             SCREENER_EMBEDDING_MATRIX = None
         finally:
             if conn: conn.close()
         return # Exit function after loading text only

    # Proceed with embedding generation if model is available
    if SCREENER_INDEX and SCREENER_EMBEDDING_MATRIX is not None:
        logger.debug("Screener index already cached (size=%d)", len(SCREENER_INDEX))
        return

    conn = None
    embeddings: List[torch.Tensor] = []
    index: List[Dict[str, str]] = []
    try:
        conn = get_db_connection()
        if not conn: return
        cur = conn.cursor()
        cur.execute("SELECT keyword, title, description FROM screeners ORDER BY keyword")
        rows = cur.fetchall()
        cur.close()
        logger.info("Fetched %d screeners for embedding index", len(rows))

        for keyword, title, desc in rows:
            text = f"Title: {title or keyword}. Description: {desc or ''}. Keyword: {keyword}".lower()
            if len(text) < 5: continue
            try:
                emb = embedding_model.encode(text, convert_to_tensor=True)
                embeddings.append(emb)
                index.append({"keyword": keyword, "text": text}) # Store keyword and text
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
        if conn: conn.close()

# ==============================================================================
# Core Logic Functions
# ==============================================================================

# --- REMOVED find_company_by_name_or_symbol ---

@log_call(logging.INFO)
def find_company_direct_db(query_name_or_symbol: str, instrument_filter: str = 'EQUITY') -> Union[None, Dict[str, Any], List[Dict[str, Any]]]:
    """
    Finds company matches directly from the database based on name or symbol.

    Args:
        query_name_or_symbol: The name or symbol to search for.
        instrument_filter: Filter by instrument type (default: 'EQUITY').

    Returns:
        - None: If no match found.
        - Dict[str, Any]: If exactly one match is found.
        - List[Dict[str, Any]]: If multiple matches are found (ambiguity).
    """
    if not query_name_or_symbol or not query_name_or_symbol.strip():
        logger.warning("find_company_direct_db called with empty query.")
        return None

    cleaned_query = query_name_or_symbol.lower().strip()
    conn = None

    # *** IMPORTANT: Ensure you have indexes on LOWER(comp_name), LOWER(symbol), and instrument_name in company_master ***
    # Example Index (PostgreSQL):
    # CREATE INDEX idx_company_master_lower_comp_name ON company_master (LOWER(comp_name));
    # CREATE INDEX idx_company_master_lower_symbol ON company_master (LOWER(symbol));
    # CREATE INDEX idx_company_master_instrument_name ON company_master (instrument_name);
    # *******************************************************************************************************************

    sql = """
        SELECT fin_code, comp_name, symbol, isin, sector
        FROM company_master
        WHERE (LOWER(comp_name) = %s OR LOWER(symbol) = %s)
          AND instrument_name = %s
        ORDER BY fin_code DESC -- Optional: Prefer higher fin_code if needed, but primarily used to detect multiple matches
    """
    params = (cleaned_query, cleaned_query, instrument_filter)
    logger.debug(f"Executing company lookup: SQL='{sql}' PARAMS={params}")

    try:
        conn = get_db_connection()
        if not conn: raise ConnectionError("Failed to get DB connection")

        cur = None
        is_dict_cursor = False
        if psycopg2_extras:
            try:
                cur = conn.cursor(cursor_factory=psycopg2_extras.DictCursor)
                is_dict_cursor = True
            except Exception as e:
                logger.warning(f"Failed DictCursor: {e}, using standard cursor.")
                cur = conn.cursor()
        else:
            cur = conn.cursor()

        cur.execute(sql, params)
        results = cur.fetchall()
        num_results = len(results)
        cur.close()

        if num_results == 0:
            logger.info(f"No company found in DB for query='{query_name_or_symbol}', instrument='{instrument_filter}'.")
            return None
        elif num_results == 1:
            row = results[0]
            company_data = dict(row) if is_dict_cursor else dict(zip([desc[0] for desc in cur.description], row))
            logger.info(f"Found 1 company match for query='{query_name_or_symbol}': {company_data.get('comp_name')} ({company_data.get('fin_code')})")
            return company_data
        else: # num_results > 1
            company_list = []
            if is_dict_cursor:
                company_list = [dict(row) for row in results]
            else:
                cols = [desc[0] for desc in cur.description]
                company_list = [dict(zip(cols, row)) for row in results]

            # Limit the list for clarification prompt if it's excessively long
            max_options = 7 # Max options to show user
            if len(company_list) > max_options:
                 logger.warning(f"Found {num_results} companies for query='{query_name_or_symbol}', limiting clarification options to {max_options}.")
                 # Optionally try to prioritize? e.g. pick top N by fin_code?
                 # Simple truncation for now:
                 company_list = company_list[:max_options]
            else:
                 logger.info(f"Found {num_results} AMBIGUOUS companies for query='{query_name_or_symbol}'. Returning list for clarification.")

            return company_list # Return list to signal ambiguity

    except psycopg2.Error as db_err:
        logger.exception(f"Database error finding company '{query_name_or_symbol}': {db_err}")
        return None # Treat DB error as not found for simplicity, or raise/handle differently
    except Exception as e:
        logger.exception(f"Unexpected error finding company '{query_name_or_symbol}': {e}")
        return None
    finally:
        if conn:
            conn.close()


def filter_dict(data_dict: Dict, keys_to_keep: List[str]) -> Dict:
    """Filters a dictionary to keep only specified keys."""
    if not isinstance(data_dict, dict): return {}
    return {k: data_dict.get(k) for k in keys_to_keep if k in data_dict}


# Keep get_specific_company_data - it fetches data *after* a fin_code is known
@log_call(logging.DEBUG)
def get_specific_company_data(fin_code: str, requested_fields: List[str]) -> Optional[Dict[str, Any]]:
    """Fetches only specific fields for a company from relevant tables."""
    conn = None
    logger.debug(f"Fetching specific fields {requested_fields} for fin_code: {fin_code}")
    if not requested_fields or not fin_code:
        logger.warning("Missing fin_code or requested_fields for specific data fetch.")
        return None

    fields_in_master = TABLE_TO_FIELDS_MAP["company_master"]
    fields_in_additional = TABLE_TO_FIELDS_MAP["company_additional_details"]
    fields_in_equity = TABLE_TO_FIELDS_MAP["consolidated_company_equity"]
    table_aliases = {'m': fields_in_master, 'ad': fields_in_additional, 'eq': fields_in_equity}

    select_parts = set()
    joins = []
    join_needed = {'ad': False, 'eq': False}
    actual_fields_to_select = set() # Store db_field name for SELECT clause

    select_parts.add("m.fin_code") # Always include fin_code from master table
    actual_fields_to_select.add("fin_code")

    # Determine required fields and joins
    for field in requested_fields:
        found = False
        for alias, field_set in table_aliases.items():
            if field in field_set:
                if field != 'fin_code':
                    db_field_name = f"{alias}.{field}"
                    select_parts.add(db_field_name)
                    actual_fields_to_select.add(field) # We need the final key name
                if alias == 'ad': join_needed['ad'] = True
                if alias == 'eq': join_needed['eq'] = True
                found = True
                break
        if not found and field != 'fin_code':
             logger.warning(f"Requested field '{field}' not mapped to a known table alias (m, ad, eq).")

    if not actual_fields_to_select:
        logger.warning(f"No valid fields to select for fin_code {fin_code} based on requested: {requested_fields}")
        return None

    if join_needed['ad']: joins.append("LEFT JOIN company_additional_details ad ON m.fin_code = ad.fin_code")
    if join_needed['eq']: joins.append("LEFT JOIN consolidated_company_equity eq ON m.fin_code = eq.fin_code")

    select_clause = ', '.join(sorted(list(select_parts)))
    sql = f"SELECT {select_clause} FROM company_master m {' '.join(joins)} WHERE m.fin_code = %s"
    logger.debug("Executing SQL:\n%s\nparams=(%s,)", sql, fin_code)

    try:
        conn = get_db_connection()
        if not conn: raise ConnectionError("Failed to get DB connection")
        cur = None
        is_dict_cursor = False
        if psycopg2_extras:
             try: cur = conn.cursor(cursor_factory=psycopg2_extras.DictCursor); is_dict_cursor = True
             except Exception as e: logger.warning(f"Failed DictCursor: {e}"); cur = conn.cursor()
        else: cur = conn.cursor()

        cur.execute(sql, (fin_code,))
        result_row = cur.fetchone() # Fetch only one row

        if not result_row:
            logger.warning(f"No specific data found for fin_code {fin_code} with SQL: {sql}")
            cur.close()
            return None

        if is_dict_cursor:
            data_dict_raw = dict(result_row)
        else:
            cols = [desc[0] for desc in cur.description]
            data_dict_raw = dict(zip(cols, result_row))

        cur.close()
        logger.info(f"Fetched raw specific data for fin_code {fin_code}")

        final_data = {}
        # Map based on actual_fields_to_select (which are the db field names without alias)
        for req_field in actual_fields_to_select: # Iterate through the fields we intended to select
            if req_field in data_dict_raw:
                 final_data[req_field] = data_dict_raw[req_field]
            else:
                 # This case should be less likely now with LEFT JOIN and specific selection
                 final_data[req_field] = None # Field was requested but not found in result
                 logger.debug(f"Requested field '{req_field}' not present in fetched DB data for {fin_code}.")

        # Ensure essential identifiers are included if they were implicitly added earlier
        for essential in ["fin_code", "comp_name", "symbol"]:
            if essential in data_dict_raw and essential not in final_data:
                 final_data[essential] = data_dict_raw[essential]

        return make_json_safe(final_data)

    except psycopg2.Error as db_err:
        logger.exception(f"DB error during specific fetch for fin_code {fin_code}: {db_err}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error during specific fetch for fin_code {fin_code}: {e}")
        return None
    finally:
        if conn:
            conn.close()

# Keep get_screener_data as is
@log_call(logging.DEBUG)
def get_screener_data(keyword: str) -> Dict[str, Any]:
    """Fetches data for a specific screener keyword."""
    # ... (implementation remains the same)
    conn = None
    logger.debug(f"Fetching screener data for keyword: {keyword}")
    default_response = {"keyword": keyword, "title": keyword, "description": "Screener not found or error.", "total_companies": 0, "companies": []}
    try:
        conn = get_db_connection()
        if not conn: raise ConnectionError("Failed DB conn")
        cur = None
        is_dict_cursor = False
        if psycopg2_extras:
             try: cur = conn.cursor(cursor_factory=psycopg2_extras.DictCursor); is_dict_cursor = True
             except Exception: cur = conn.cursor()
        else: cur = conn.cursor()

        cur.execute('SELECT keyword, title, description, fin_codes FROM screeners WHERE keyword = %s', (keyword,))
        screener_info_row = cur.fetchone()

        if not screener_info_row:
            logger.warning(f"Screener keyword '{keyword}' not found in database.")
            cur.close()
            return default_response

        if is_dict_cursor: screener_info = dict(screener_info_row)
        else: cols = [desc[0] for desc in cur.description]; screener_info = dict(zip(cols, screener_info_row))

        fin_codes_str = screener_info.get('fin_codes')
        if not fin_codes_str:
            logger.warning(f"Screener '{keyword}' found but has no associated fin_codes.")
            cur.close()
            return make_json_safe({"keyword": keyword, "title": screener_info.get('title', keyword), "description": screener_info.get('description', ''), "total_companies": 0, "companies": []})

        fin_codes = [code.strip() for code in fin_codes_str.split(',') if code.strip()]
        total_companies = len(fin_codes)
        logger.info(f"Screener '{keyword}' has {total_companies} companies.")

        companies = []
        limit = 10
        if fin_codes:
            fin_codes_to_fetch = tuple(fin_codes[:limit])
            if len(fin_codes_to_fetch) > 0:
                placeholders = '%s' if len(fin_codes_to_fetch) == 1 else ', '.join(['%s'] * len(fin_codes_to_fetch))
                sql = f'SELECT fin_code, comp_name, symbol, sector FROM company_master WHERE fin_code IN ({placeholders})'
                # Re-use cursor or create new one if needed
                cur.execute(sql, fin_codes_to_fetch)
                company_cols = [desc[0] for desc in cur.description]
                companies = [dict(zip(company_cols, row)) for row in cur.fetchall()]
                logger.debug(f"Fetched {len(companies)} example companies for screener '{keyword}'.")

        cur.close()
        return make_json_safe({"keyword": keyword, "title": screener_info.get('title', keyword), "description": screener_info.get('description', ''), "total_companies": total_companies, "companies": companies})
    except psycopg2.Error as db_err:
        logger.exception(f"Error getting screener data for keyword '{keyword}': {db_err}")
        return default_response
    except Exception as e:
        logger.exception(f"Unexpected error getting screener data for keyword '{keyword}': {e}")
        return default_response
    finally:
        if conn: conn.close()


# Keep generate_direct_answer - it works on fetched data
def generate_direct_answer(question: str, company_data: Dict[str, Any], company_name: str) -> Optional[str]:
    """Attempts to generate a direct answer from fetched DB data based on keywords."""
    # ... (implementation remains the same)
    question_lower = question.lower()
    answer = None
    logger.debug(f"Attempting direct answer generation for '{question}' with data keys: {list(company_data.keys())}")

    # Define specific patterns and the EXACT field they map to
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
        (r'\b(price|ltp|cmp)\b.*\b(on|for)\s+nse\b', 'nse_last_closed_price', "Last closing price of {name} on NSE: ₹{val:.2f}."),
        (r'\b(price|ltp|cmp)\b.*\b(on|for)\s+bse\b', 'bse_last_closed_price', "Last closing price of {name} on BSE: ₹{val:.2f}."),
        # Limits
        (r'\bnse\s+upper\s+(limit|band|circuit)\b', 'nse_upper_limit', "NSE upper circuit limit for {name}: ₹{val:.2f}."),
        (r'\bbse\s+upper\s+(limit|band|circuit)\b', 'bse_upper_limit', "BSE upper circuit limit for {name}: ₹{val:.2f}."),
        (r'\bnse\s+lower\s+(limit|band|circuit)\b', 'nse_lower_limit', "NSE lower circuit limit for {name}: ₹{val:.2f}."),
        (r'\bbse\s+lower\s+(limit|band|circuit)\b', 'bse_lower_limit', "BSE lower circuit limit for {name}: ₹{val:.2f}."),
        (r'\bcircuit\s+limit(s)?\b', ['nse_lower_limit', 'nse_upper_limit', 'bse_lower_limit', 'bse_upper_limit'], "Circuit limits..."),
        # Ratios & Financials
        (r'\bp(.)?e\s+ratio\b', 'pe_ratio', "P/E Ratio for {name}: {val:.2f}."),
        (r'\b(market cap|mcap|market capitalization)\b', 'market_capital', "Market Cap of {name}: {val}."),
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
        (r'\bverdict\b|\boutlook\b|\brecommendation\b', ['short_term_verdict', 'long_term_verdict'], "Verdicts..."),
        # Other
        (r'\blot size\b', 'lot_size', "Lot size for {name}: {val}."),
        (r'\boi\b|open interest\b', 'oi', "Open Interest for {name}: {val}."),
    ]

    matched_value = None
    formatting_string = None
    matched_pattern_keyword = None
    data_keys_available = set(k for k, v in company_data.items() if v is not None) # Only consider keys with non-None values

    for pattern, db_field_or_list, fmt_string in specific_patterns:
        if re.search(pattern, question_lower):
            matched_pattern_keyword = pattern
            logger.debug(f"Direct answer pattern matched: '{pattern}'")

            if isinstance(db_field_or_list, list):
                required_fields = set(db_field_or_list)
                # Check if ALL required fields have non-None values
                if required_fields.issubset(data_keys_available):
                    matched_value = {f: company_data.get(f) for f in db_field_or_list}
                    formatting_string = fmt_string
                    logger.debug(f"Multi-field pattern '{pattern}' matched with available data.")
                    break
                else:
                    missing_fields = required_fields - data_keys_available
                    logger.debug(f"Multi-field pattern '{pattern}' matched, but required data fields {missing_fields} have None values.")
                    answer = f"Some specific data ({', '.join(f.replace('_',' ') for f in missing_fields)}) for {company_name} is currently unavailable to answer this precisely."
                    return answer
            else: # Single field case
                db_field = db_field_or_list
                if db_field in data_keys_available: # Check if field exists AND is not None
                    matched_value = company_data[db_field]
                    formatting_string = fmt_string
                    logger.debug(f"Single-field pattern '{pattern}' matched with available data for field '{db_field}'.")
                    break
                else:
                    logger.debug(f"Pattern '{pattern}' matched, but data field '{db_field}' is None or missing.")
                    answer = f"The specific data ({db_field.replace('_', ' ')}) for {company_name} is currently unavailable."
                    return answer

    # Format the Answer
    if matched_value is not None and formatting_string is not None:
        try:
            if formatting_string == "Circuit limits...":
                 nse_low = matched_value.get('nse_lower_limit'); nse_high = matched_value.get('nse_upper_limit')
                 bse_low = matched_value.get('bse_lower_limit'); bse_high = matched_value.get('bse_upper_limit')
                 parts = []
                 if nse_low is not None and nse_high is not None: parts.append(f"NSE Limits: ₹{float(nse_low):.2f}-₹{float(nse_high):.2f}")
                 if bse_low is not None and bse_high is not None: parts.append(f"BSE Limits: ₹{float(bse_low):.2f}-₹{float(bse_high):.2f}")
                 answer = f"Circuit limits for {company_name}: {'; '.join(parts)}." if parts else f"Circuit limit information for {company_name} is currently unavailable."
            elif formatting_string == "Verdicts...":
                st = matched_value.get('short_term_verdict'); lt = matched_value.get('long_term_verdict')
                if st and lt: answer = f"{company_name}: Short-term: '{st}', Long-term: '{lt}'."
                elif st: answer = f"{company_name} Short-term: '{st}'."
                elif lt: answer = f"{company_name} Long-term: '{lt}'."
                else: answer = f"Verdicts for {company_name} are currently unavailable."
            elif "Market Cap" in formatting_string:
                 mcap = matched_value; mcap_str = f"{mcap}"
                 try:
                     mcap_f = float(mcap)
                     if mcap_f >= 1e7: mcap_str = f"₹{mcap_f/1e7:.2f} Cr"
                     elif mcap_f >= 1e5: mcap_str = f"₹{mcap_f/1e5:.2f} Lac"
                     else: mcap_str = f"₹{mcap_f:.2f}"
                 except (ValueError, TypeError): pass
                 answer = formatting_string.format(name=company_name, val=mcap_str)
            else:
                 # General formatting, attempt float conversion if format suggests
                 try:
                      if isinstance(matched_value, (int, float)) or ("{val:.2f}" in formatting_string):
                           answer = formatting_string.format(name=company_name, val=float(matched_value))
                      else: # Treat as string
                           answer = formatting_string.format(name=company_name, val=str(matched_value))
                 except (ValueError, TypeError): # Fallback if float conversion fails
                     answer = formatting_string.format(name=company_name, val=str(matched_value))

            if answer: logger.info(f"Generated DIRECT answer using pattern '{matched_pattern_keyword}'."); return answer
            else: logger.warning(f"Formatting resulted in empty answer for pattern '{matched_pattern_keyword}'.")
        except Exception as fmt_err:
            logger.error(f"Formatting error for pattern '{matched_pattern_keyword}' with value '{matched_value}': {fmt_err}", exc_info=True)

    # If no specific pattern matched or formatting failed
    logger.debug(f"No specific direct answer rule applied or formatted for: '{question}'")
    return None


# Keep get_rasa_response_payload as is
@log_call(logging.DEBUG)
def get_rasa_response_payload(question: str, rasa_url: str = "http://localhost:5005") -> Optional[Dict[str, Any]]:
    """Sends question to Rasa REST webhook, returns 'custom' JSON payload."""
    # ... (implementation remains the same)
    if not rasa_url: logger.error("Rasa URL not configured."); return None
    webhook_endpoint = f"{rasa_url.rstrip('/')}/webhooks/rest/webhook"
    sender_id = f"backend_caller_{hashlib.sha1(question.encode()).hexdigest()[:10]}"
    payload = {"sender": sender_id, "message": question}
    try:
        logger.debug(f"Sending request to Rasa: {webhook_endpoint} | Sender: {sender_id}")
        res = requests.post(webhook_endpoint, json=payload, timeout=15)
        res.raise_for_status()
        response_data = res.json()
        logger.debug(f"Raw response from Rasa: {json.dumps(response_data, indent=2)}")
        if not isinstance(response_data, list): return None
        for msg in reversed(response_data):
            if isinstance(msg, dict) and "custom" in msg and isinstance(msg["custom"], dict) and msg["custom"]:
                custom_payload = msg["custom"]
                logger.info(f"Extracted 'custom' payload from Rasa for question: '{question}'")
                if custom_payload.get("query_intent"): return custom_payload
        logger.warning(f"No message with a valid 'custom' dictionary found in Rasa response for: '{question}'")
        return None
    except requests.exceptions.Timeout: logger.error(f"Rasa timeout: {webhook_endpoint}"); return None
    except requests.exceptions.ConnectionError as e: logger.error(f"Rasa conn error: {e}"); return None
    except requests.exceptions.RequestException as e: logger.error(f"Rasa HTTP error: {e}"); return None
    except json.JSONDecodeError as e: logger.error(f"Rasa JSON decode error: {e}"); return None
    except Exception as e: logger.exception(f"Unexpected error calling Rasa: {e}"); return None


# ==============================================================================
# LLM Interaction Functions (Keep as is)
# ==============================================================================
@log_call(logging.INFO)
def call_gemini_api(prompt_text: str, *, is_json_output: bool = False) -> Union[str, Dict]:
    """Low-level wrapper for calling Google Gemini API."""
    # ... (implementation remains the same)
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key: logger.error("GEMINI_API_KEY missing."); return {"error": "API key missing"} if is_json_output else "Config Error: API key missing."
    model = os.getenv("GEMINI_MODEL", "gemini-1.5-flash-latest")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt_text}]}], "generationConfig": {"temperature": 0.6, "topP": 0.9, "topK": 40, "maxOutputTokens": 1024}}
    if is_json_output: payload["generationConfig"]["response_mime_type"] = "application/json"
    headers = {"Content-Type": "application/json"}
    timeout_seconds = 90
    logger.info("Calling Gemini API → Model=%s | JSON Output=%s | Prompt Length=%d", model, is_json_output, len(prompt_text))
    try:
        res = requests.post(url, headers=headers, json=payload, timeout=timeout_seconds)
        res.raise_for_status()
        data = res.json()
        if not data.get("candidates") and data.get("promptFeedback"):
            block_reason = data["promptFeedback"].get("blockReason")
            logger.error(f"Gemini request blocked. Reason: {block_reason}")
            return {"error": "Blocked Content", "details": block_reason} if is_json_output else f"Request blocked ({block_reason})."
        raw_text = data["candidates"][0]["content"]["parts"][0].get("text", "")
    except requests.exceptions.Timeout: logger.exception("Gemini timeout"); return {"error": "API Timeout"} if is_json_output else "Error: Timeout."
    except requests.exceptions.RequestException as exc: logger.exception("Gemini API request failed: %s", exc); return {"error": "API Request Failed", "details": str(exc)} if is_json_output else "Error: LLM communication failed."
    except (KeyError, IndexError, TypeError) as e: logger.exception("Failed parsing Gemini response: %s", e); return {"error": "Malformed API Response"} if is_json_output else "Error: Invalid LLM response."
    except Exception as e: logger.exception("Unexpected error during Gemini API call: %s", e); return {"error": "Unexpected Error"} if is_json_output else "An unexpected error occurred."

    if is_json_output:
        cleaned = raw_text.strip()
        if cleaned.startswith("```json"): cleaned = cleaned[7:-3].strip()
        elif cleaned.startswith("```"): cleaned = cleaned[3:-3].strip()
        try: return json.loads(cleaned)
        except json.JSONDecodeError as json_err: logger.error(f"Gemini JSON parse failed: {json_err}"); return {"error": "JSON Parse Error", "raw_text": raw_text}
    else: return raw_text


def get_llm_answer(prompt: str, is_json_output: bool = False, original_question: Optional[str] = None, company_data: Optional[Dict] = None, screener_data: Optional[Dict] = None) -> Union[str, Dict]:
    """Gets answer from LLM, logs interaction, handles errors."""
    # ... (implementation remains the same, logging part is useful)
    llm_response = call_gemini_api(prompt, is_json_output=is_json_output)
    if isinstance(llm_response, dict) and "error" in llm_response:
        error_type = llm_response.get("error", "Unknown Error")
        logger.error(f"LLM call failed: {error_type}. Details: {llm_response.get('details')}")
        return llm_response if is_json_output else f"Sorry, error processing: {error_type}."
    # Log successful text interactions
    if not is_json_output and isinstance(llm_response, str) and original_question:
        data_hash = None; log_data_ref = None
        if company_data: data_hash = compute_data_hash(company_data); log_data_ref = "company"
        elif screener_data: data_hash = compute_data_hash(screener_data); log_data_ref = "screener"
        log_entry = {"ts": time.time(), "q": original_question, "p_snippet": prompt[:500], "a": llm_response, "h": data_hash, "ref": log_data_ref, "int": "final_answer"}
        try:
            log_dir = "logs"; os.makedirs(log_dir, exist_ok=True)
            log_filepath = os.path.join(log_dir,"qa_llm_interactions.jsonl")
            with open(log_filepath, "a", encoding='utf-8') as f: f.write(json.dumps(log_entry) + "\n")
            logger.debug(f"LLM interaction logged")
        except Exception as log_e: logger.error(f"Failed writing LLM log: {log_e}")
    return llm_response


# Keep find_screener_by_keywords (uses embeddings)
@log_call(logging.DEBUG)
def find_screener_by_keywords(keywords: List[str]) -> Optional[str]:
    """Finds the best matching screener keyword using embeddings."""
    build_screener_index() # Ensure index text is loaded
    if not SCREENER_INDEX or not keywords: logger.warning("Screener index not loaded or no keywords."); return None
    if SCREENER_EMBEDDING_MATRIX is None or not embedding_model: logger.warning("Screener embeddings/model not available, falling back to text match."); return None # Or implement basic text match

    query = " ".join(keywords).lower().strip()
    if not query: return None
    try:
        query_emb = embedding_model.encode(query, convert_to_tensor=True)
        if query_emb.ndim == 1: query_emb = query_emb.unsqueeze(0)
        if SCREENER_EMBEDDING_MATRIX.ndim != 2 or query_emb.ndim != 2: logger.error("Screener Dim mismatch"); return None
        cosine_scores = util.pytorch_cos_sim(query_emb, SCREENER_EMBEDDING_MATRIX)[0]
    except Exception as e: logger.exception("Screener embedding/sim error"); return None

    best_idx = int(torch.argmax(cosine_scores))
    best_score = cosine_scores[best_idx].item()
    match_threshold = 0.60
    if best_score >= match_threshold:
        matched_keyword = SCREENER_INDEX[best_idx]["keyword"]
        logger.info("Screener matched – Keywords='%s' → Matched Keyword: %s (Score: %.3f)", keywords, matched_keyword, best_score)
        return matched_keyword
    else:
        logger.warning("No confident screener match for keywords '%s' (Best Score: %.3f < %.2f)", keywords, best_score, match_threshold)
        return None


# Keep get_full_company_data (can be useful for debugging or specific cases)
def get_full_company_data(fin_code: str) -> Optional[Dict[str, Any]]:
    """Fetches comprehensive data (all tables) for a company. Use Sparingly."""
    # ... (implementation remains the same)
    conn = None; logger.debug(f"Fetching FULL data for fin_code: {fin_code}"); full_data = {}
    if not fin_code: return None
    try:
        conn = get_db_connection(); dict_cur = None; is_dict_cursor = False
        if psycopg2_extras:
             try: dict_cur = conn.cursor(cursor_factory=psycopg2_extras.DictCursor); is_dict_cursor = True
             except Exception: dict_cur = conn.cursor()
        else: dict_cur = conn.cursor()
        dict_cur.execute('SELECT * FROM company_master WHERE fin_code = %s', (fin_code,))
        master_data_row = dict_cur.fetchone()
        if not master_data_row: logger.warning(f"No master data for {fin_code}"); dict_cur.close(); return None
        master_cols = list(master_data_row.keys()) if is_dict_cursor else [desc[0] for desc in dict_cur.description]
        full_data.update(dict(master_data_row) if is_dict_cursor else dict(zip(master_cols, master_data_row)))
        dict_cur.execute('SELECT * FROM company_additional_details WHERE fin_code = %s', (fin_code,))
        additional_data_row = dict_cur.fetchone()
        if additional_data_row:
            is_add_dict = isinstance(additional_data_row, dict) or (psycopg2_extras and isinstance(additional_data_row, psycopg2_extras.DictRow))
            add_cols = list(additional_data_row.keys()) if is_add_dict else [desc[0] for desc in dict_cur.description]
            add_data_dict = dict(additional_data_row) if is_add_dict else dict(zip(add_cols, additional_data_row))
            full_data.update({k: v for k, v in add_data_dict.items() if k != 'fin_code'})
        dict_cur.execute('SELECT * FROM consolidated_company_equity WHERE fin_code = %s', (fin_code,))
        equity_data_row = dict_cur.fetchone()
        if equity_data_row:
            is_eq_dict = isinstance(equity_data_row, dict) or (psycopg2_extras and isinstance(equity_data_row, psycopg2_extras.DictRow))
            eq_cols = list(equity_data_row.keys()) if is_eq_dict else [desc[0] for desc in dict_cur.description]
            eq_data_dict = dict(equity_data_row) if is_eq_dict else dict(zip(eq_cols, equity_data_row))
            full_data.update({k: v for k, v in eq_data_dict.items() if k != 'fin_code'})
        dict_cur.close(); logger.info(f"Fetched full data for {fin_code}"); return make_json_safe(full_data)
    except psycopg2.Error as db_err: logger.exception(f"DB error full fetch {fin_code}: {db_err}"); return None
    except Exception as e: logger.exception(f"Error full fetch {fin_code}: {e}"); return None
    finally:
        if conn: conn.close()

# ==============================================================================
# Prompt Templates (Keep as is)
# ==============================================================================
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
# @log_call(logging.INFO) # Enable if needed
def smart_ask():
    """ Handles user questions using Rasa for intent/table detection and direct DB lookups. """
    start_time = time.time()
    gemini_call_counter = 0
    session_id = None
    response_data = {"type": "error_initialization"}

    try:
        data = request.get_json()
        if not data: raise ValueError("Request body is empty or not valid JSON.")
        question = data.get("question", "").strip()
        session_id = data.get("session_id")
        resolve_context_id = data.get("resolve_ambiguity_context_id")
        selected_fin_code = data.get("selected_fin_code")

        if not question and not (resolve_context_id and selected_fin_code):
            logger.error("Received empty question and no ambiguity context.")
            return jsonify({"error": "Empty question received."}), 400

    except Exception as req_err:
        logger.error(f"Failed to parse request body: {req_err}", exc_info=True)
        return jsonify({"error": "Malformed request body or missing question."}), 400

    # --- Initialize variables ---
    current_company_match_info = None # Will hold the single matched company dict
    current_fin_code = None
    current_company_name = None     # Official name from DB match
    rasa_extracted_name = None      # Name extracted by Rasa (used for lookup)
    is_ambiguity_resolved = False
    original_question = question    # Store original question text
    rasa_payload = None
    rasa_intent = None
    rasa_extracted_concept = None
    rasa_target_concepts = None
    rasa_source_table = None

    # --- Ambiguity Resolution Logic ---
    if resolve_context_id and selected_fin_code:
        logger.info(f"Attempting ambiguity resolution: ContextID={resolve_context_id}, SelectedFinCode={selected_fin_code} | Session: {session_id}")
        context_data = AMBIGUITY_CACHE.get(resolve_context_id)

        if context_data and (time.time() - context_data.get('timestamp', 0) <= AMBIGUITY_CACHE_EXPIRY):
            original_question = context_data['question'] # Use original question from context
            logger.info(f"Found ambiguity context. Original Question: '{original_question}'")
            cached_options = context_data.get('options', [])

            # Find the selected company details *from the cached options*
            resolved_match = next((c for c in cached_options if str(c.get("fin_code")) == str(selected_fin_code)), None)

            if not resolved_match:
                logger.error(f"Selected fin_code {selected_fin_code} not found in cached ambiguity options for context {resolve_context_id}.")
                return jsonify({
                    "type": "error_processing_selection",
                    "message": "Sorry, I couldn't process your selection. It might have expired. Please try asking again.",
                    "source": "system_error"
                }), 200
            else:
                # Successfully resolved
                current_fin_code = selected_fin_code
                current_company_name = resolved_match.get("comp_name", "Selected Company")
                current_company_match_info = resolved_match # Store the selected company's dict
                is_ambiguity_resolved = True
                logger.info(f"Ambiguity resolved to company: {current_company_name} ({current_fin_code})")
                # Remove context from cache
                try: del AMBIGUITY_CACHE[resolve_context_id]
                except KeyError: pass
                # *** Now we have the fin_code and original question, proceed to call Rasa ***
        else:
            logger.warning(f"Ambiguity context {resolve_context_id} expired or not found. Session: {session_id}")
            return jsonify({
                "type": "error_context_lost",
                "message": "Sorry, the selection context has expired. Could you please ask your original question again?",
                "source": "system_error"
            }), 200

    # --- DB Connection Check (Early check) ---
    try:
        conn_test = get_db_connection()
        if not conn_test: raise ConnectionError("get_db_connection returned None.")
        conn_test.close()
        logger.debug("Initial DB connection check successful.")
    except Exception as db_conn_err:
        logger.critical(f"Initial DB connection check failed: {db_conn_err}", exc_info=True)
        return jsonify({"type": "database_error", "message": "System error: Cannot connect to data source.", "source": "system_error"}), 503

    # --- Rasa Interaction ---
    # Always call Rasa with the original_question (could be from input or ambiguity cache)
    # unless we are *currently* handling an ambiguity clarification request itself.
    if not response_data.get("type") == "ask_clarification": # Avoid calling Rasa if we just generated a clarification q
        rasa_url = os.getenv("RASA_WEBHOOK_URL", "http://localhost:5005")
        log_msg = f"Calling Rasa for question: '{original_question}' | Session: {session_id}"
        if is_ambiguity_resolved: log_msg += " (Post-Ambiguity Resolution)"
        logger.info(log_msg)
        rasa_payload = get_rasa_response_payload(original_question, rasa_url=rasa_url)

        if rasa_payload:
            rasa_intent = rasa_payload.get("query_intent")
            # Prioritize resolved_company, fall back to extracted_company
            company_name_from_rasa = rasa_payload.get("resolved_company") or rasa_payload.get("extracted_company")
            rasa_extracted_concept = rasa_payload.get("extracted_concept")
            rasa_target_concepts = rasa_payload.get("target_concepts")
            rasa_source_table = rasa_payload.get("source_table")
            rasa_extracted_name = company_name_from_rasa # Keep track of what Rasa gave us

            logger.info(f"Rasa Payload Parsed: Intent='{rasa_intent}', Company='{rasa_extracted_name}', Concept(s)='{rasa_target_concepts or rasa_extracted_concept}', Table='{rasa_source_table}' | Session: {session_id}")
        else:
            logger.warning(f"Rasa did not return a usable payload for: '{original_question}'")
            # Proceed to fallback logic later

    # --- Company Identification (Direct DB Lookup if not resolved via ambiguity) ---
    if not is_ambiguity_resolved and rasa_extracted_name:
        logger.info(f"Rasa identified company '{rasa_extracted_name}'. Performing direct DB lookup...")
        db_lookup_result = find_company_direct_db(rasa_extracted_name, instrument_filter='EQUITY')

        if isinstance(db_lookup_result, dict): # Single match
            current_company_match_info = db_lookup_result
            current_fin_code = current_company_match_info.get("fin_code")
            current_company_name = current_company_match_info.get("comp_name") # Use official name
            logger.info(f"Direct DB lookup found unique match: {current_company_name} ({current_fin_code})")
        elif isinstance(db_lookup_result, list): # Multiple matches (Ambiguity)
            logger.warning(f"Direct DB lookup found {len(db_lookup_result)} ambiguous matches for '{rasa_extracted_name}'.")
            context_id = str(uuid.uuid4())
            # Prepare options for clarification (only essential info)
            options_for_user = [
                {"fin_code": c.get("fin_code"), "name": c.get("comp_name"), "symbol": c.get("symbol"), "sector": c.get("sector")}
                for c in db_lookup_result
            ]
            # Store the original question and the DB options in the cache
            AMBIGUITY_CACHE[context_id] = {
                "question": original_question,
                "options": db_lookup_result, # Store full dicts from DB briefly
                "timestamp": time.time()
            }
            logger.info(f"Storing ambiguity context {context_id} with {len(options_for_user)} options.")
            # Return clarification request
            response_data = {
                "type": "ask_clarification",
                "message": f"I found multiple potential matches for '{rasa_extracted_name}'. Please select the correct one:",
                "options": options_for_user,
                "resolve_ambiguity_context_id": context_id,
                "source": "system_clarification"
            }
            # Immediately return the clarification request
            cleanup_expired_ambiguity_cache()
            processing_time = round((time.time() - start_time) * 1000)
            response_data["processing_time_ms"] = processing_time
            logger.info(f"Request resulted in AMBIGUITY CLARIFICATION in {processing_time} ms.")
            return jsonify(response_data), 200
        else: # No match found
            logger.warning(f"Direct DB lookup found NO match for company '{rasa_extracted_name}'.")
            # fin_code remains None, proceed, likely handled by checks below

    # --- Main Logic based on Rasa Intent and Identified Company (if any) ---
    if rasa_intent:
        # --- Handle Investment Advice Intent ---
        if rasa_intent == "ask_investment_advice":
            response_data["type"] = "investment_advice"
            logger.debug(f"Processing 'ask_investment_advice' for '{current_company_name or rasa_extracted_name or 'Unknown'}'")
            if not current_fin_code:
                logger.warning("Cannot process advice: Company fin_code is missing.")
                response_data.update({"type": "error_missing_info", "message": f"Could not identify the specific company ('{rasa_extracted_name}') for the advice request." if rasa_extracted_name else "Please specify which company you'd like advice for."})
            elif not rasa_target_concepts or not rasa_source_table:
                 logger.warning(f"Cannot process advice: Missing concepts/table ({rasa_target_concepts}/{rasa_source_table}).")
                 response_data.update({"type": "error_processing_advice", "message": "Could not fully understand the advice request details."})
            else:
                logger.info(f"Fetching advice data for {current_company_name} ({current_fin_code}). Needs: {rasa_target_concepts} from {rasa_source_table}")
                advice_data = get_specific_company_data(current_fin_code, rasa_target_concepts)
                if advice_data:
                    verdict_prompt = f"Analyze the following investment verdicts for {current_company_name} and provide a brief summary interpretation. Focus ONLY on the provided verdicts. State if data is missing.\nShort-term Verdict: {advice_data.get('short_term_verdict', 'Not Available')}\nLong-term Verdict: {advice_data.get('long_term_verdict', 'Not Available')}\nSummary:"
                    gemini_call_counter += 1; logger.info(f"[Gemini Call #{gemini_call_counter}] for advice: {current_company_name}")
                    gemini_advice_answer = get_llm_answer(verdict_prompt, original_question=original_question, company_data=advice_data)
                    response_data.update({"company": current_company_match_info or {"name": current_company_name, "fin_code": current_fin_code}, "answer": gemini_advice_answer, "source": "llm_with_db_verdicts"})
                else:
                    logger.error(f"Failed to fetch advice data ({rasa_target_concepts}) for {current_company_name} ({current_fin_code})")
                    response_data.update({"type": "data_fetch_failed", "message": f"Sorry, couldn't retrieve investment outlook for '{current_company_name}'."})

        # --- Handle Standard Data Source Intent ---
        elif rasa_intent == "ask_data_source":
            response_data["type"] = "company_query"
            logger.debug(f"Processing 'ask_data_source' for concept '{rasa_extracted_concept}', table '{rasa_source_table}'")

            if not rasa_extracted_concept or not rasa_source_table:
                logger.warning(f"Cannot process data source: Missing concept/table ({rasa_extracted_concept}/{rasa_source_table}).")
                response_data.update({"type": "error_processing_data_query", "message": "Could not understand which data point you're asking about."})
            elif rasa_source_table == 'screeners':
                 logger.info(f"Handling screener request for keyword: '{rasa_extracted_concept}'")
                 screener_data = get_screener_data(rasa_extracted_concept)
                 if screener_data and screener_data.get("total_companies", 0) > 0:
                     companies_list = screener_data.get("companies", [])
                     companies_list_str = "\n".join([f"- {c.get('comp_name', 'N/A')} ({c.get('symbol', 'N/A')}) - {c.get('sector', 'N/A')}" for c in companies_list]) if companies_list else "No examples available."
                     prompt = SCREENER_PROMPT_TEMPLATE.format(screener_title=screener_data.get('title', rasa_extracted_concept), keyword=rasa_extracted_concept, description=screener_data.get('description', 'N/A'), total_companies=screener_data.get('total_companies', 0), companies_list_str=companies_list_str, question=original_question)
                     gemini_call_counter += 1; logger.info(f"[Gemini Call #{gemini_call_counter}] for screener: {rasa_extracted_concept}")
                     answer = get_llm_answer(prompt, original_question=original_question, screener_data=screener_data)
                     response_data.update({"type": "screener_result", "screener_keyword": rasa_extracted_concept, "screener_title": screener_data.get('title', rasa_extracted_concept), "answer": answer, "source": "llm_with_screener_data"})
                 elif screener_data:
                     logger.warning(f"Screener '{rasa_extracted_concept}' found but has no companies.")
                     response_data.update({"type": "screener_result_empty", "answer": f"Screener '{screener_data.get('title', rasa_extracted_concept)}' currently has no companies.", "source": "database"})
                 else:
                     logger.error(f"Screener keyword '{rasa_extracted_concept}' not found.")
                     response_data.update({"type": "screener_not_found", "answer": f"Sorry, couldn't find screener '{rasa_extracted_concept}'.", "source": "database"})
            elif not current_fin_code: # Must have fin_code for non-screener data source queries
                logger.warning(f"Cannot process data source: Company fin_code is missing for concept '{rasa_extracted_concept}'.")
                response_data.update({"type": "error_missing_info", "message": f"Please specify which company you're asking about for '{rasa_extracted_concept.replace('_',' ')}'. I couldn't identify '{rasa_extracted_name}'." if rasa_extracted_name else f"Please specify the company for '{rasa_extracted_concept.replace('_',' ')}'."})
            else: # Handle Company-Specific Data Point Queries
                logger.info(f"Fetching specific data for {current_company_name} ({current_fin_code}). Concept: {rasa_extracted_concept}, Table: {rasa_source_table}")
                fields_to_get_set = TABLE_TO_FIELDS_MAP.get(rasa_source_table, set())
                if not fields_to_get_set:
                    logger.warning(f"No fields mapped for table '{rasa_source_table}'. Using concept only.")
                    fields_to_get_set = {rasa_extracted_concept} # Minimal fetch
                else:
                    fields_to_get_set = fields_to_get_set.copy()
                    fields_to_get_set.add(rasa_extracted_concept)
                fields_to_get_set.update(["fin_code", "comp_name", "symbol"]) # Ensure essentials
                fields_to_get = list(fields_to_get_set)

                specific_data = get_specific_company_data(current_fin_code, fields_to_get)

                if specific_data:
                    db_company_name = specific_data.get("comp_name", current_company_name) # Prefer DB name
                    logger.debug(f"[Data Fetched from {rasa_source_table}]:\n{json.dumps(specific_data, indent=2, default=str)}")
                    direct_answer = generate_direct_answer(original_question, specific_data, db_company_name)
                    if direct_answer:
                        logger.info(f"✅ DIRECT answer generated for {db_company_name}")
                        response_data.update({"company": current_company_match_info or {"name": db_company_name, "fin_code": current_fin_code}, "answer": direct_answer, "source": "database"})
                    else:
                        logger.info(f"No direct answer. Calling Gemini using data from '{rasa_source_table}'.")
                        essential_data_for_llm = specific_data
                        gemini_call_counter += 1; logger.info(f"[Gemini Call #{gemini_call_counter}] for {db_company_name}|Concept:{rasa_extracted_concept}")
                        prompt = COMPANY_QUERY_PROMPT_TEMPLATE.format(company_name=db_company_name, essential_company_data=json.dumps(essential_data_for_llm, indent=2, default=str), question=original_question)
                        answer = get_llm_answer(prompt, original_question=original_question, company_data=essential_data_for_llm)
                        response_data.update({"company": current_company_match_info or {"name": db_company_name, "fin_code": current_fin_code}, "answer": answer, "source": "llm_with_db_data"})
                else:
                    logger.error(f"Failed to fetch specific data ({fields_to_get}) for {current_company_name} ({current_fin_code}) based on table '{rasa_source_table}'")
                    response_data.update({"type": "data_fetch_failed", "message": f"Sorry, couldn't retrieve '{rasa_extracted_concept.replace('_',' ')}' for '{current_company_name}'."})

        # --- Handle Other Intents ---
        else:
            logger.info(f"Handling non-data/advice Rasa intent: '{rasa_intent}'")
            response_data["type"] = f"general_fallback_rasa_{rasa_intent}"
            prompt = GENERAL_FINANCE_PROMPT_TEMPLATE.format(question=original_question)
            gemini_call_counter += 1; logger.info(f"[Gemini Call #{gemini_call_counter}] General Fallback (Rasa Intent: {rasa_intent})")
            answer = get_llm_answer(prompt, original_question=original_question)
            response_data.update({"answer": answer, "source": "llm_only"})

    # --- Fallback if Rasa Interaction Failed or No Intent Identified ---
    else:
         # If ambiguity was resolved, but Rasa failed, we might still have fin_code
         # Prioritize using fin_code if available for a generic fetch
         if current_fin_code and current_company_name:
             logger.warning(f"Rasa failed, but fin_code {current_fin_code} ({current_company_name}) is known. Fetching essential data for generic LLM call.")
             response_data["type"] = "general_fallback_with_company"
             # Fetch a small set of essential data
             essential_data = get_specific_company_data(current_fin_code, ESSENTIAL_COMPANY_KEYS)
             if essential_data:
                 prompt = COMPANY_QUERY_PROMPT_TEMPLATE.format(
                     company_name=essential_data.get("comp_name", current_company_name),
                     essential_company_data=json.dumps(essential_data, indent=2, default=str),
                     question=original_question # Ask the original question against this general data
                 )
                 gemini_call_counter += 1; logger.info(f"[Gemini Call #{gemini_call_counter}] Fallback (Rasa Failed, Known Company)")
                 answer = get_llm_answer(prompt, original_question=original_question, company_data=essential_data)
                 response_data.update({"company": current_company_match_info or {"name": current_company_name, "fin_code": current_fin_code}, "answer": answer, "source": "llm_with_db_data_fallback"})
             else: # Failed fetch even essentials
                 logger.error(f"Fallback failed: Could not fetch essential data for known fin_code {current_fin_code}")
                 response_data.update({"type": "data_fetch_failed", "answer": f"Sorry, I couldn't retrieve essential info for {current_company_name} after an internal error.", "source": "system_error"})
         else: # No fin_code, total fallback
            logger.warning(f"Rasa failed/no intent. Falling back to general LLM. | Session: {session_id} | Q: '{original_question}'")
            response_data["type"] = "general_fallback_no_rasa"
            prompt = GENERAL_FINANCE_PROMPT_TEMPLATE.format(question=original_question)
            gemini_call_counter += 1; logger.info(f"[Gemini Call #{gemini_call_counter}] General Fallback (No Rasa, No Company)")
            answer = get_llm_answer(prompt, original_question=original_question)
            response_data.update({"answer": answer, "source": "llm_only"})


    # --- Final Response Assembly ---
    end_time = time.time()
    processing_time = round((end_time - start_time) * 1000)
    response_data["processing_time_ms"] = processing_time

    if "answer" not in response_data or not response_data.get("answer"):
        error_message = response_data.get("message", "Sorry, I encountered an issue processing your request. Please try rephrasing.") # Use specific message if set
        logger.error(f"Processing finished but no answer generated for: '{original_question}' | Final Response Data: {response_data}")
        response_data["answer"] = error_message
        response_data["type"] = response_data.get("type", "error_unknown")
        response_data["source"] = response_data.get("source", "system_error")

    logger.info(f"Request processed in {processing_time} ms | Final Type: {response_data.get('type')} | Source: {response_data.get('source')} | Gemini Calls: {gemini_call_counter} | Session: {session_id}")

    cleanup_expired_ambiguity_cache()
    return jsonify(response_data), 200


# ==============================================================================
# Other Helper Routes (Keep llm-status, adjust reload-indices)
# ==============================================================================
@session_bp.route('/llm-status', methods=['GET'])
def llm_status():
    """Checks connectivity and basic response from the configured LLM."""
    # ... (implementation remains the same)
    logger.info("Checking LLM status...")
    try:
        prompt = "Respond with 'Operational' if you are functioning correctly."
        response = call_gemini_api(prompt)
        if isinstance(response, str) and "operational" in response.lower():
            logger.info("LLM status check: OK"); return jsonify({"status": "ok", "response": response}), 200
        elif isinstance(response, dict) and "error" in response:
            logger.error(f"LLM status check failed: {response}"); return jsonify({"status": "error", "details": response}), 503
        else:
            logger.warning(f"LLM status check unexpected: {str(response)[:200]}"); return jsonify({"status": "error", "message": "Unexpected LLM response."}), 500
    except Exception as e: logger.exception("Exception during LLM status check."); return jsonify({"status": "error", "details": str(e)}), 500


@session_bp.route('/reload-indices', methods=['POST'])
def reload_indices():
    """Manually triggers reloading of screener embedding indices."""
    # Add auth if needed
    logger.info("Manual index reload triggered via API (Screener only).")
    results = {}
    status_code = 200

    # --- Company index reload is removed ---
    results["company_index"] = "Company index is no longer loaded into memory."

    try:
        build_screener_index() # Reloads screener text and potentially embeddings
        results["screener_index"] = f"Reloaded {len(SCREENER_INDEX)} screeners."
        if SCREENER_EMBEDDING_MATRIX is not None:
             results["screener_index"] += f" ({SCREENER_EMBEDDING_MATRIX.shape[0]} embeddings)."
        else:
             results["screener_index"] += " (Embeddings might be unavailable)."
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
# Helper Functions (Keep cleanup, load_local_qa_store if used)
# ==============================================================================

# Keep load_local_qa_store if you plan to use it
def load_local_qa_store():
    """Loads Q&A pairs from a local file (Optional)."""
    # ... (implementation remains the same)
    global qa_store, qa_embeddings
    _initialize_embedding_model() # Ensure model loaded first if QA needs embeddings
    qa_store = []
    embeddings = []
    filepath = os.path.join(os.path.dirname(__file__), "..", "..", "logs", "qa_training_data.jsonl") # Adjust path
    logger.info(f"Attempting to load local Q&A store from: {filepath}")
    try:
        with open(filepath, "r", encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try: item = json.loads(line)
                except Exception: logger.warning(f"Skipping invalid JSON line {line_num} in QA store"); continue
                if item.get("prompt") and item.get("answer") and embedding_model:
                    qa_store.append(item)
                    try: embeddings.append(embedding_model.encode(str(item["prompt"]), convert_to_tensor=True))
                    except Exception as e: logger.error(f"Error encoding QA line {line_num}: {e}"); qa_store.pop() # Remove if embedding fails?
        if embeddings: qa_embeddings = torch.stack(embeddings); logger.info(f"Loaded {len(qa_store)} Q&A pairs with embeddings.")
        else: logger.info(f"No valid Q&A pairs found/loaded with embeddings."); qa_embeddings = None
    except FileNotFoundError: logger.warning(f"Local Q&A file '{filepath}' not found.")
    except Exception as e: logger.error(f"Failed loading local Q&A store: {e}", exc_info=True)


def cleanup_expired_ambiguity_cache():
    """Removes expired entries from the ambiguity cache."""
    # ... (implementation remains the same)
    now = time.time()
    expired_keys = [ key for key, data in list(AMBIGUITY_CACHE.items()) if now - data.get('timestamp', 0) > AMBIGUITY_CACHE_EXPIRY ]
    if expired_keys:
        logger.info(f"Cleaning up {len(expired_keys)} expired ambiguity cache entries.")
        for key in expired_keys:
            try: del AMBIGUITY_CACHE[key]
            except KeyError: pass

# ==============================================================================
# Optional: Application Startup Tasks
# ==============================================================================
# Adjust startup tasks - no company index build needed
# with app.app_context():
#     logger.info("Performing startup tasks...")
#     _initialize_embedding_model() # Load model if screeners/QA need it
#     build_screener_index()      # Build only screener index
#     load_local_qa_store()       # If using local QA
#     logger.info("Startup tasks complete.")