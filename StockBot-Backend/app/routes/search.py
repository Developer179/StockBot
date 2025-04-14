# # app/routes/search.py
# """
# Enhanced search-related routes for the Flask application.
# """
# from flask import Blueprint, request, jsonify, current_app
# import logging
# import re
# from app.utils.db import get_db_connection
# from app.utils.helpers import fuzzy_filter_rows, make_json_safe

# # Create blueprint
# search_bp = Blueprint('search', __name__)
# logger = logging.getLogger(__name__)

# @search_bp.route('/search', methods=['GET', 'OPTIONS'])
# def search():
#     """
#     Search endpoint that finds companies matching a query.
#     Handles CORS preflight requests properly.
#     """
#     # Handle preflight OPTIONS request explicitly
#     if request.method == 'OPTIONS':
#         response = current_app.make_default_options_response()
#         return response

#     query = request.args.get('q', '').lower().strip()
#     if not query:
#         return jsonify({'error': 'Missing query param'}), 400

#     conn = None
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor()

#         # First try direct search on company_master table
#         search_terms = sanitize_search_query(query)
#         search_pattern = f"%{search_terms}%"

#         # Search in multiple columns for the term
#         cur.execute('''
#             SELECT fin_code, comp_name, industry, sector FROM "company_master"
#             WHERE
#                 LOWER(comp_name) LIKE LOWER(%s) OR
#                 LOWER(scrip_name) LIKE LOWER(%s) OR
#                 LOWER(symbol) LIKE LOWER(%s) OR
#                 LOWER(bse_symbol) LIKE LOWER(%s)
#             ORDER BY
#                 CASE
#                     WHEN LOWER(comp_name) = LOWER(%s) THEN 1
#                     WHEN LOWER(comp_name) LIKE LOWER(%s) THEN 2
#                     ELSE 3
#                 END
#             LIMIT 10;
#         ''', (search_pattern, search_pattern, search_pattern, search_pattern, search_terms, f"{search_terms}%"))

#         direct_matches = cur.fetchall()

#         if direct_matches:
#             # Direct matches found
#             companies = [{
#                 "company_name": row[1],
#                 "fin_code": row[0],
#                 "industry": row[2],
#                 "sector": row[3],
#                 "match_quality": "direct",
#                 "options_available": [
#                     "📊 Stock Details",
#                     "📈 Market Performance",
#                     "🧠 Fundamentals"
#                 ]
#             } for row in direct_matches]
#         else:
#             # If no direct matches, try fuzzy matching
#             cur.execute('SELECT fin_code, comp_name, industry, sector FROM "company_master";')
#             rows = cur.fetchall()
#             fuzzy_matches = fuzzy_match_companies(rows, query)

#             if not fuzzy_matches:
#                 return jsonify({'message': f'No company found matching "{query}".'}), 404

#             companies = [{
#                 "company_name": match[1],
#                 "fin_code": match[0],
#                 "industry": match[2],
#                 "sector": match[3],
#                 "match_quality": "fuzzy",
#                 "options_available": [
#                     "📊 Stock Details",
#                     "📈 Market Performance",
#                     "🧠 Fundamentals"
#                 ]
#             } for match in fuzzy_matches[:10]]  # Limit to 10 results

#         # Create response with explicit CORS headers
#         response = jsonify({"companies": companies})
#         response.headers.add('Access-Control-Allow-Origin', '*')
#         response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
#         response.headers.add('Access-Control-Allow-Methods', 'GET,OPTIONS')
#         return response

#     except Exception as e:
#         logger.exception(f"Database error in search endpoint: {e}")
#         return jsonify({"error": "Database error", "details": str(e)}), 500
#     finally:
#         if conn:
#             conn.close()

# def sanitize_search_query(query):
#     """Remove special characters and normalize search query"""
#     # Remove any non-alphanumeric characters except spaces
#     query = re.sub(r'[^\w\s]', '', query)
#     # Replace multiple spaces with single space
#     query = re.sub(r'\s+', ' ', query)
#     return query.strip()

# def fuzzy_match_companies(companies, query):
#     """Perform fuzzy matching on company names"""
#     query = query.lower()
#     query_words = query.split()

#     scored_matches = []

#     for company in companies:
#         fin_code, name, industry, sector = company
#         if not name:
#             continue

#         name_lower = name.lower()

#         # Calculate match score
#         score = 0

#         # Exact match gets highest score
#         if name_lower == query:
#             score = 100
#         # Starts with query
#         elif name_lower.startswith(query):
#             score = 90
#         # Contains query as a substring
#         elif query in name_lower:
#             score = 80
#         else:
#             # Check word by word
#             company_words = name_lower.split()
#             for q_word in query_words:
#                 if q_word in company_words:
#                     score += 10
#                 for c_word in company_words:
#                     if c_word.startswith(q_word) or q_word.startswith(c_word):
#                         score += 5

#         if score > 0:
#             scored_matches.append((score, (fin_code, name, industry, sector)))

#     # Sort by score descending
#     scored_matches.sort(reverse=True, key=lambda x: x[0])

#     # Return just the companies, not the scores
#     return [match[1] for match in scored_matches]

# """
# Enhanced company extraction endpoint for the search.py file
# """

# @search_bp.route('/extract-company', methods=['POST'])
# def extract_company():
#     """
#     Try to extract a company name from a user question
#     and search for matching companies from the DB.
#     """
#     data = request.json
#     question = data.get('question')

#     if not question:
#         return jsonify({'error': 'Missing question parameter'}), 400

#     # Extract possible company mentions
#     potential_names = extract_company_names(question)
#     if not potential_names:
#         return jsonify({
#             'status': 'no_company_found',
#             'message': 'Could not identify a company name in your question'
#         })

#     conn = get_db_connection()
#     found_companies = []

#     try:
#         cur = conn.cursor()
#         for name in potential_names:
#             search_pattern = f"%{name}%"
#             logger.info(f"Searching for company with pattern: {search_pattern}")

#             cur.execute('''
#                 SELECT fin_code, comp_name, industry, sector FROM "company_master"
#                 WHERE
#                     LOWER(comp_name) LIKE LOWER(%s) OR
#                     LOWER(scrip_name) LIKE LOWER(%s) OR
#                     LOWER(symbol) LIKE LOWER(%s)
#                 ORDER BY
#                     CASE
#                         WHEN LOWER(comp_name) = LOWER(%s) THEN 1
#                         WHEN LOWER(comp_name) LIKE LOWER(%s) THEN 2
#                         ELSE 3
#                     END
#                 LIMIT 10;
#             ''', (search_pattern, search_pattern, search_pattern, name.lower(), f"{name.lower()}%"))

#             results = cur.fetchall()
#             for row in results:
#                 found_companies.append({
#                     'fin_code': row[0],
#                     'company_name': row[1],
#                     'industry': row[2],
#                     'sector': row[3],
#                     'extracted_name': name,
#                     'match_quality': 'db_match'
#                 })

#     except Exception as e:
#         logger.exception(f"Error in extract_company: {e}")
#         return jsonify({"error": "Database error", "details": str(e)}), 500
#     finally:
#         conn.close()

#     if found_companies:
#         return jsonify({
#             'status': 'success',
#             'companies': found_companies,
#             'extracted_names': potential_names
#         })
#     else:
#         return jsonify({
#             'status': 'no_match_found',
#             'message': f'Identified terms: {potential_names}, but found no DB matches.',
#             'extracted_names': potential_names
#         })


# def extract_company_names(question):
#     """
#     Extract potential company names from a question
#     Enhanced to better handle partial names like "Tata"
#     """
#     # Common phrases used when asking about companies
#     company_indicators = [
#         'about', 'for', 'on', 'regarding', 'of',
#         'tell me about', 'what about', 'how is',
#         'information on', 'analysis of', 'details of',
#         'financials of', 'stock of', 'share price of'
#     ]

#     # Common partial company names that should be recognized
#     common_companies = [
#         'tata', 'reliance', 'infosys', 'hdfc', 'icici',
#         'bharti', 'airtel', 'mahindra', 'wipro', 'adani'
#     ]

#     potential_companies = []
#     question_lower = question.lower()

#     # First check for common company names
#     for company in common_companies:
#         if company in question_lower:
#             potential_companies.append(company)

#     # If we found common companies, prioritize those
#     if potential_companies:
#         return potential_companies

#     # Try indicator phrases
#     for indicator in company_indicators:
#         if indicator + ' ' in question_lower:
#             parts = question_lower.split(indicator + ' ')
#             if len(parts) > 1:
#                 # Take the next part after the indicator
#                 after_indicator = parts[1].strip()
#                 # Extract up to the next punctuation or common stop word
#                 company_name = re.split(r'[,\.?!;:]|\band\b|\bor\b|\bis\b|\bhas\b|\bwill\b|\bshould\b', after_indicator)[0].strip()

#                 if len(company_name) > 2:  # Avoid too short extractions
#                     potential_companies.append(company_name)

#     # Look for capitalized words that might be company names
#     if not potential_companies:
#         words = question.split()
#         for i, word in enumerate(words):
#             # Check for capitalized words that might be company names
#             if len(word) > 0 and word[0].isupper() and len(word) > 1:
#                 # Look for multi-word company names
#                 if i < len(words) - 1 and words[i+1][0].isupper():
#                     potential_companies.append(f"{word} {words[i+1]}")
#                 potential_companies.append(word)

#     # Return unique extracted names
#     return list(set(potential_companies))


# app/routes/search.py
"""
Enhanced search-related routes for the Flask application.
"""
from flask import Blueprint, request, jsonify, current_app
import logging
import re

import psycopg2
from app.utils.db import get_db_connection
# Removed fuzzy_filter_rows from this import as it's not used here
from app.utils.helpers import make_json_safe

# Create blueprint
search_bp = Blueprint('search', __name__, url_prefix='/api/search') # Added prefix for consistency
logger = logging.getLogger(__name__)

@search_bp.route('/companies', methods=['GET']) # Changed route to be more specific
def search_companies_route(): # Renamed function for clarity
    """
    Search endpoint that finds companies matching a query.
    """
    query = request.args.get('q', '').lower().strip()
    if not query:
        # Return empty list if no query, standard for search/autocomplete
        return jsonify({"companies": []})

    conn = None
    try:
        conn = get_db_connection()
        # Use dictionary cursor for easier access
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor) # Assuming psycopg2

        # Sanitize query (optional, DB might handle some cases)
        search_terms = sanitize_search_query(query)
        search_pattern = f"%{search_terms}%"
        logger.debug(f"Executing direct search for pattern: {search_pattern}")

        # Search in multiple columns for the term
        # Using ILIKE for case-insensitive search in PostgreSQL
        # Adjusted ORDER BY for relevance
        cur.execute('''
            SELECT fin_code, comp_name, symbol, industry, sector
            FROM company_master
            WHERE
                comp_name ILIKE %s OR
                scrip_name ILIKE %s OR
                symbol ILIKE %s OR
                bse_symbol ILIKE %s
            ORDER BY
                CASE
                    WHEN comp_name ILIKE %s THEN 1       -- Exact match (case-insensitive)
                    WHEN symbol ILIKE %s THEN 2         -- Symbol match
                    WHEN comp_name ILIKE %s THEN 3       -- Starts with query
                    ELSE 4                              -- Contains query
                END,
                comp_name -- Alphabetical secondary sort
            LIMIT 10;
        ''', (search_pattern, search_pattern, search_pattern, search_pattern,
              search_terms, search_terms, f"{search_terms}%")) # Arguments for ORDER BY

        direct_matches = cur.fetchall()
        logger.debug(f"Found {len(direct_matches)} direct matches.")

        # Convert DictRow to plain dict
        companies = [dict(row) for row in direct_matches]

        # Optional: Add fuzzy matching *if* direct results are insufficient
        # (The local fuzzy_match_companies function can be used here if needed,
        # but often direct DB ILIKE is sufficient for basic search)
        # if not companies:
        #     logger.debug("No direct matches, attempting fuzzy search (using local implementation)...")
        #     cur.execute('SELECT fin_code, comp_name, symbol, industry, sector FROM company_master;')
        #     all_companies_data = cur.fetchall() # Fetch all data as list of tuples/dicts
        #     # Note: fuzzy_match_companies expects tuples based on its original structure
        #     # Convert fetched dicts back to tuples if needed, or adapt the function
        #     all_companies_tuples = [(c['fin_code'], c['comp_name'], c['symbol'], c['industry'], c['sector']) for c in all_companies_data]
        #     fuzzy_matched_tuples = fuzzy_match_companies(all_companies_tuples, query)
        #     companies = [{
        #          'fin_code': fm[0], 'comp_name': fm[1], 'symbol': fm[2],
        #          'industry': fm[3], 'sector': fm[4]
        #      } for fm in fuzzy_matched_tuples[:10]] # Limit results

        cur.close()

        # Make response JSON safe (handles Decimals, Dates etc.)
        safe_companies = make_json_safe(companies)

        response = jsonify({"companies": safe_companies})
        # CORS headers are usually handled globally by Flask-CORS extension
        # response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    except Exception as e:
        logger.exception(f"Database error in search endpoint: {e}")
        # Use make_json_safe for error details too, just in case
        error_details = make_json_safe(str(e))
        return jsonify({"error": "Database error", "details": error_details}), 500
    finally:
        if conn:
            conn.close()

# --- Helper Functions (kept as defined in the original file) ---

def sanitize_search_query(query):
    """Remove special characters and normalize search query"""
    query = re.sub(r'[^\w\s]', '', query)
    query = re.sub(r'\s+', ' ', query)
    return query.strip()

def fuzzy_match_companies(companies_tuples, query):
    """
    Perform fuzzy matching on company names.
    Expects companies as a list of tuples: (fin_code, name, symbol, industry, sector)
    """
    query = query.lower()
    query_words = set(query.split()) # Use set for faster lookup

    scored_matches = []

    for company_tuple in companies_tuples:
        # Ensure tuple has enough elements before unpacking
        if len(company_tuple) < 5:
             logger.warning(f"Skipping malformed company tuple: {company_tuple}")
             continue
        fin_code, name, symbol, industry, sector = company_tuple # Unpack tuple
        if not name: continue

        name_lower = name.lower()
        symbol_lower = symbol.lower() if symbol else ""
        name_words = set(name_lower.split())

        # --- Scoring Logic ---
        score = 0
        if name_lower == query or symbol_lower == query:
            score = 100 # Exact Name/Symbol Match
        elif name_lower.startswith(query):
            score = 90  # Name Starts With
        elif symbol_lower.startswith(query):
             score = 85 # Symbol Starts With
        elif query in name_lower:
            score = 75  # Name Contains Query
        elif query in symbol_lower:
             score = 70 # Symbol Contains Query
        else:
            # Word overlap scoring
            common_words = query_words.intersection(name_words)
            score += len(common_words) * 20 # Higher score for full word match

            # Partial word match (starts with) - less score
            for q_word in query_words:
                for n_word in name_words:
                     if n_word.startswith(q_word) and q_word != n_word:
                          score += 5
                     elif q_word.startswith(n_word) and q_word != n_word:
                          score += 3


        # Normalize score slightly (e.g., cap at 100)
        score = min(score, 100)

        if score > 40: # Keep only reasonably good matches (adjust threshold)
            scored_matches.append((score, company_tuple)) # Store score and original tuple

    # Sort by score descending
    scored_matches.sort(reverse=True, key=lambda x: x[0])

    # Return just the company tuples
    return [match[1] for match in scored_matches]


# --- Extract Company Route (kept mostly as is, but consider if needed) ---
# This route seems less useful now with /smart-ask handling extraction.
# Consider removing it unless specifically needed for a different UI feature.

@search_bp.route('/extract-company', methods=['POST'])
def extract_company_route(): # Renamed function
    """
    Try to extract a company name from a user question using simple heuristics.
    NOTE: This is less sophisticated than the LLM approach in /smart-ask.
    """
    data = request.json
    question = data.get('question')

    if not question:
        return jsonify({'error': 'Missing question parameter'}), 400

    potential_names = extract_company_names(question)
    if not potential_names:
        return jsonify({
            'status': 'no_company_found',
            'message': 'Could not identify a company name in your question using basic extraction.'
        }), 404 # Not found is more appropriate than success

    conn = None
    found_companies = []

    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor) # Use dict cursor

        # Limit the number of names to search to avoid excessive queries
        names_to_search = list(set(potential_names))[:5] # Search for top 5 unique names
        logger.info(f"Attempting basic extraction search for names: {names_to_search}")

        all_results = []
        for name in names_to_search:
             # Prioritize exact or starting matches if possible
             search_pattern_like = f"%{name}%"
             search_pattern_starts = f"{name}%"
             cur.execute('''
                 SELECT fin_code, comp_name, symbol, industry, sector FROM company_master
                 WHERE
                     comp_name ILIKE %s OR symbol ILIKE %s
                 ORDER BY
                     CASE
                         WHEN comp_name ILIKE %s THEN 1
                         WHEN symbol ILIKE %s THEN 2
                         WHEN comp_name ILIKE %s THEN 3
                         ELSE 4
                     END,
                     comp_name
                 LIMIT 5;
             ''', (search_pattern_like, name, # ILIKE for contains name/exact symbol
                   name, name, # Exact name/symbol for ORDER BY
                   search_pattern_starts)) # Starts with for ORDER BY

             results = cur.fetchall()
             for row in results:
                 # Add extracted name for context, avoid duplicates based on fin_code
                 company_data = dict(row)
                 company_data['extracted_term'] = name
                 # Basic check to avoid adding the same company multiple times
                 if not any(c['fin_code'] == company_data['fin_code'] for c in all_results):
                      all_results.append(company_data)

        cur.close()

    except Exception as e:
        logger.exception(f"Error in extract_company_route: {e}")
        return jsonify({"error": "Database error", "details": make_json_safe(str(e))}), 500
    finally:
        if conn:
            conn.close()

    if all_results:
        # Sort final combined results (optional, DB already sorted partially)
        # Example: Sort by how well the extracted term matches the found name
        from rapidfuzz import fuzz
        for company in all_results:
             company['match_score'] = fuzz.WRatio(company['extracted_term'], company['comp_name'].lower())
        all_results.sort(key=lambda x: x['match_score'], reverse=True)


        return jsonify({
            'status': 'success',
            'companies': make_json_safe(all_results[:10]), # Limit final output
            'extracted_terms': names_to_search
        })
    else:
        return jsonify({
            'status': 'no_match_found',
            'message': f'Identified potential terms ({", ".join(names_to_search)}), but found no matching companies in the database.',
            'extracted_terms': names_to_search
        }), 404


def extract_company_names(question):
    """
    Extract potential company names from a question using heuristics.
    (Kept original implementation, but consider LLM is better)
    """
    company_indicators = [
        'about', 'for', 'on', 'regarding', 'of',
        'tell me about', 'what about', 'how is',
        'information on', 'analysis of', 'details of',
        'financials of', 'stock of', 'share price of'
    ]
    common_companies = [
        'tata', 'reliance', 'infosys', 'hdfc', 'icici',
        'bharti', 'airtel', 'mahindra', 'wipro', 'adani'
    ] # This list is limited

    potential_companies = []
    question_lower = question.lower()

    # 1. Check for common full names explicitly mentioned (simple check)
    # Requires a larger list or DB lookup for effectiveness
    # Example check:
    # if "reliance industries" in question_lower: potential_companies.append("reliance industries")
    # elif "tata motors" in question_lower: potential_companies.append("tata motors")
    # ... etc. ...

    # 2. Check for common partial names
    found_common = False
    for company in common_companies:
        # Use regex to match whole words to avoid matching 'tata' in 'ratatan'
        if re.search(r'\b' + re.escape(company) + r'\b', question_lower):
            potential_companies.append(company)
            found_common = True

    # 3. If no common partial names found, try indicator phrases
    if not found_common:
        for indicator in company_indicators:
            # Ensure indicator is followed by a space
            indicator_phrase = indicator + ' '
            if indicator_phrase in question_lower:
                # Split only once to get text after the first occurrence
                parts = question_lower.split(indicator_phrase, 1)
                if len(parts) > 1:
                    after_indicator = parts[1].strip()
                    # Extract up to the next punctuation or common conjunction/verb
                    # Improved regex to handle more delimiters
                    match = re.split(r'[,\.?!;:]|\s+(and|or|is|was|has|had|for|the|a|an|in|on|at)\s+', after_indicator, 1)
                    company_name_candidate = match[0].strip()

                    # Basic validation: not too long, contains letters
                    if 2 < len(company_name_candidate) < 50 and re.search(r'[a-zA-Z]', company_name_candidate):
                         potential_companies.append(company_name_candidate)

    # 4. If still nothing, look for capitalized words (less reliable)
    if not potential_companies:
        words = question.split()
        for i, word in enumerate(words):
            # Check if it's likely a proper noun (starts upper, not all upper, longer than 1 char)
            # Exclude start of sentence unless it's a single word question
            if len(word) > 1 and word[0].isupper() and not word.isupper() and (i > 0 or len(words) == 1):
                # Simple check for multi-word proper nouns (e.g., HDFC Bank)
                if i + 1 < len(words) and len(words[i+1]) > 0 and words[i+1][0].isupper():
                     # Add potential two-word name
                     potential_companies.append(f"{word} {words[i+1]}")
                     # Add single word too, might be correct on its own
                     potential_companies.append(word)
                else:
                    potential_companies.append(word) # Add single capitalized word


    # Clean up: remove duplicates and very short strings, prioritize longer names
    unique_names = sorted(list(set(p for p in potential_companies if len(p) > 2)), key=len, reverse=True)
    return unique_names[:5] # Return top 5 longest unique candidates