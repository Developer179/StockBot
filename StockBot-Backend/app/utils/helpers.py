# # app/utils/helpers.py
# """
# Helper functions for the Flask application.
# """
# import logging
# from datetime import datetime, date
# from decimal import Decimal
# import difflib

# logger = logging.getLogger(__name__)

# def fuzzy_filter_rows(rows, query, threshold=0.6):
#     """
#     Filter rows using fuzzy matching on company name.

#     Args:
#         rows (list): List of dictionaries containing company data
#         query (str): Search query
#         threshold (float): Similarity threshold for matching

#     Returns:
#         list: Filtered list of dictionaries that match the query
#     """
#     matches = []

#     for row in rows:
#         comp_name = row.get("comp_name", "").lower()
#         # Exact match on substring
#         if query in comp_name:
#             matches.append(row)
#             continue

#         # Fuzzy match using difflib
#         try:
#             similarity = difflib.SequenceMatcher(None, query, comp_name).ratio()
#             if similarity >= threshold:
#                 matches.append(row)
#         except Exception as e:
#             logger.warning(f"Error in fuzzy matching: {e}")

#     # Sort by similarity (most similar first)
#     return sorted(matches, key=lambda x: difflib.SequenceMatcher(
#         None, query, x.get("comp_name", "").lower()).ratio(), reverse=True)

# def make_json_safe(obj):
#     """Convert an object to a JSON-safe format.
#     Handles Decimal, datetime, date, and other non-JSON serializable objects."""
#     import decimal
#     import datetime

#     if isinstance(obj, dict):
#         return {k: make_json_safe(v) for k, v in obj.items()}
#     elif isinstance(obj, list):
#         return [make_json_safe(item) for item in obj]
#     elif isinstance(obj, (decimal.Decimal)):
#         # Convert Decimal to float for JSON serialization
#         return float(obj)
#     elif isinstance(obj, (datetime.datetime, datetime.date)):
#         # Convert datetime to string for JSON serialization
#         return obj.isoformat()
#     else:
#         return obj


# app/utils/helpers.py
import json
from decimal import Decimal
import datetime

def make_json_safe(data):
    """Converts non-serializable types in a dictionary to JSON-safe types."""
    if isinstance(data, dict):
        return {k: make_json_safe(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [make_json_safe(item) for item in data]
    elif isinstance(data, Decimal):
        # Convert Decimal to float or string as appropriate
        return float(data) # Or str(data) if precision is critical
    elif isinstance(data, (datetime.date, datetime.datetime)):
        return data.isoformat()
    elif isinstance(data, bytes):
        try:
            return data.decode('utf-8') # Attempt to decode bytes
        except UnicodeDecodeError:
            return repr(data) # Fallback for non-utf8 bytes
    # Add other type conversions if needed
    return data

def compute_data_hash(data):
    """Computes an MD5 hash of JSON-serializable data."""
    import hashlib
    # Use make_json_safe first to ensure consistency before hashing
    safe_data = make_json_safe(data)
    return hashlib.md5(json.dumps(safe_data, sort_keys=True).encode()).hexdigest()