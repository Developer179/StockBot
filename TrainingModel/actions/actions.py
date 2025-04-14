# actions/actions.py
import logging
from typing import Any, Text, Dict, List, Optional

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher
from rasa_sdk.types import DomainDict
from difflib import get_close_matches
import os
import json

# Configure logger for this module specifically if needed, or rely on root config
logger = logging.getLogger(__name__)
# Ensure debug messages are shown if root logger level is INFO
# logger.setLevel(logging.DEBUG) # Uncomment this if your root logger level is higher than DEBUG

# --- Mapping: Canonical Identifier → Table Name ---
CANONICAL_ID_TO_SOURCE_MAP = {
    # company_master
    "comp_name": "company_master", "scrip_name": "company_master", "symbol": "company_master",
    "bse_symbol": "company_master", "mcx_symbol": "company_master", "lot_size": "company_master",
    "isin": "company_master", "industry": "company_master", "sector": "company_master",
    "strike_price": "company_master", "nse_lower_limit": "company_master", "nse_upper_limit": "company_master",
    "bse_lower_limit": "company_master", "bse_upper_limit": "company_master", "segment": "company_master",
    "instrument_name": "company_master",

    # company_additional_details
    "nse_todays_low": "company_additional_details", "nse_todays_high": "company_additional_details",
    "nse_todays_open": "company_additional_details", "nse_last_closed_price": "company_additional_details",
    "bse_todays_low": "company_additional_details", "bse_todays_high": "company_additional_details",
    "bse_todays_open": "company_additional_details", "bse_last_closed_price": "company_additional_details",
    "oi": "company_additional_details",
    "short_term_verdict": "company_additional_details",
    "long_term_verdict": "company_additional_details",

    # consolidated_company_equity
    "market_capital": "consolidated_company_equity", "pe_ratio": "consolidated_company_equity",
    "bool_value": "consolidated_company_equity", "pb_ratio": "consolidated_company_equity",
    "face_value": "consolidated_company_equity", "eps": "consolidated_company_equity",
    "type": "consolidated_company_equity",

    # screeners
    "FUTURES_TOP_PRICE_GAINERS": "screeners", "NIFTY50": "screeners",
    "LONG_TERM_VERDICT_BUY": "screeners", "VOLUME_SHOCKERS": "screeners",
    "HIGH_DIVIDEND_STOCKS": "screeners", "GOLDEN_CROSSOVER": "screeners",
}

# --- Enhanced fuzzy_match_company function ---
def fuzzy_match_company(raw_name: str, threshold: float = 0.75) -> Optional[str]:

    """
    Attempts to find the closest match for a company name from a predefined list,
    with enhanced logging.
    """
    company_file = None # Initialize for logging in case of early errors
    try:
        # Dynamically construct path relative to *this file's* directory
        current_dir = os.path.dirname(os.path.abspath(__file__)) # Use absolute path of current file
        company_file = os.path.join(current_dir, '..', 'data', 'company_names.txt')
        # Log the absolute path being checked
        logger.debug(f"Fuzzy Match: Absolute path being checked for company file: {os.path.abspath(company_file)}")

        if not os.path.exists(company_file):
            logger.error(f"Fuzzy Match Error: Company name file NOT FOUND at resolved path: {os.path.abspath(company_file)}")
            return None

        # Use UTF-8 encoding for broader compatibility
        with open(company_file, "r", encoding='utf-8') as file:
            # Read companies, strip whitespace robustly, ignore empty lines
            companies = [line.strip() for line in file if line and not line.isspace()]

        if not companies:
             logger.warning(f"Fuzzy Match Warning: Company name file exists at '{os.path.abspath(company_file)}' but is EMPTY or contains only whitespace.")
             return None
        else:
             logger.debug(f"Fuzzy Match: Loaded {len(companies)} company names from file.")
             # Log first few companies for verification
             logger.debug(f"Fuzzy Match: First few companies loaded: {companies[:5]}")


        raw_name_lower = raw_name.lower()
        companies_lower = [c.lower() for c in companies]
        logger.debug(f"Fuzzy Match: Comparing '{raw_name_lower}' against {len(companies_lower)} lowercased names.")

        # Get potential matches BEFORE applying the strict threshold for debugging
        potential_matches = get_close_matches(raw_name_lower, companies_lower, n=3, cutoff=0.6) # Lower cutoff for logging
        logger.debug(f"Fuzzy Match: Potential close matches (cutoff > 0.6) for '{raw_name_lower}': {potential_matches}")

        # Now apply the actual threshold for the result
        matches = get_close_matches(raw_name_lower, companies_lower, n=1, cutoff=threshold)

        if matches:
            matched_lower = matches[0]
            # Find match by lowercased comparison, returning first match with same lowercase
            resolved_name = next((c for c in companies if c.lower() == matched_lower), None)
            if resolved_name:
                logger.info(f"Fuzzy Match SUCCESS: Input '{raw_name}' resolved to '{resolved_name}' (Score > {threshold})")
                return resolved_name
            else:
                logger.error(f"Fuzzy Match Internal Error: Could not resolve matched company from lowercased version '{matched_lower}'.")
                return None
            
        else:
            logger.warning(f"Fuzzy Match FAILED: No company name found in list matching '{raw_name}' with threshold > {threshold}. Potential matches were: {potential_matches}")
            return None

    except FileNotFoundError:
        # This case should be caught by os.path.exists, but added for extra safety
         logger.error(f"Fuzzy Match Error: FileNotFoundError encountered for path: {company_file}")
         return None
    except IOError as io_err:
        logger.error(f"Fuzzy Match Error: IOError reading company file '{company_file}': {io_err}", exc_info=True)
        return None
    except Exception as e:
        # Catch any other unexpected errors during the process
        logger.error(f"Fuzzy Match Error: Unexpected error processing '{raw_name}' using file '{company_file}': {e}", exc_info=True)
        return None


# --- ActionGetDataSource class (Keep As Is) ---
class ActionGetDataSource(Action):
    def name(self) -> Text:
        return "action_get_data_source"

    async def run(self,
                  dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: DomainDict) -> List[Dict[Text, Any]]:
        # ... (previous implementation is fine) ...
        all_data_concepts = list(tracker.get_latest_entity_values("data_concept"))
        all_company_names = list(tracker.get_latest_entity_values("company_name"))
        data_concept = next(iter(all_data_concepts), None)
        company_name = next(iter(all_company_names), None)
        logger.debug(f"[{self.name()}] Intent: {tracker.latest_message.get('intent', {}).get('name')}, Text: '{tracker.latest_message.get('text')}'")
        logger.debug(f"[{self.name()}] Entities: Concepts={all_data_concepts}, Companies={all_company_names}")
        logger.debug(f"[{self.name()}] Using Concept: {data_concept}, Company: {company_name}")
        if not data_concept:
            logger.warning(f"[{self.name()}] 'data_concept' entity missing.")
            dispatcher.utter_message(response="utter_ask_what_data")
            return []
        source_table = CANONICAL_ID_TO_SOURCE_MAP.get(data_concept)
        if not source_table:
            logger.warning(f"[{self.name()}] Could not find source table for concept: '{data_concept}'.")
            dispatcher.utter_message(response="utter_not_found", data_concept=data_concept)
            return []
        matched_company = None
        if company_name:
            matched_company = fuzzy_match_company(company_name) # Calls the enhanced function
        response_payload = {
            "query_intent": tracker.latest_message.get('intent', {}).get('name'),
            "query_text": tracker.latest_message.get('text'),
            "extracted_concept": data_concept,
            "all_extracted_concepts": all_data_concepts,
            "extracted_company": company_name,
            "all_extracted_companies": all_company_names,
            "resolved_company": matched_company,
            "source_table": source_table,
        }
        dispatcher.utter_message(json_message=response_payload)
        if matched_company:
             dispatcher.utter_message(text=f"The source for '{data_concept}' regarding '{matched_company}' is the '{source_table}' category.")
        elif company_name:
             dispatcher.utter_message(text=f"The source for '{data_concept}' is the '{source_table}' category. (Note: Company '{company_name}' was mentioned but not precisely matched).") # Refined text
        else:
             dispatcher.utter_message(text=f"The source for '{data_concept}' is the '{source_table}' category.")
        return []


# --- ActionGetInvestmentAdvice class (Keep As Is, relies on fuzzy_match_company) ---
class ActionGetInvestmentAdvice(Action):
    def name(self) -> Text:
        return "action_get_investment_advice"

    async def run(self,
                  dispatcher: CollectingDispatcher,
                  tracker: Tracker,
                  domain: DomainDict) -> List[Dict[Text, Any]]:
        # --- Configuration ---
        TARGET_CONCEPTS = ["short_term_verdict", "long_term_verdict"]
        SOURCE_TABLE = None
        try:
             # Use first concept to find table, then verify others
             SOURCE_TABLE = CANONICAL_ID_TO_SOURCE_MAP[TARGET_CONCEPTS[0]]
             for concept in TARGET_CONCEPTS:
                 if CANONICAL_ID_TO_SOURCE_MAP.get(concept) != SOURCE_TABLE:
                     logger.error(f"[{self.name()}] Config Error: Concepts {TARGET_CONCEPTS} map to different tables.")
                     dispatcher.utter_message(text="Sorry, there's an internal configuration issue getting that advice.")
                     return []
        except KeyError:
            logger.error(f"[{self.name()}] Config Error: Cannot find source table for key concept '{TARGET_CONCEPTS[0]}'.")
            dispatcher.utter_message(text="Sorry, there's an internal configuration issue getting that advice.")
            return []
        if not SOURCE_TABLE: # Should be caught above, but safety check
             logger.error(f"[{self.name()}] Config Error: SOURCE_TABLE could not be determined.")
             dispatcher.utter_message(text="Sorry, there's an internal configuration issue getting that advice.")
             return []

        # --- Extract Company ---
        company_name_entities = list(tracker.get_latest_entity_values("company_name"))
        company_name = next(iter(company_name_entities), None)
        logger.debug(f"[{self.name()}] Intent: {tracker.latest_message.get('intent', {}).get('name')}, Text: '{tracker.latest_message.get('text')}'")
        logger.debug(f"[{self.name()}] Entities: Companies={company_name_entities}")
        logger.debug(f"[{self.name()}] Using Company: {company_name}")

        if not company_name:
            logger.warning(f"[{self.name()}] 'company_name' entity missing.")
            dispatcher.utter_message(response="utter_ask_for_company")
            return []

        # --- Resolve Company ---
        # Calls the enhanced fuzzy_match_company function
        matched_company = fuzzy_match_company(company_name)

        # If matching fails, the utter_company_not_found_for_advice is triggered
        if not matched_company:
            logger.warning(f"[{self.name()}] Company resolution failed for input: '{company_name}'")
            # The fuzzy_match_company function already logged details
            dispatcher.utter_message(response="utter_company_not_found_for_advice", company=company_name)
            return [] # IMPORTANT: Return after dispatching the error message

        # --- Prepare & Dispatch Response ---
        logger.info(f"[{self.name()}] Request for advice identified for {matched_company}. Needs {TARGET_CONCEPTS} from {SOURCE_TABLE}.")
        response_payload = {
            "query_intent": "ask_investment_advice",
            "query_text": tracker.latest_message.get('text'),
            "extracted_company": company_name,
            "resolved_company": matched_company,
            "target_concepts": TARGET_CONCEPTS,
            "source_table": SOURCE_TABLE
        }
        dispatcher.utter_message(json_message=response_payload)
        dispatcher.utter_message(response="utter_fetching_advice", company=matched_company)

        return []