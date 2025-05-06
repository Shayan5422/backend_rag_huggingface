import os
import json
from typing import Dict, Any, Optional
import logging
import time
from openai import OpenAI, APIError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_DATA_DIR = "model_data_json"
EXPLANATION_KEY = "model_explanation_gemini"
DESCRIPTION_KEY = "description"
MAX_RETRIES = 3 # Retries for API calls
RETRY_DELAY_SECONDS = 5 # Delay between retries

# --- DeepSeek API Configuration ---
DEEPSEEK_API_KEY_ENV_VAR = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL_NAME = "deepseek-chat"

# Global client variable
client: Optional[OpenAI] = None

def configure_llm_client():
    """Configures the OpenAI client for DeepSeek API using the API key from environment variables."""
    global client
    api_key = os.getenv(DEEPSEEK_API_KEY_ENV_VAR)
    if not api_key:
        logging.error(f"Error: {DEEPSEEK_API_KEY_ENV_VAR} environment variable not set.")
        logging.error("Please set the environment variable before running the script.")
        return False
    try:
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        logging.info("DeepSeek API client configured successfully.")
        return True
    except Exception as e:
        logging.error(f"Failed to configure DeepSeek API client: {e}")
        client = None
        return False

# --- End DeepSeek API Configuration ---

def generate_explanation(model_id: str, description: str) -> Optional[str]:
    """
    Generates a short English explanation for the model based on its description
    by calling the DeepSeek API via the OpenAI library.

    Args:
        model_id: The ID of the model (for context).
        description: The model description text.

    Returns:
        A short English explanation string from DeepSeek, or None if generation fails.
    """
    global client
    if not client:
        logging.error(f"[{model_id}] DeepSeek client not configured. Cannot generate explanation.")
        return None

    if not description or not isinstance(description, str):
        logging.warning(f"[{model_id}] Description is empty or not a string. Skipping explanation generation.")
        return None

    # Truncate very long descriptions
    max_desc_length = 4000
    if len(description) > max_desc_length:
        logging.warning(f"[{model_id}] Description truncated to {max_desc_length} chars for API call.")
        description = description[:max_desc_length] + "... [truncated]"

    # Construct the messages for DeepSeek API
    messages = [
        {"role": "system", "content": "You are an AI assistant tasked with summarizing Hugging Face model descriptions concisely."},        
        {"role": "user", "content": (
            f"Analyze the following description for the Hugging Face model '{model_id}'. "
            f"Based **only** on this description, provide a concise, one-sentence explanation in English "
            f"summarizing what this model does and its primary purpose or task. "
            f"Focus on the core functionality mentioned. Avoid adding introductory phrases like 'This model is...' or 'The model...'."
            f"\n\n---\nModel Description:\n{description}\n---\n\nConcise Explanation:"
        )}
    ]

    retries = 0
    while retries < MAX_RETRIES:
        try:
            logging.info(f"[{model_id}] Calling DeepSeek API (Attempt {retries + 1}/{MAX_RETRIES})...")
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL_NAME,
                messages=messages,
                stream=False,
                max_tokens=100, # Limit response length
                temperature=0.2 # Lower temperature for more focused summary
            )

            explanation = response.choices[0].message.content.strip()
            logging.info(f"[{model_id}] Explanation received from DeepSeek: '{explanation}'")
            # Basic post-processing: remove potential quotes
            if explanation.startswith('"') and explanation.endswith('"'):
                explanation = explanation[1:-1]
            return explanation

        except APIError as e:
            retries += 1
            logging.error(f"[{model_id}] DeepSeek API Error (Attempt {retries}/{MAX_RETRIES}): {e}")
            if retries < MAX_RETRIES:
                logging.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logging.error(f"[{model_id}] Max retries reached. Failed to generate explanation via DeepSeek.")
                return None
        except Exception as e: # Catch other potential errors
            logging.error(f"[{model_id}] Unexpected error during DeepSeek API call: {e}")
            return None # Don't retry for unexpected errors

    return None

def process_json_file(filepath: str):
    """Reads, updates, and writes a single JSON file."""
    model_id = os.path.basename(filepath).replace('.json', '')
    logging.info(f"Processing {filepath}...")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logging.error(f"[{model_id}] Invalid JSON format in {filepath}. Skipping.")
        return
    except FileNotFoundError:
        logging.error(f"[{model_id}] File not found: {filepath}. Skipping.")
        return
    except Exception as e:
        logging.error(f"[{model_id}] Error reading {filepath}: {e}. Skipping.")
        return

    if not isinstance(data, dict):
        logging.error(f"[{model_id}] Expected JSON object (dict) but got {type(data)} in {filepath}. Skipping.")
        return

    description = data.get(DESCRIPTION_KEY)
    explanation_overwritten = False

    # --- Deletion Logic: Always remove existing explanation before trying to regenerate ---
    if EXPLANATION_KEY in data:
        logging.info(f"[{model_id}] Existing explanation found. Deleting before regenerating.")
        del data[EXPLANATION_KEY]
        explanation_overwritten = True # Mark that we intend to replace it

    # --- Generation Logic ---
    if not description:
         logging.warning(f"[{model_id}] Description field is missing or empty. Cannot generate explanation.")
         return

    explanation = generate_explanation(model_id, description) # Try to generate a new one

    # --- Update and Write Logic ---    
    if explanation: # Only update if generation was successful
        data[EXPLANATION_KEY] = explanation
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            if explanation_overwritten:
                 logging.info(f"[{model_id}] Successfully overwrote and updated {filepath} with new explanation.")
            else:
                 logging.info(f"[{model_id}] Successfully generated and updated {filepath} with new explanation.")
        except IOError as e:
            logging.error(f"[{model_id}] Error writing updated data to {filepath}: {e}")
        except Exception as e:
            logging.error(f"[{model_id}] Unexpected error writing {filepath}: {e}")
    else: # Explanation generation failed
         log_message = f"[{model_id}] Failed to generate new explanation for {filepath} via API."
         if explanation_overwritten:
             log_message += " Existing explanation was removed but not replaced due to API failure."
         logging.warning(log_message)


def main():
    """Main function to iterate through the directory and process files."""
    # Configure LLM client at the start
    if not configure_llm_client():
        return # Stop if API key is not configured

    if not os.path.isdir(MODEL_DATA_DIR):
        logging.error(f"Directory not found: {MODEL_DATA_DIR}")
        return

    logging.info(f"Starting processing directory: {MODEL_DATA_DIR}")
    processed_files = 0
    updated_files = 0
    skipped_files = 0

    all_files = [f for f in os.listdir(MODEL_DATA_DIR) if f.lower().endswith(".json")]
    total_files = len(all_files)
    logging.info(f"Found {total_files} JSON files to process.")

    for i, filename in enumerate(all_files):
        filepath = os.path.join(MODEL_DATA_DIR, filename)
        logging.info(f"--- Processing file {i+1}/{total_files}: {filename} ---")
        try:
            # Check if explanation exists before calling process_json_file
            # to potentially save API calls if already done.
            # However, process_json_file already has this check.
            process_json_file(filepath)
            processed_files +=1 # Count as processed even if skipped due to existing explanation

            # Check if file was actually updated (optional metric)
            # Re-read might be inefficient, could return status from process_json_file
            # For simplicity, we just log success/failure in process_json_file

        except Exception as e:
            logging.error(f"Unexpected error processing file {filename}: {e}")
            skipped_files += 1
        # Add a small delay between files to potentially avoid hitting rate limits
        time.sleep(0.5) # Adjust delay as needed


    logging.info(f"--- Processing complete ---")
    # Refine reporting slightly
    logging.info(f"Total JSON files found: {total_files}")
    logging.info(f"Files processed (attempted): {processed_files}")
    # A more accurate count of updated files would require modifying process_json_file to return status
    logging.info(f"Files skipped due to unexpected errors: {skipped_files}")

if __name__ == "__main__":
    main() 