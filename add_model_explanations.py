import os
import json
from typing import Dict, Any, Optional
import logging
import time
# import google.generativeai as genai # Remove Gemini import
from openai import OpenAI, APIError # Add back OpenAI imports

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_DATA_DIR = "model_data_json"
EXPLANATION_KEY = "model_explanation_gemini"
DESCRIPTION_KEY = "description"
MAX_RETRIES = 3 # Retries for API calls
RETRY_DELAY_SECONDS = 5 # Delay between retries

# --- DeepSeek API Configuration (Restored) ---
DEEPSEEK_API_KEY_ENV_VAR = "DEEPSEEK_API_KEY" # Environment variable for the key
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL_NAME = "deepseek-chat"
# ---

# Remove Gemini configuration
# GEMINI_API_KEY_ENV_VAR = "GEMINI_API_KEY"
# GEMINI_MODEL_NAME = "gemini-1.5-flash-latest"

# Global client variable for DeepSeek/OpenAI client
client: Optional[OpenAI] = None # Use OpenAI client type
# gemini_model: Optional[genai.GenerativeModel] = None # Remove Gemini model variable

def configure_llm_client():
    """Configures the OpenAI client for DeepSeek API using the API key from environment variables."""
    global client
    # global gemini_model # Remove
    api_key = os.getenv(DEEPSEEK_API_KEY_ENV_VAR) # Use DeepSeek env var
    if not api_key:
        logging.error(f"Error: {DEEPSEEK_API_KEY_ENV_VAR} environment variable not set.")
        logging.error("Please set the environment variable with your DeepSeek API key before running the script.")
        return False
    try:
        # Configure OpenAI client for DeepSeek
        client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
        logging.info(f"DeepSeek API client configured successfully for model: {DEEPSEEK_MODEL_NAME}.")
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
    global client # Use OpenAI client
    # global gemini_model # Remove
    if not client:
        logging.error(f"[{model_id}] DeepSeek client not configured. Cannot generate explanation.")
        return None

    if not description or not isinstance(description, str):
        logging.warning(f"[{model_id}] Description is empty or not a string. Skipping explanation generation.")
        return None

    # Truncate very long descriptions (adjust limit back if needed for DeepSeek)
    max_desc_length = 4000
    if len(description) > max_desc_length:
        logging.warning(f"[{model_id}] Description truncated to {max_desc_length} chars for API call.")
        description = description[:max_desc_length] + "... [truncated]"

    # Construct the messages for DeepSeek API (Restore original format)
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

    # Remove Gemini prompt construction
    # prompt = (...) 

    retries = 0
    while retries < MAX_RETRIES:
        try:
            logging.info(f"[{model_id}] Calling DeepSeek API (Attempt {retries + 1}/{MAX_RETRIES})...")
            # Use OpenAI client call format
            response = client.chat.completions.create(
                model=DEEPSEEK_MODEL_NAME,
                messages=messages,
                stream=False,
                max_tokens=100, # Limit response length
                temperature=0.2 # Lower temperature for more focused summary
            )

            # Remove Gemini response handling
            # if not response.candidates: ...
            
            explanation = response.choices[0].message.content.strip() # Get explanation from OpenAI response structure
            logging.info(f"[{model_id}] Explanation received from DeepSeek: '{explanation}'")
            
            # Basic post-processing: remove potential quotes
            if explanation.startswith('"') and explanation.endswith('"'):
                explanation = explanation[1:-1]
            # Remove Gemini specific post-processing
            # explanation = explanation.replace('**', '') 
            return explanation

        # Restore specific APIError catch for OpenAI client
        except APIError as e:
            retries += 1
            logging.error(f"[{model_id}] DeepSeek API Error (Attempt {retries}/{MAX_RETRIES}): {e}")
            if retries < MAX_RETRIES:
                logging.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                logging.error(f"[{model_id}] Max retries reached. Failed to generate explanation via DeepSeek.")
                return None
        # Keep general Exception catch
        except Exception as e:
            retries += 1 # Consider retrying general errors too or handle differently
            logging.error(f"[{model_id}] Unexpected Error during API call (Attempt {retries}/{MAX_RETRIES}): {e}")
            if retries < MAX_RETRIES:
                 logging.info(f"Retrying in {RETRY_DELAY_SECONDS} seconds...")
                 time.sleep(RETRY_DELAY_SECONDS)
            else:
                logging.error(f"[{model_id}] Max retries reached. Failed to generate explanation due to unexpected errors.")
                return None

    return None # Should not be reached if loop finishes without returning

def process_json_file(filepath: str):
    """Reads, updates (only if explanation missing), and writes a single JSON file."""
    model_id = os.path.basename(filepath).replace('.json', '')
    logging.info(f"Processing {filepath}...")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        logging.error(f"[{model_id}] Invalid JSON format in {filepath}. Skipping.")
        return False # Indicate failure/skip
    except FileNotFoundError:
        logging.error(f"[{model_id}] File not found: {filepath}. Skipping.")
        return False
    except Exception as e:
        logging.error(f"[{model_id}] Error reading {filepath}: {e}. Skipping.")
        return False

    if not isinstance(data, dict):
        logging.error(f"[{model_id}] Expected JSON object (dict) but got {type(data)} in {filepath}. Skipping.")
        return False

    # --- Check if explanation already exists ---
    if EXPLANATION_KEY in data and data[EXPLANATION_KEY]: # Check if key exists AND has non-empty content
        logging.info(f"[{model_id}] Explanation already exists. Skipping generation.")
        return False # Indicate no update was needed

    # --- Deletion Logic REMOVED ---
    # if EXPLANATION_KEY in data: ...

    # --- Generation Logic ---
    description = data.get(DESCRIPTION_KEY)
    if not description:
         logging.warning(f"[{model_id}] Description field is missing or empty. Cannot generate explanation.")
         return False # Cannot generate, so no update possible

    explanation = generate_explanation(model_id, description) # Try to generate a new one

    # --- Update and Write Logic ---
    if explanation: # Only update if generation was successful
        data[EXPLANATION_KEY] = explanation
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4)
            logging.info(f"[{model_id}] Successfully generated and updated {filepath} with new explanation.")
            return True # Indicate success/update
        except IOError as e:
            logging.error(f"[{model_id}] Error writing updated data to {filepath}: {e}")
            return False
        except Exception as e:
            logging.error(f"[{model_id}] Unexpected error writing {filepath}: {e}")
            return False
    else: # Explanation generation failed
         logging.warning(f"[{model_id}] Failed to generate new explanation for {filepath} via API. File not updated.")
         return False # Indicate failure/no update


def main():
    """Main function to iterate through the directory and process files."""
    if not configure_llm_client():
        return # Stop if API key is not configured

    if not os.path.isdir(MODEL_DATA_DIR):
        logging.error(f"Directory not found: {MODEL_DATA_DIR}")
        return

    logging.info(f"Starting processing directory: {MODEL_DATA_DIR}")
    processed_files = 0
    updated_files = 0 # Count files actually updated
    skipped_existing = 0 # Count files skipped because explanation existed
    skipped_error = 0 # Count files skipped due to read/write/API errors or no description

    all_files = [f for f in os.listdir(MODEL_DATA_DIR) if f.lower().endswith(".json")]
    total_files = len(all_files)
    logging.info(f"Found {total_files} JSON files to process.")

    for i, filename in enumerate(all_files):
        filepath = os.path.join(MODEL_DATA_DIR, filename)
        logging.info(f"--- Processing file {i+1}/{total_files}: {filename} ---")
        try:
            # process_json_file now returns True if updated, False otherwise
            updated = process_json_file(filepath)
            processed_files += 1
            if updated:
                updated_files += 1
            else:
                # Need to differentiate why it wasn't updated. Re-read is inefficient.
                # Let's rely on logs from process_json_file for now.
                # A better way would be for process_json_file to return status codes.
                pass # Logging within the function indicates reason (skipped existing, API fail, etc.)

        except Exception as e:
            logging.error(f"Unexpected error processing file loop for {filename}: {e}")
            skipped_error += 1 # Count generic loop errors
        # Add a small delay between files to potentially avoid hitting rate limits
        # Adjust delay based on Gemini quota/limits (might need less than 0.5s)
        time.sleep(0.2)


    logging.info(f"--- Processing complete ---")
    logging.info(f"Total JSON files found: {total_files}")
    logging.info(f"Files processed (attempted): {processed_files}")
    logging.info(f"Files successfully updated with new explanation: {updated_files}")
    # Cannot precisely count skipped_existing vs skipped_error without better return values
    # logging.info(f"Files skipped (existing explanation, errors, or no description): {total_files - updated_files}")


if __name__ == "__main__":
    main() 