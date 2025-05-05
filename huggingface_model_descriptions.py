import os
import requests
from tqdm import tqdm
import time
import re
import json
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, EntryNotFoundError, HFValidationError
from requests.exceptions import RequestException
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle # Add pickle for caching

# Create a directory to store JSON data
OUTPUT_DIR = "model_data_json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Number of worker threads for parallel processing - REDUCED
NUM_WORKERS = 4

# Add a delay between download attempts across threads
DOWNLOAD_DELAY_SECONDS = 0.2 # Adjust as needed

# --- README Cleaning ---
def clean_readme_content(text):
    """Basic cleaning of README markdown: remove code blocks, links."""
    if not text:
        return ""
    
    # Remove fenced code blocks (``` ... ```)
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove inline code (`...`)
    text = re.sub(r'`[^`]+`', '', text)
    # Remove markdown links ([text](url))
    text = re.sub(r'\[([^]]+)\]\([^)]+\)', r'\1', text) # Keep link text
    # Remove standalone URLs (simple version)
    text = re.sub(r'https?://\S+', '', text)
    # Remove markdown images (![alt](url))
    text = re.sub(r'!\[[^]]*\]\([^)]+\)', '', text)
    # Replace multiple newlines/spaces with single ones
    text = ' '.join(text.split())
    return text
# ---

MODELS_CACHE_FILE = "models_list_cache.pkl" # File to cache the raw model list

def get_all_models_with_downloads(min_downloads=10000):
    """Fetch all models from Hugging Face with at least min_downloads, using a local cache for the list."""
    models_list = None
    
    # 1. Check for cache
    if os.path.exists(MODELS_CACHE_FILE):
        try:
            print(f"Loading cached model list from {MODELS_CACHE_FILE}...")
            with open(MODELS_CACHE_FILE, 'rb') as f:
                models_list = pickle.load(f)
            print(f"Loaded {len(models_list)} models from cache.")
        except Exception as e:
            print(f"Error loading cache file {MODELS_CACHE_FILE}: {e}. Fetching from API.")
            models_list = None # Ensure fetching if cache loading fails
    
    # 2. Fetch from API if cache doesn't exist or failed to load
    if models_list is None:
        print(f"Fetching all models with more than {min_downloads} downloads from API...")
        try:
            print("Initializing HfApi...")
            api = HfApi()
            print("HfApi initialized. Calling list_models...")
            # Fetch the iterator
            models_iterator = api.list_models(sort="downloads", direction=-1, fetch_config=False, cardData=True)
            print("list_models call returned. Converting iterator to list...")
            # Convert the iterator to a list TO ALLOW CACHING
            models_list = list(models_iterator)
            print(f"Converted to list with {len(models_list)} models.")
            
            # Save to cache
            try:
                print(f"Saving model list to cache file: {MODELS_CACHE_FILE}...")
                with open(MODELS_CACHE_FILE, 'wb') as f:
                    pickle.dump(models_list, f)
                print("Model list saved to cache.")
            except Exception as e:
                print(f"Error saving cache file {MODELS_CACHE_FILE}: {e}")
            
        except Exception as e:
            print(f"Error during HfApi initialization or list_models call: {e}")
            return [] # Return empty list on error
    
    # 3. Filter the loaded/fetched list
    if not models_list:
        print("Model list is empty after fetching/loading.")
        return []
    
    qualifying_models = []
    print(f"Filtering {len(models_list)} models by download count...")
    for model in models_list: # Iterate through the list (from cache or API)
        # No need for prints inside this loop now, as it should be fast
        if not hasattr(model, 'downloads') or model.downloads is None:
            continue
        
        if model.downloads < min_downloads:
            # Since the list is sorted by downloads, we can stop
            break
        
        qualifying_models.append(model)
    
    print(f"Found {len(qualifying_models)} models with more than {min_downloads} downloads")
    return qualifying_models

def get_model_readme(model_id):
    """Get README.md content for a specific model using hf_hub_download. Returns None if not found or inaccessible."""
    filenames_to_try = ["README.md", "readme.md"]
    branches_to_try = ["main", "master"]
    
    for branch in branches_to_try:
        for filename in filenames_to_try:
            try:
                # print(f"Attempting download: repo={model_id}, branch={branch}, file={filename}") # Debug
                # Use hf_hub_download which uses stored token
                readme_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    revision=branch,
                    repo_type="model",
                    local_files_only=False, # Ensure it tries to download
                    # token=True # Often not needed if logged in via CLI, but can be explicit
                )
                
                # If download succeeded, read the content
                # print(f"Successfully downloaded {filename} from {branch} to {readme_path}") # Debug
                with open(readme_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                return content

            except RepositoryNotFoundError:
                print(f"Repository {model_id} not found.")
                return None # If repo doesn't exist, no point trying other files/branches
            except EntryNotFoundError: 
                # print(f"{filename} not found in branch {branch} for {model_id}. Trying next...") # Debug
                continue # File not found in this specific branch/filename combination, try next
            except HFValidationError as e: # Catch invalid repo ID or filename errors
                 print(f"Validation error for {model_id} (branch: {branch}, file: {filename}): {e}")
                 continue # Try next filename/branch
            except Exception as e: # Catch other potential errors (like 401 HfHubHTTPError, network issues)
                print(f"Error downloading {filename} from branch {branch} for {model_id}: {e}")
                # Check if it's a likely authentication error (401/403)
                if "401" in str(e) or "403" in str(e):
                    print(f"Authentication error (401/403) for {model_id}. Ensure you are logged in and accepted terms.")
                    return None # Don't try other files/branches if auth failed
                # For other errors, we continue to the next filename/branch attempt
                continue 
                
    # If all attempts failed
    print(f"Could not fetch README for {model_id} from any standard location.")
    return None

def get_filename_for_model(model_id):
    """Generate JSON filename for a model"""
    safe_id = model_id.replace("/", "_")
    return os.path.join(OUTPUT_DIR, f"{safe_id}.json") # Change extension to .json

def save_model_data(model_id, data):
    """Save model data (description, tags, downloads) to a JSON file."""
    filename = get_filename_for_model(model_id)
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        return filename
    except Exception as e:
        print(f"Error saving JSON for {model_id} to {filename}: {e}")
        return None

def file_exists_for_model(model_id):
    """Check if a JSON file already exists for this model"""
    filename = get_filename_for_model(model_id)
    return os.path.exists(filename)

def process_model(model):
    """Process a single model - fetch README, clean it, save as JSON."""
    model_id = model.modelId
    downloads = model.downloads
    tags = getattr(model, 'tags', []) # Get tags if available
    
    # Check if JSON file already exists
    if file_exists_for_model(model_id):
        return (model_id, downloads, None, "skipped")
    
    # --- Add Delay Before Download Attempt ---
    time.sleep(DOWNLOAD_DELAY_SECONDS) 
    # ---------------------------------------
    
    # Get model README content
    readme_content = get_model_readme(model_id)
    
    # If README is not available, skip saving this model
    if readme_content is None:
        return (model_id, downloads, None, "no_readme")
    
    # Clean the README
    cleaned_readme = clean_readme_content(readme_content)
    
    # Prepare data payload
    model_data = {
        "model_id": model_id,
        "downloads": downloads,
        "tags": tags,
        "description": cleaned_readme
    }
    
    # Save data as JSON
    filename = save_model_data(model_id, model_data)
    if filename:
        return (model_id, downloads, filename, "downloaded")
    else:
        return (model_id, downloads, None, "save_failed")

def main():
    qualifying_models = get_all_models_with_downloads(min_downloads=10000)
    if not qualifying_models:
        print("No qualifying models found")
        return
    
    print(f"Processing {len(qualifying_models)} models, saving to '{OUTPUT_DIR}'...")
    downloaded = 0
    skipped = 0
    no_readme = 0
    failed = 0
    
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_model = {executor.submit(process_model, model): model for model in qualifying_models}
        
        for future in tqdm(as_completed(future_to_model), total=len(qualifying_models)):
            try:
                model_id, downloads, filename, status = future.result()
                if status == "downloaded":
                    # Don't print every success to avoid clutter
                    # print(f"Saved data for {model_id} ({downloads} downloads) to {filename}")
                    downloaded += 1
                elif status == "skipped":
                    skipped += 1
                elif status == "no_readme":
                    no_readme += 1
                else: # save_failed or other errors
                    failed += 1
            except Exception as e:
                # Extract model_id for better error reporting if possible
                processed_model = future_to_model[future]
                print(f"Error processing model {getattr(processed_model, 'modelId', 'unknown')}: {e}")
                failed += 1
    
    print(f"\nCompleted! Downloaded: {downloaded}, Skipped existing: {skipped}, No README found: {no_readme}, Failed: {failed}")

if __name__ == "__main__":
    main() 