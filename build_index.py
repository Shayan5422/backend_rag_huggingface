import os
os.environ['OMP_NUM_THREADS'] = '1' # Limit OpenMP threads, might help prevent crashes
import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
import pickle
import json # Import json module
from tqdm import tqdm

# --- Configuration ---
MODEL_DATA_DIR = "model_data_json"  # Path to downloaded JSON data
INDEX_FILE = "index.faiss"
MAP_FILE = "index_to_metadata.pkl" # Changed filename to reflect content
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # Efficient and good quality model
ENCODE_BATCH_SIZE = 32  # Process descriptions in smaller batches
# Tags to exclude from indexing text
COMMON_EXCLUDED_TAGS = {'transformers'} # Add other common tags if needed
EXCLUDED_TAG_PREFIXES = ('arxiv:', 'base_model:', 'dataset:', 'diffusers:', 'license:') # Add other prefixes if needed
MODEL_EXPLANATION_KEY = "model_explanation_gemini" # Key for the new explanation field
# ---

def load_model_data(directory):
    """Loads model data, filters tags (by length, common words, prefixes), and combines relevant info for indexing."""
    all_texts = [] # Store combined text (model_id + description + filtered_tags)
    all_metadata = [] # Store dicts: {'model_id': ..., 'tags': ..., 'downloads': ...}
    print(f"Loading model data from JSON files in: {directory}")
    if not os.path.isdir(directory):
        print(f"Error: Directory not found: {directory}")
        return [], []

    filenames = [f for f in os.listdir(directory) if f.endswith(".json")] # Look for .json files
    for filename in tqdm(filenames, desc="Reading JSON files"):
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Ensure required fields exist
                if 'description' in data and 'model_id' in data:
                    description = data['description']
                    model_id = data['model_id'] # Get model_id
                    if description: # Only index if description is not empty
                        original_tags = data.get('tags', [])
                        # Filter tags: remove short tags, common tags, and tags with specific prefixes
                        filtered_tags = [
                            str_tag for tag in original_tags
                            if (
                                tag and isinstance(tag, str) and # Ensure tag exists and is a string
                                len(tag) > 3 and
                                (str_tag := str(tag)).lower() not in COMMON_EXCLUDED_TAGS and
                                not str_tag.lower().startswith(EXCLUDED_TAG_PREFIXES) # Check for prefixes
                            )
                        ]
                        tag_string = " ".join(filtered_tags)
                        explanation = data.get(MODEL_EXPLANATION_KEY) # Get the new explanation

                        # --- Construct combined text with priority weighting ---
                        text_parts = []
                        # 1. Add explanation (repeated for emphasis) if available
                        if explanation and isinstance(explanation, str):
                            text_parts.append(f"Summary: {explanation}")
                            text_parts.append(f"Summary: {explanation}") # Repeat for higher weight
                        # 2. Add model name
                        text_parts.append(f"Model: {model_id}")
                        # 3. Add filtered tags if available
                        if tag_string:
                            text_parts.append(f"Tags: {tag_string}")
                        # 4. Add original description
                        text_parts.append(f"Description: {description}")

                        combined_text = " ".join(text_parts).strip() # Join all parts
                        # --- End construction ---

                        all_texts.append(combined_text)
                        # Add explanation to metadata as well for potential display
                        metadata_entry = {
                            "model_id": model_id,
                            "tags": original_tags, # Keep ORIGINAL tags in metadata
                            "downloads": data.get('downloads', 0)
                        }
                        if explanation and isinstance(explanation, str):
                            metadata_entry[MODEL_EXPLANATION_KEY] = explanation
                        all_metadata.append(metadata_entry)
                else:
                    print(f"Warning: Skipping {filename}, missing 'description' or 'model_id' key.")
        except json.JSONDecodeError:
            print(f"Warning: Skipping {filename}, invalid JSON.")
        except Exception as e:
            print(f"Warning: Could not read or process {filename}: {e}")

    print(f"Loaded data for {len(all_texts)} models with valid descriptions after tag filtering.")
    return all_texts, all_metadata

def build_and_save_index(texts_to_index, metadata_list):
    """Builds and saves the FAISS index and metadata mapping based on combined text."""
    if not texts_to_index:
        print("No text data to index.")
        return

    print(f"Loading sentence transformer model: {EMBEDDING_MODEL}")
    # Consider adding device='mps' if on Apple Silicon and PyTorch supports it well enough,
    # but start with CPU for stability.
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Generating embeddings for combined text in batches of {ENCODE_BATCH_SIZE}...")
    all_embeddings = []
    for i in tqdm(range(0, len(texts_to_index), ENCODE_BATCH_SIZE), desc="Encoding batches"):
        batch = texts_to_index[i:i+ENCODE_BATCH_SIZE]
        batch_embeddings = model.encode(batch, convert_to_numpy=True)
        all_embeddings.append(batch_embeddings)

    if not all_embeddings:
        print("No embeddings generated. Cannot build index.")
        return

    embeddings = np.vstack(all_embeddings) # Combine embeddings from all batches

    # Ensure embeddings are float32 for FAISS
    embeddings = embeddings.astype('float32')

    # Build FAISS index
    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)  # Using simple L2 distance
    index.add(embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    # Save the index
    faiss.write_index(index, INDEX_FILE)
    print(f"FAISS index saved to: {INDEX_FILE}")

    # Create mapping from index position to metadata dictionary
    index_to_metadata = {i: metadata for i, metadata in enumerate(metadata_list)}
    with open(MAP_FILE, 'wb') as f:
        pickle.dump(index_to_metadata, f)
    print(f"Index-to-Metadata mapping saved to: {MAP_FILE}")

if __name__ == "__main__":
    combined_texts, metadata_list = load_model_data(MODEL_DATA_DIR)
    build_and_save_index(combined_texts, metadata_list)
    print("\nIndex building complete.") 