import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # Disable tokenizer parallelism
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import traceback

app = Flask(__name__) # Create app object FIRST
# Allow requests from the Vercel frontend and localhost for development
CORS(app, origins=["http://127.0.0.1:3000", "http://localhost:3000", "https://rag-huggingface.vercel.app"], supports_credentials=True)

# --- Configuration ---
INDEX_FILE = "index.faiss"
MAP_FILE = "index_to_metadata.pkl"
EMBEDDING_MODEL = 'all-mpnet-base-v2'
# Corrected path joining for model_data_json - relative to app.py location
MODEL_DATA_DIR = os.path.join(os.path.dirname(__file__), 'model_data_json')
# ---

# --- Global variables for resources ---
faiss = None
pickle = None
index = None
index_to_metadata = None
model = None
SentenceTransformer = None # Keep track of the imported class
RESOURCES_LOADED = False
# ---

def load_resources():
    """Loads all necessary resources (model, index, map) only once."""
    global faiss, pickle, index, index_to_metadata, model, SentenceTransformer, RESOURCES_LOADED
    if RESOURCES_LOADED: # Prevent re-loading
        print("Resources already loaded.")
        return

    print("Loading resources...")
    try:
        # Deferred Import of Faiss and Pickle inside the function
        print("Importing Faiss and Pickle...")
        import faiss as faiss_local
        import pickle as pickle_local
        faiss = faiss_local
        pickle = pickle_local
        print("Faiss and Pickle imported successfully.")

        # Load Sentence Transformer Model
        print(f"Importing SentenceTransformer and loading model: {EMBEDDING_MODEL}")
        from sentence_transformers import SentenceTransformer as SentenceTransformer_local
        SentenceTransformer = SentenceTransformer_local # Store the class globally if needed elsewhere
        model_local = SentenceTransformer(EMBEDDING_MODEL)
        model = model_local # Assign to global variable
        print("Sentence transformer model loaded successfully.")

        # Load FAISS Index
        index_path = os.path.join(os.path.dirname(__file__), INDEX_FILE)
        print(f"Loading FAISS index from: {index_path}")
        if not os.path.exists(index_path):
             raise FileNotFoundError(f"FAISS index file not found at {index_path}")
        index_local = faiss.read_index(index_path)
        index = index_local # Assign to global variable
        print("FAISS index loaded successfully.")

        # Load Index-to-Metadata Map
        map_path = os.path.join(os.path.dirname(__file__), MAP_FILE)
        print(f"Loading index-to-Metadata map from: {map_path}")
        if not os.path.exists(map_path):
            raise FileNotFoundError(f"Metadata map file not found at {map_path}")
        with open(map_path, 'rb') as f:
            index_to_metadata_local = pickle.load(f)
        index_to_metadata = index_to_metadata_local # Assign to global variable
        print("Index-to-Metadata map loaded successfully.")

        print("All resources loaded successfully.")
        RESOURCES_LOADED = True

    except FileNotFoundError as fnf_error:
        print(f"Error: {fnf_error}")
        print(f"Please ensure {INDEX_FILE} and {MAP_FILE} exist in the 'backend' directory relative to app.py.")
        print("You might need to run 'python build_index.py' first.")
        RESOURCES_LOADED = False # Keep as False
    except ImportError as import_error:
         print(f"Import Error loading resources: {import_error}")
         traceback.print_exc()
         RESOURCES_LOADED = False
    except Exception as e:
        print(f"Unexpected error loading resources: {e}")
        traceback.print_exc() # Print full traceback for loading errors
        RESOURCES_LOADED = False # Keep as False

# --- Load resources when the module is imported ---
# This should be executed only once by Gunicorn when it imports 'app:app'
load_resources()
# ---

@app.route('/search', methods=['POST'])
def search():
    """Handles search requests, embedding the query and searching the FAISS index."""
    # Check if resources are loaded at the beginning of the request
    if not RESOURCES_LOADED:
        # You could attempt to reload here, but it's often better to fail
        # if the initial load failed, as something is wrong with the environment/files.
        print("Error: Search request received, but resources are not loaded.")
        return jsonify({"error": "Backend resources not initialized. Check server logs."}), 500

    # Check for necessary components loaded by load_resources
    if model is None or index is None or index_to_metadata is None or faiss is None:
         print("Error: Search request received, but some core components (model, index, map, faiss) are None.")
         return jsonify({"error": "Backend components inconsistency. Check server logs."}), 500

    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    query = data['query']
    top_k = data.get('top_k', 10) # Default to top 10

    try:
        # Embed the query
        # Ensure model is not None (already checked above, but good practice)
        if model is None:
             return jsonify({"error": "Model not loaded."}), 500
        query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')

        # Search the index
        # Ensure index is not None
        if index is None:
             return jsonify({"error": "Index not loaded."}), 500
        distances, indices = index.search(query_embedding, top_k)

        # Get the results with full metadata
        results = []
        if indices.size > 0: # Check if search returned any indices
            # Ensure index_to_metadata is not None
            if index_to_metadata is None:
                print("Error: index_to_metadata is None during result processing.")
                return jsonify({"error": "Metadata map not loaded."}), 500

            for i in range(len(indices[0])):
                idx = indices[0][i]
                dist = distances[0][i]

                # Check index validity MORE robustly
                if idx < 0 or idx not in index_to_metadata:
                     print(f"Warning: Index {idx} out of bounds or not found in metadata mapping.")
                     continue # Skip this result

                metadata = index_to_metadata[idx].copy() # Copy to avoid mutating original
                metadata['distance'] = float(dist) # Add distance to the result dict

                # --- Add description from model_data_json ---
                model_id = metadata.get('model_id')
                description = None
                # Use the globally defined and corrected MODEL_DATA_DIR
                if model_id and MODEL_DATA_DIR:
                    filename = model_id.replace('/', '_') + '.json'
                    filepath = os.path.join(MODEL_DATA_DIR, filename)
                    if os.path.exists(filepath):
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                model_data = json.load(f)
                                description = model_data.get('description')
                        except Exception as e:
                            print(f"Error reading description file {filepath}: {e}")
                            # Keep description as None
                    # else: # Optional: Log if description file doesn't exist
                    #    print(f"Description file not found: {filepath}")

                metadata['description'] = description or 'No description available.'
                # ---

                results.append(metadata) # Append the whole metadata dict

        else:
            print("Warning: FAISS search returned empty indices.")

        return jsonify({"results": results})

    except Exception as e:
        print(f"Error during search: {e}")
        traceback.print_exc() # Print full traceback for search errors
        return jsonify({"error": "An error occurred during search."}), 500

# The if __name__ == '__main__': block remains removed. 