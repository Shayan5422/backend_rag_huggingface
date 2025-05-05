import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # Disable tokenizer parallelism to prevent potential crashes
# import faiss # Deferred import
from flask import Flask, request, jsonify
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
import numpy as np
import json
# import pickle # Deferred import

# --- Configuration ---
INDEX_FILE = "index.faiss"
MAP_FILE = "index_to_metadata.pkl"
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
MODEL_DATA_DIR = os.path.join(os.path.dirname(__file__), '/model_data_json')
# ---

app = Flask(__name__)
# CORS(app) # Enable CORS for requests from the React frontend - Replaced by more specific config below
CORS(app, origins=["http://127.0.0.1:3000", "http://localhost:3000"], supports_credentials=True) # Allow both localhost and 127.0.0.1

# --- Load Model and Index (Load once on startup) ---
print("Loading resources...")

# Initialize variables
faiss = None
pickle = None
index = None
index_to_metadata = None
model = None
RESOURCES_LOADED = False

try:
    # Load Sentence Transformer Model FIRST
    print(f"Loading sentence transformer model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Sentence transformer model loaded.")

    # Deferred Import of Faiss and Pickle
    import faiss
    import pickle
    print("Faiss and Pickle imported.")

    # Load FAISS Index and Map AFTER
    print(f"Loading FAISS index from: {INDEX_FILE}")
    index = faiss.read_index(INDEX_FILE)
    print("FAISS index loaded.")

    print(f"Loading index-to-Metadata map from: {MAP_FILE}")
    with open(MAP_FILE, 'rb') as f:
        index_to_metadata = pickle.load(f)
    print("Index-to-Metadata map loaded.")
    
    print("Resources loaded successfully.")
    RESOURCES_LOADED = True
except FileNotFoundError:
    print("Error: Index or map files not found!")
    print(f"Please ensure {INDEX_FILE} and {MAP_FILE} exist.")
    print("You might need to run 'python build_index.py' first.")
    RESOURCES_LOADED = False # Keep as False
except Exception as e:
    print(f"Error loading resources: {e}")
    import traceback
    traceback.print_exc() # Print full traceback for loading errors
    RESOURCES_LOADED = False # Keep as False
# ---

@app.route('/search', methods=['POST'])
def search():
    if not RESOURCES_LOADED or model is None or index is None or index_to_metadata is None or faiss is None:
        return jsonify({"error": "Backend resources not loaded properly. Check logs and restart."}), 500
        
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' in request body"}), 400

    query = data['query']
    top_k = data.get('top_k', 10) # Default to top 10

    try:
        # Embed the query
        query_embedding = model.encode([query], convert_to_numpy=True).astype('float32')

        # Search the index
        distances, indices = index.search(query_embedding, top_k)
        
        # Get the results with full metadata
        results = []
        if indices.size > 0: # Check if search returned any indices
            for i in range(len(indices[0])):
                idx = indices[0][i]
                dist = distances[0][i]
                if idx in index_to_metadata: # Check if index is valid in the metadata map
                    metadata = index_to_metadata[idx].copy() # Copy to avoid mutating original
                    metadata['distance'] = float(dist) # Add distance to the result dict

                    # --- Add description from model_data_json ---
                    model_id = metadata.get('model_id')
                    description = None
                    if model_id:
                        # Convert model_id to filename (replace / with _)
                        filename = model_id.replace('/', '_') + '.json'
                        filepath = os.path.join(MODEL_DATA_DIR, filename)
                        try:
                            with open(filepath, 'r', encoding='utf-8') as f:
                                model_data = json.load(f)
                                description = model_data.get('description')
                        except Exception as e:
                            description = None
                    metadata['description'] = description or 'No description available.'
                    # ---

                    results.append(metadata) # Append the whole metadata dict
                else:
                    print(f"Warning: Index {idx} not found in metadata mapping.")
        else:
            print("Warning: FAISS search returned empty indices.")

        return jsonify({"results": results})

    except Exception as e:
        print(f"Error during search: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for search errors
        return jsonify({"error": "An error occurred during search."}), 500

if __name__ == '__main__':
    if RESOURCES_LOADED:
        print("Starting Flask server...")
        app.run(debug=True, port=5000) # Runs on http://127.0.0.1:5000
    else:
        print("Flask server cannot start due to resource loading errors.") 