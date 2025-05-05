import logging
import sys
import traceback

# Configure basic logging for the orchestration script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_step(step_func, step_name):
    """Runs a step and logs its success or failure."""
    logging.info(f"--- Starting step: {step_name} ---")
    try:
        step_func()
        logging.info(f"--- Finished step: {step_name} successfully ---")
        return True
    except Exception as e:
        logging.error(f"--- Step failed: {step_name} ---")
        logging.error(f"Error: {e}")
        # Log the full traceback for detailed debugging
        logging.error(traceback.format_exc())
        return False

def main():
    """Runs the daily update sequence."""
    logging.info("=== Starting Daily Model Update Process ===")

    all_steps_succeeded = True

    # --- Step 1: Fetch new/updated model descriptions ---
    try:
        # Import the script's main function dynamically
        from huggingface_model_descriptions import main as fetch_models_main
        if not run_step(fetch_models_main, "Fetch Hugging Face Models"):
            all_steps_succeeded = False
            # Decide if we should continue if fetching fails (maybe index can still be built?)
            # For now, let's stop if the first step fails.
            logging.error("Stopping update process for this cycle due to failure in fetching models.")
            return # Exit the main function for this cycle
    except ImportError:
        logging.error("Failed to import huggingface_model_descriptions.py. Ensure it's in the same directory or Python path.")
        all_steps_succeeded = False
        return # Exit the main function for this cycle
    except Exception as e: # Catch any unexpected error during import/setup
        logging.error(f"Unexpected error setting up model fetching step: {e}")
        logging.error(traceback.format_exc())
        all_steps_succeeded = False
        return # Exit the main function for this cycle


    # --- Step 2: Add explanations using Gemini ---
    # Only proceed if the previous step was successful
    if all_steps_succeeded:
        try:
            from add_model_explanations import main as add_explanations_main
            # Check for API key *before* running the step
            import os
            if not os.getenv("GEMINI_API_KEY"):
                 logging.warning("GEMINI_API_KEY environment variable not set. Explanation step will fail or do nothing.")
                 # Optionally, you could skip this step entirely if the key is missing:
                 # logging.warning("Skipping explanation generation step.")
                 # pass # Move to the next step
            
            if not run_step(add_explanations_main, "Generate Model Explanations (Gemini)"):
                all_steps_succeeded = False
                # Decide if index building should proceed if explanations fail
                logging.warning("Explanation generation failed. Index will be built with potentially missing explanations.")
                # We will continue to the next step in this case

        except ImportError:
            logging.error("Failed to import add_model_explanations.py. Ensure it's in the same directory or Python path.")
            all_steps_succeeded = False
            # Stop if explanation script is missing
            return # Exit the main function for this cycle
        except Exception as e: # Catch any unexpected error during import/setup
            logging.error(f"Unexpected error setting up explanation generation step: {e}")
            logging.error(traceback.format_exc())
            all_steps_succeeded = False
            return # Exit the main function for this cycle

    # --- Step 3: Rebuild the search index ---
    # Only proceed if fetching models (Step 1) succeeded. Allow proceeding if Step 2 failed.
    if 'fetch_models_main' in locals() or 'fetch_models_main' in globals(): # Check if Step 1 setup occurred
        try:
            from build_index import main as build_index_main
            if not run_step(build_index_main, "Build Search Index (FAISS)"):
                all_steps_succeeded = False
                logging.error("Index building failed. The search index may be outdated or corrupted.")
                # Stop if index building fails
                return # Exit the main function for this cycle
        except ImportError:
            logging.error("Failed to import build_index.py. Ensure it's in the same directory or Python path.")
            all_steps_succeeded = False
            return # Exit the main function for this cycle
        except Exception as e: # Catch any unexpected error during import/setup
            logging.error(f"Unexpected error setting up index building step: {e}")
            logging.error(traceback.format_exc())
            all_steps_succeeded = False
            return # Exit the main function for this cycle


    logging.info("===========================================")
    if all_steps_succeeded:
        logging.info("=== Daily Model Update Process Completed Successfully ===")
    else:
        logging.error("=== Daily Model Update Process Completed with Errors ===")

if __name__ == "__main__":
    main() 