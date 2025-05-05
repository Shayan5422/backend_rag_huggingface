import os

OUTPUT_DIR = "model_descriptions"
NO_README_TEXT = "No README available"

def main():
    if not os.path.isdir(OUTPUT_DIR):
        print(f"Error: Directory '{OUTPUT_DIR}' not found.")
        return

    deleted_count = 0
    print(f"Scanning '{OUTPUT_DIR}' for files containing only '{NO_README_TEXT}'...")

    for filename in os.listdir(OUTPUT_DIR):
        filepath = os.path.join(OUTPUT_DIR, filename)
        
        # Check if it's a file
        if os.path.isfile(filepath):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content == NO_README_TEXT:
                    os.remove(filepath)
                    print(f"Deleted: {filename}")
                    deleted_count += 1
            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"\nCleanup complete. Deleted {deleted_count} files.")

if __name__ == "__main__":
    main() 