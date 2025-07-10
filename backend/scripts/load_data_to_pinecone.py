import json
import os
import sys

# Dynamically adds the project root to sys.path to ensure module imports work correctly.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# Imports necessary functions from the Pinecone utility module.
from backend.utils.pinecone_utils import upsert_chunks_to_pinecone, initialize_pinecone_client, pc as pinecone_client_instance

# Configuration constants for the processed data file and Pinecone index.
PROCESSED_CHUNKS_FILE = 'backend/data/processed/refined_processed_chunks_v4.json'
PINECONE_INDEX_NAME = "apec2027-chatbot" 

if __name__ == "__main__":
    # Checks if the processed chunks file exists before attempting to load data.
    if not os.path.exists(PROCESSED_CHUNKS_FILE):
        print(f"Error: Processed chunks file not found at '{PROCESSED_CHUNKS_FILE}'.")
        print("Please ensure 'chunk_apec_data.py' has been run to generate this file.")
    else:
        print(f"Loading chunks from '{PROCESSED_CHUNKS_FILE}'...")
        try:
            with open(PROCESSED_CHUNKS_FILE, 'r', encoding='utf-8') as f:
                chunks_to_upsert = json.load(f)
            print(f"Successfully loaded {len(chunks_to_upsert)} chunks.")

            # Initializes the Pinecone client and then upserts the loaded chunks.
            # This sequential process ensures Pinecone is ready before data transfer.
            initialize_pinecone_client() 
            upsert_chunks_to_pinecone(PINECONE_INDEX_NAME, chunks_to_upsert)
            print(f"Data upsert to Pinecone index '{PINECONE_INDEX_NAME}' complete.")
            
            # Optional: Verifies the total record count in the Pinecone index.
            # This step provides immediate confirmation of successful data ingestion.
            if pinecone_client_instance:
                index_stats = pinecone_client_instance.describe_index_stats(PINECONE_INDEX_NAME)
                print(f"Total records in '{PINECONE_INDEX_NAME}': {index_stats.total_vector_count}")

        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from '{PROCESSED_CHUNKS_FILE}': {e}")
        except Exception as e:
            print(f"An unexpected error occurred during data upsert to Pinecone: {e}")