import os
from dotenv import load_dotenv
from pinecone import Pinecone as PC_Client, ServerlessSpec # Explicitly import ServerlessSpec
import google.generativeai as genai
import itertools
from tqdm.auto import tqdm

# Defines the path to the .env file. This explicit pathing ensures environment
# variables are loaded reliably regardless of where the script is executed from.
dotenv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '.env'))
load_dotenv(dotenv_path=dotenv_path) 

# Retrieves API keys from environment variables.
# Raises an error if essential keys are missing to prevent runtime failures.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
# Collects up to 5 Gemini API keys for rotation, enhancing rate limit management.
GEMINI_API_KEYS = [os.getenv(f"GEMINI_API_KEY_0{i}") for i in range(1, 6) if os.getenv(f"GEMINI_API_KEY_0{i}")]

if not PINECONE_API_KEY or not PINECONE_ENVIRONMENT or not GEMINI_API_KEYS:
    raise ValueError("Missing Pinecone or Gemini API keys. Please ensure your .env file is correctly configured.")

# Cycles through available Gemini API keys for distributed usage.
gemini_api_key_cycler = itertools.cycle(GEMINI_API_KEYS)

def get_gemini_embedding(text: str, task_type: str = "RETRIEVAL_DOCUMENT"):
    """
    Generates an embedding for the given text using a Gemini text embedding model.
    It rotates through configured API keys to manage rate limits.
    """
    current_api_key = next(gemini_api_key_cycler)
    genai.configure(api_key=current_api_key)
    try:
        response = genai.embed_content(model="models/text-embedding-004", content=text, task_type=task_type)
        return response['embedding']
    except Exception as e:
        # Logs the embedding error without halting the process, useful for debugging batches.
        print(f"Embedding error for '{text[:50]}...': {e}")
        return None

# Defines the standard embedding dimension for consistency with the model.
EMBEDDING_DIMENSION = 768
pc_client_instance = None # Global variable to hold the single Pinecone client instance.

def initialize_pinecone_client():
    """
    Initializes the global Pinecone client instance. This function ensures
    the client is only initialized once, preventing redundant connections.
    """
    global pc_client_instance
    if pc_client_instance is None:
        try:
            pc_client_instance = PC_Client(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            print("Pinecone client initialized successfully.")
        except Exception as e:
            print(f"Pinecone client initialization failed: {e}")
            raise # Re-raises the exception as this is a critical failure.

def create_or_connect_pinecone_index(index_name: str):
    """
    Creates a new Pinecone index if it doesn't exist, or connects to an
    existing one. Utilizes ServerlessSpec for optimized resource management.
    """
    initialize_pinecone_client() # Ensures the Pinecone client is ready.

    existing_index_names = [idx['name'] for idx in pc_client_instance.list_indexes()]

    if index_name not in existing_index_names:
        print(f"Creating Pinecone index '{index_name}' with dimension {EMBEDDING_DIMENSION}...")
        pc_client_instance.create_index(
            name=index_name, 
            dimension=EMBEDDING_DIMENSION, 
            metric='cosine', 
            spec=ServerlessSpec(cloud='aws', region=PINECONE_ENVIRONMENT) # Using ServerlessSpec directly
        )
        print(f"Index '{index_name}' created successfully.")
    else:
        print(f"Connected to existing Pinecone index '{index_name}'.")
    
    return pc_client_instance.Index(index_name) # Returns the index object for interaction.

def upsert_chunks_to_pinecone(index_name: str, chunks: list, batch_size: int = 100):
    """
    Generates embeddings for a list of text chunks and upserts them into
    the specified Pinecone index in batches for efficiency.
    Includes original text in metadata for easier retrieval and debugging.
    """
    pinecone_index = create_or_connect_pinecone_index(index_name)
    print(f"Preparing to upsert {len(chunks)} chunks to '{index_name}'...")
    
    for i in tqdm(range(0, len(chunks), batch_size), desc="Upserting chunks"):
        batch = chunks[i:i + batch_size]
        vectors_to_upsert = []
        for chunk in batch:
            embedding = get_gemini_embedding(chunk["content"])
            if embedding:
                chunk_metadata = chunk["metadata"].copy()
                # Stores the original text in metadata for direct retrieval,
                # avoiding re-fetching from a separate source.
                chunk_metadata['original_text'] = chunk["content"] 
                
                vectors_to_upsert.append({
                    "id": chunk["id"],
                    "values": embedding,
                    "metadata": chunk_metadata
                })
        
        if vectors_to_upsert:
            try:
                pinecone_index.upsert(vectors=vectors_to_upsert)
            except Exception as e:
                # Logs batch-specific errors without stopping the entire upsert process.
                print(f"Error upserting batch {i}-{i+len(vectors_to_upsert)}: {e}")
    print(f"Finished upserting all batches to '{index_name}'.")

def query_pinecone_index(index_name: str, query_text: str, top_k: int = 5):
    """
    Queries the specified Pinecone index with a given text query.
    It embeds the query, performs a similarity search, and retrieves
    the top 'k' matching chunks along with their scores and metadata.
    """
    pinecone_index = create_or_connect_pinecone_index(index_name)
    query_embedding = get_gemini_embedding(query_text, task_type="RETRIEVAL_QUERY")
    
    if not query_embedding: 
        print("Could not generate embedding for the query. Returning empty results.")
        return []
    
    try:
        results = pinecone_index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
        retrieved_chunks = []
        for match in results.matches:
            retrieved_chunks.append({
                "id": match.id,
                "score": match.score,
                "content": match.metadata.get('original_text', 'N/A'), # Retrieves original text from metadata
                "metadata": match.metadata
            })
        return retrieved_chunks
    except Exception as e:
        print(f"Error querying Pinecone index '{index_name}': {e}")
        return []