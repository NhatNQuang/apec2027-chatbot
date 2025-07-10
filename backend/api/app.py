from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
from datetime import datetime
import re
from langdetect import detect
import os
from backend.scripts.rag_pipeline import load_json_data

app = FastAPI()

# Configuration for data paths. I've centralizing these for clarity.
EMBEDDINGS_DIR = "backend/data/embeddings"
RAW_DATA_DIR = "backend/data/raw"
INDEX_PATH = os.path.join(EMBEDDINGS_DIR, "apec2025_index.bin")
METADATA_PATH = os.path.join(EMBEDDINGS_DIR, "apec2025_metadata.json")
# This path points to the specific raw data file. It's crucial for the RAG pipeline.
JSON_PATH = os.path.join(RAW_DATA_DIR, "apec2025_all_info_20250708_221755.json")

# Loading the core components of the RAG system. This setup ensures everything is
# ready when the application starts.
model = SentenceTransformer('all-MiniLM-L6-v2')
index = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, 'r', encoding='utf-8') as f:
    metadata = json.load(f)
with open(JSON_PATH, 'r', encoding='utf-8') as f:
    # Using the pre-defined load_json_data function for consistency with the RAG pipeline.
    chunks, _ = load_json_data(f)

# Defines the expected structure for incoming API requests.
class Query(BaseModel):
    text: str

def query_rag(query: str, k: int = 5):
    """
    Performs a RAG query, integrating language detection and date-based event filtering.
    This function encapsulates the core logic for retrieving relevant information.
    """
    lang = detect(query)
    query_embedding = model.encode([query])[0]
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)

    current_date = datetime.now().strftime('%Y-%m-%d')
    all_results = []
    events_for_today = []

    for idx in indices[0]:
        meta = metadata[idx]
        text = chunks[idx]
        
        # Checking for event details and date to enable specific filtering.
        if 'event' in meta and 'date' in meta and meta['date'] != '-':
            try:
                # My logic for parsing various date formats. This is important for robustness.
                date_match = re.search(r'(\w+ \d{1,2}(?: - \d{1,2})?, \d{4})|(\w+ \d{1,2}, \d{4})', meta['date'])
                if date_match:
                    # Prioritizing the first group that matches to handle both range and single dates.
                    date_part = date_match.group(1) or date_match.group(2)
                    start_date = datetime.strptime(date_part.split(' - ')[0], '%B %d, %Y')
                    if start_date.strftime('%Y-%m-%d') == current_date:
                        events_for_today.append({
                            'text': text,
                            'metadata': meta,
                            'distance': distances[0][list(indices[0]).index(idx)]
                        })
            except Exception as e:
                # Logging date parsing errors helps in debugging without stopping the process.
                print(f"Error parsing date '{meta['date']}': {e}")
        
        all_results.append({
            'text': text,
            'metadata': meta,
            'distance': distances[0][list(indices[0]).index(idx)]
        })
    
    # Conditional return based on query intent for "today's events".
    # This decision flow prioritizes immediate user needs.
    if "hôm nay" in query.lower() or "today" in query.lower():
        if events_for_today:
            return events_for_today
        return [{"text": "Không có sự kiện nào diễn ra hôm nay theo lịch APEC 2025.", "metadata": {}, "distance": 0}]
    
    return all_results

@app.post("/query")
async def query_endpoint(query: Query):
    """
    API endpoint for handling chat queries. It routes the user's request
    through the RAG pipeline.
    """
    results = query_rag(query.text)
    return {"results": results}

if __name__ == "__main__":
    import uvicorn
    # Running the FastAPI application. This is the entry point for the API server.
    uvicorn.run(app, host="0.0.0.0", port=8000)