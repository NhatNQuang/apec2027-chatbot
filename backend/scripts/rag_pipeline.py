import json
import os
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Defines the output directories for raw data and generated embeddings.
RAW_DATA_DIR = "backend\\data\\raw"
EMBEDDINGS_DIR = "backend/data/embeddings"
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

def load_json_data(json_file_path):
    """
    Loads JSON data from the specified path and transforms it into a list of
    text chunks and corresponding metadata. This function is critical for
    preparing the data for the RAG pipeline.
    """
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    chunks = []
    metadata = []
    
    for page in data:
        category = page.get('category', '')
        page_name = page.get('page', '')
        for section in page['sections']:
            section_title = section.get('section', '')
            content = section.get('content', '')
            
            if isinstance(content, str):
                chunks.append(content)
                metadata.append({
                    'category': category,
                    'page': page_name,
                    'section': section_title
                })
            elif isinstance(content, list):
                for item in content:
                    sub_section = item.get('sub_section', '')
                    sub_content = item.get('content', '')
                    
                    if isinstance(sub_content, str):
                        chunks.append(sub_content)
                        metadata.append({
                            'category': category,
                            'page': page_name,
                            'section': section_title,
                            'sub_section': sub_section
                        })
                    elif isinstance(sub_content, dict):
                        if 'main_content' in sub_content:
                            chunks.append(sub_content['main_content'])
                            metadata.append({
                                'category': category,
                                'page': page_name,
                                'section': section_title,
                                'sub_section': sub_section
                            })
                        if 'events' in sub_content:
                            for event in sub_content['events']:
                                event_text = f"{event.get('event', '')} on {event.get('date', '')} at {event.get('venue', '')}"
                                chunks.append(event_text)
                                metadata.append({
                                    'category': category,
                                    'page': page_name,
                                    'section': section_title,
                                    'sub_section': sub_section,
                                    'event': event.get('event', '')
                                })
                        if 'seasons' in sub_content:
                            for season in sub_content['seasons']:
                                season_text = f"{season.get('season', '')} ({season.get('period', '')}): {season.get('description', '')}"
                                chunks.append(season_text)
                                metadata.append({
                                    'category': category,
                                    'page': page_name,
                                    'section': section_title,
                                    'sub_section': sub_section,
                                    'season': season.get('season', '')
                                })
                        if 'sub_sections' in sub_content:
                            for sub_sub_section in sub_content['sub_sections']:
                                sub_sub_title = sub_sub_section.get('sub_section', '')
                                for item in sub_sub_section.get('items', []):
                                    chunks.append(item)
                                    metadata.append({
                                        'category': category,
                                        'page': page_name,
                                        'section': section_title,
                                        'sub_section': sub_section,
                                        'sub_sub_section': sub_sub_title
                                    })
                        if 'information' in sub_content:
                            for info in sub_content['information']:
                                chunks.append(info)
                                metadata.append({
                                    'category': category,
                                    'page': page_name,
                                    'section': section_title,
                                    'sub_section': sub_section,
                                    'info': info
                                })
    
    return chunks, metadata

def create_embeddings(chunks, metadata, model_name='all-MiniLM-L6-v2'):
    """
    Generates embeddings for text chunks using a SentenceTransformer model
    and creates a FAISS index for efficient similarity search. The index
    and associated metadata are saved to disk.
    """
    model = SentenceTransformer(model_name)
    
    embeddings = model.encode(chunks, show_progress_bar=True)
    
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # Using L2 distance for similarity
    index.add(embeddings.astype('float32'))
    
    # Persisting the FAISS index and metadata for later retrieval.
    faiss.write_index(index, os.path.join(EMBEDDINGS_DIR, 'apec2025_index.bin'))
    with open(os.path.join(EMBEDDINGS_DIR, 'apec2025_metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
    
    return model, index, chunks, metadata

def query_rag(query, model, index, chunks, metadata, k=5):
    """
    Performs a Retrieval-Augmented Generation (RAG) query.
    It embeds the query, searches the FAISS index for relevant chunks,
    and returns the top 'k' results with their text, metadata, and distance.
    """
    query_embedding = model.encode([query])[0]
    
    distances, indices = index.search(np.array([query_embedding]).astype('float32'), k)
    
    results = []
    for idx in indices[0]:
        # The original code used list(indices[0]).index(idx) which is inefficient.
        # Direct indexing `distances[0][idx]` is also incorrect as `idx` is the FAISS internal index.
        # The `distances` array already corresponds to the `indices` array.
        # So, `distances[0][list(indices[0]).index(idx)]` should be simplified.
        # Here, I'm assuming 'distances[0]' and 'indices[0]' are already aligned
        # where `distances[0][i]` corresponds to `chunks[indices[0][i]]`.
        # I'll stick to your current result structure, but note this potential optimization
        # if performance becomes a bottleneck on large 'k'. For small 'k', it's negligible.
        results.append({
            'text': chunks[idx],
            'metadata': metadata[idx],
            'distance': distances[0][np.where(indices[0] == idx)[0][0]] # Corrected logic
        })
    
    return results

if __name__ == "__main__":
    # My specified input JSON file containing processed data.
    json_file = os.path.join(RAW_DATA_DIR, "apec2025_all_info_20250708_221755.json")
    
    print(f"Starting RAG pipeline with data from: {json_file}")

    # Step 1: Load and process JSON data into chunks and metadata.
    chunks, metadata = load_json_data(json_file)
    print(f"Loaded {len(chunks)} chunks for embedding.")
    
    # Step 2: Create embeddings and build the FAISS index.
    model, index, chunks, metadata = create_embeddings(chunks, metadata)
    print("Embeddings created and FAISS index saved.")
    
    # Step 3: Perform a sample query to test the RAG pipeline.
    sample_query = "Where is Informal Senior Officials’ Meeting (ISOM) held?"
    print(f"\nPerforming sample query: '{sample_query}'")
    results = query_rag(sample_query, model, index, chunks, metadata)
    
    # Displaying query results for verification.
    print("\n--- Query Results ---")
    if results:
        for i, result in enumerate(results):
            print(f"Result {i+1}:")
            print(f"  Text: {result['text']}")
            print(f"  Metadata: {result['metadata']}")
            print(f"  Distance: {result['distance']:.4f}") # Formatting distance for readability
            print("-" * 50)
    else:
        print("No results found for the query.")