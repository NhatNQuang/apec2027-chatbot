# APEC 2027 AI Chatbot Demo: Enhancing International Guest Experience

🌟 **Project Overview & Highlights**  
This project is a multilingual AI chatbot I developed for APEC 2027, designed to provide accurate event information and effectively assist international guests.  
**Chatbot conversation Example**
<br>
<p align="center">
  <img src="https://github.com/NhatNQuang/image-captioning/blob/develop/apecchatbot.png" alt="Sample conversation" style="width:50%;"/>
</p>
<br>
**Core Technologies & Key Advantages:**  
- **RAG Architecture (Retrieval-Augmented Generation):** Ensures accurate, context-based responses from real data.  
- **Data Source:** Deeply crawled data from the APEC 2025 website ([https://apec2025.kr](https://apec2025.kr)).  
- **Embedding Model:** Utilizes `sentence-transformers` with the `all-MiniLM-L6-v2` model for text-to-vector conversion.  
- **Vector Database:** Flexible implementation with FAISS (CPU-based) for local demo, with readiness to integrate Pinecone (Cloud Vector Database) for future scalability.  
- **Extended LLM Intelligence:** The chatbot leverages LLM's background knowledge to answer questions beyond crawled data (e.g., Vietnamese/Phú Quốc culture), offering a versatile and comprehensive experience.  
- **Smart Multilingual Support:** Supports Vietnamese and English with automatic language detection.  
- **User Interface (UI):** Rapidly developed with Gradio, optimized for mobile, and includes convenient "Quick Replies" feature.  
- **Conversation Context Management:** Unique solution enabling the chatbot to "remember" chat history by summarizing context, ensuring coherent responses to related questions.  

## 1. Project Introduction  
This chatbot is designed to assist international guests at APEC 2027, providing detailed information about the event, schedules, venues, procedures, and the ability to answer diverse questions.  

**Key Features:**  
- **In-Depth Information Retrieval:** Covers APEC 2025 overview, event schedules, venues, entry/visa/medical procedures, and transportation details.  
- **Knowledge Expansion:** Capable of answering topics outside crawled data (e.g., Vietnamese culture) using LLM intelligence.  
- **Intuitive UI:** Minimalist, mobile-friendly design with integrated "Quick Replies" buttons.  

## 2. Project Structure  
The project is organized clearly and professionally:  

```
apec2027-chatbot/
├── backend/
│   ├── data/
│   │   ├── raw/                  # Raw crawled data
│   │   ├── processed/            # Chunked data
│   │   └── embeddings/           # FAISS index and metadata
│   ├── api/                      # FastAPI backend
│   ├── scripts/                  # Data preparation scripts
│   │   ├── crawler.py            # Web data crawling
│   │   ├── chunk_data.py         # Data processing and chunking
│   │   └── rag_pipeline.py       # Embedding and FAISS index creation
│   └── utils/                    # Shared utilities
│       └── pinecone_utils.py     # Utilities for Pinecone and Gemini Embedding
│   └── requirements.txt          # Python library dependencies for backend
├── demo/
│   └── app.py                    # Gradio UI application
├── .env.example                  # Sample environment variable configuration
└── README.md                     # Project documentation (you are reading this)
```

## 3. Environment Setup  
To run the chatbot demo, you need to install Python and the required libraries.  

**System Requirements:**  
- **OS:** Windows 10/11 (or macOS/Linux)  
- **Python:** 3.10.11  
- **Git:** Installed  
- **GPU (Optional):** NVIDIA GeForce RTX 3050 Ti Laptop GPU (or equivalent)  

**Note:** This demo uses `faiss-cpu` to optimize compatibility on Windows without requiring complex CUDA setup.  

**Setup Steps:**  

1. **Create and Activate Virtual Environment:**  
   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1   # For PowerShell
   # Or: venv\Scripts\activate.bat # For Command Prompt (CMD)
   # Or: source venv/bin/activate  # For Git Bash / Linux / macOS
   ```

2. **Install Python Libraries:**  
   ```powershell
   pip install -r backend/requirements.txt
   ```

3. **Configure Environment Variables:**  
   Create a `.env` file in the project root (`apec2027_chatbot/`) based on the provided `.env.example`. Fill in the required API keys:  
   ```ini
   # .env
   PINECONE_API_KEY=YOUR_PINECONE_API_KEY
   PINECONE_ENVIRONMENT=YOUR_PINECONE_ENVIRONMENT # e.g., us-east-1
   GEMINI_API_KEY_01=YOUR_GEMINI_API_KEY_1
   GEMINI_API_KEY_02=YOUR_GEMINI_API_KEY_2 # Optional, add for load balancing
   GEMINI_API_KEY_03=YOUR_GEMINI_API_KEY_3
   GEMINI_API_KEY_04=YOUR_GEMINI_API_KEY_4
   GEMINI_API_KEY_05=YOUR_GEMINI_API_KEY_5
   ```

   **Important Note:** `PINECONE_API_KEY`, `PINECONE_ENVIRONMENT`, and at least one `GEMINI_API_KEY` are mandatory. I designed the system to rotate multiple Gemini API keys to mitigate rate-limiting risks for embedding queries.

## 4. Running the Chatbot Demo  
To run the chatbot, you need to perform two main steps: prepare the data and start the backend/frontend services.  

### 4.1. Step 1: Data Preparation  
This is the process I designed to transform raw data into embeddings ready for RAG.  

1. **Crawl Data from APEC 2025 (`crawler.py`):**  
   ```powershell
   python backend/scripts/crawler.py
   ```  
   Uses `requests` and `BeautifulSoup` to crawl predefined URLs from APEC 2025. I developed separate functions for different page structures (text, tables, lists) to extract information efficiently. Raw data is saved as JSON in `backend/data/raw/`.  

2. **Build and Normalize Data Chunks (`chunk_data.py`):**  
   ```powershell
   python backend/scripts/chunk_data.py
   ```  
   Converts raw JSON data into small, contextually meaningful "chunks." I focused on combining related information (e.g., main content with contact details) into single chunks while separating independent items (e.g., event lists). The process includes deduplication and assigning unique IDs. Final chunks are saved in `backend/data/processed/`.  

3. **Create Embeddings and Vector Database (`rag_pipeline.py` & `pinecone_utils.py`):**  
   ```powershell
   python backend/scripts/rag_pipeline.py
   ```  
   - Generates vector embeddings from data chunks using `sentence-transformers`.  
   - Builds a local FAISS index for efficient context retrieval. FAISS index and metadata are saved in `backend/data/embeddings/`.  

   **Vector Database Choice:** I chose `faiss-cpu` for this demo to simplify deployment and ensure Windows compatibility. However, the architecture is designed to easily scale to Pinecone, a cloud-based Vector Database, for large-scale data and production environments.  

### 4.2. Step 2: Launch the Chatbot  
You need to open TWO separate terminal windows and ensure the virtual environment is activated in each.  

1. **Launch FastAPI Backend (`backend/api/app.py`):**  
   In the first terminal:  
   ```powershell
   (venv) PS C:\Users\nguye\OneDrive\Desktop\Hekate\Case_01\apec2027_chatbot> python backend\api\app.py
   ```  
   The API will run on `http://127.0.0.1:8000` or `http://0.0.0.0:8000`.  

2. **Launch Gradio UI (`demo/app.py`):**  
   In the second terminal:  
   ```powershell
   (venv) PS C:\Users\nguye\OneDrive\Desktop\Hekate\Case_01\apec2027_chatbot> python demo\app.py
   ```  
   The Gradio UI will launch and provide a local URL (typically `http://127.0.0.1:7860`). Open this URL in your browser.  

## 5. Interaction Examples & Solutions to Challenges  
I specifically focused on addressing two major challenges to enhance user experience:  

### 5.1. Issue A: Conversation Context Management (Chatbot Memory)  
**Challenge:** Basic RAG chatbots process each query independently, failing to "remember" prior chat context (e.g., "Can you repeat the answer?", "When will it be held?" requires recalling the subject).  

**My Solution:**  
I implemented a conversation history management mechanism for the current session:  
- **History Storage:** Each user_message - bot_response pair is stored.  
- **Context Summarization:** Before each new query, recent chat turns are summarized (or included verbatim if short) to create a concise context.  
- **Augment LLM Prompt:** The summarized history is added to the prompt sent to the LLM, alongside Vector DB context and the current question. This enables the LLM to understand and respond to related questions naturally and coherently.  

### 5.2. Issue B: Leveraging LLM Knowledge When Data Is Missing  
**Challenge:** When a user question lacks relevant information in the vector database (due to uncrawled data or mismatch), a basic RAG chatbot may respond with "no information found."  

**My Solution:**  
I implemented a smart "fall-back" mechanism leveraging LLM's background knowledge:  
- **Score Threshold:** After querying the Vector DB, I check the similarity score of retrieved chunks. If all chunks score too low (e.g., below 0.6 or 0.5), it indicates a lack of relevant information.  
- **Fall-back Proposal:** The chatbot doesn't respond immediately. Instead, it suggests: "I couldn't find precise information in my data. Would you like me to try answering based on my general knowledge?"  
- **Use LLM Knowledge:** If the user agrees, the chatbot sends the original question to the LLM without Vector DB context. The LLM then uses its inherent knowledge and reasoning to answer, including questions about Vietnamese culture and Phú Quốc (not in APEC 2025 data).  

**Example Questions to Try:**  
- What is today's event schedule?  
- Who are the APEC members?  
- Tell me about Gyeongju.  
- What are the entry procedures for South Korea?  
- How's the winter weather in South Korea?  
- Where is the Informal Senior Officials’ Meeting (ISOM) held?  
- Tell me about Jeju Island's nature.  
- What is the capital of Vietnam? *(Triggers fall-back, LLM uses general knowledge.)*  
- What’s special about Phú Quốc? *(Triggers fall-back, LLM uses general knowledge.)*  

## 6. Development Notes  
- **Diverse Data:** The ability to answer questions about Vietnam and Phú Quốc currently relies on LLM's background knowledge. For deeper responses, integrating specific Vietnam-related data sources in the future is necessary.  
- **LLM Expansion:** The demo currently uses `sentence-transformers` for embeddings and leverages LLM via API. Integrating larger LLMs (e.g., LLaMA 3 or Mistral) locally for generation would require more powerful GPU resources and complex setup.  

## 7. License  
<<<<<<< HEAD
This project is released under the MIT License.

---
license: MIT
language: en
tags:
  - gradio
  - chatbot
  - apec2027
---

