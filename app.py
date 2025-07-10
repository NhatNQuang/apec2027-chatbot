import os
import sys
import gradio as gr
import google.generativeai as genai
from dotenv import load_dotenv
import itertools
import functools
import re

# Thiết lập sys.path để Python tìm thấy gói 'backend'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# Import các hàm cần thiết từ pinecone_utils
from backend.utils.pinecone_utils import query_pinecone_index, initialize_pinecone_client

# Tải các biến môi trường từ file .env
dotenv_path = os.path.join(project_root, '.env')
load_dotenv(dotenv_path=dotenv_path)

# --- Cấu hình API Keys và LLM ---
PINECONE_INDEX_NAME = "apec2027-chatbot"

# --- HẰNG SỐ CẤU HÌNH LLM VÀ KHÁC ---
LLM_GENERATION_MODEL = 'gemini-2.0-flash'

CONFIDENCE_THRESHOLD = 0.5
AWAITING_GENERAL_KNOWLEDGE_CONFIRMATION_TAG = "[AWAITING_GK_CONFIRMATION]"
MAX_CONVERSATION_HISTORY_TOKENS = 700
MAX_CONVERSATION_TURNS_FOR_SUMMARIZATION = 5

# Hardcoded English prompt for GK confirmation as requested
ENGLISH_GK_CONFIRMATION_PROMPT = "I couldn't find this information in my documents. Would you like me to try to find out about it using my general knowledge? Please respond with 'yes' or 'no'."

# Lấy tất cả các Gemini API keys từ biến môi trường
GEMINI_API_KEYS = [
    os.getenv(f"GEMINI_API_KEY_0{i}") for i in range(1, 6)
    if os.getenv(f"GEMINI_API_KEY_0{i}")
]

if not GEMINI_API_KEYS:
    raise ValueError("No GEMINI_API_KEY found in .env file for LLM/Translation.")

gemini_llm_key_cycler = itertools.cycle(GEMINI_API_KEYS)

def get_gemini_llm_model(model_name=LLM_GENERATION_MODEL):
    current_llm_key = next(gemini_llm_key_cycler)
    genai.configure(api_key=current_llm_key)
    return genai.GenerativeModel(model_name)

initialize_pinecone_client()

# --- Hàm phát hiện ngôn ngữ của truy vấn ---
def detect_language(text: str) -> str:
    """
    Sử dụng LLM để phát hiện ngôn ngữ của văn bản.
    Trả về mã ngôn ngữ ISO 639-1 (ví dụ: 'en', 'vi', 'ko').
    """
    model = get_gemini_llm_model()
    prompt_detect_lang = f"""Detect the primary language of the following text and respond with ONLY its ISO 639-1 two-letter language code.
    For example: 'en' for English, 'vi' for Vietnamese, 'ko' for Korean, 'zh' for Chinese, 'fr' for French, 'de' for German, 'es' for Spanish.
    If the language cannot be confidently determined or is not one of the common languages, default to 'en'.

    Text: {text}

    Language Code:"""
    try:
        response = model.generate_content(prompt_detect_lang)
        lang_code = response.text.strip().lower()

        if len(lang_code) == 2 and lang_code.isalpha():
            return lang_code
        else:
            print(f"LLM returned invalid language code '{lang_code}' for detection. Defaulting to 'en'.")
            return 'en'
    except Exception as e:
        print(f"Error during LLM language detection: {e}. Defaulting to 'en'.")
        return 'en'

# --- Hàm dịch tự động thông báo lỗi ---
@functools.lru_cache(maxsize=128)
def translate_error_message(error_message_en: str, target_lang_code: str) -> str:
    if target_lang_code == 'en':
        return error_message_en

    model = get_gemini_llm_model()
    prompt_translate_error = f"""Translate the following error message into {target_lang_code} language.
    Respond with only the translated error message, without any additional commentary.

    [ERROR MESSAGE IN ENGLISH]
    {error_message_en}

    [TRANSLATED ERROR MESSAGE]
    """
    try:
        response = model.generate_content(prompt_translate_error)
        translated_message = response.text.strip()
        print(f"Translated error '{error_message_en}' to '{target_lang_code}': '{translated_message}'")
        return translated_message
    except Exception as e:
        print(f"Error translating error message '{error_message_en}' to '{target_lang_code}': {e}")
        return error_message_en

def get_localized_error_message(lang_code: str, error_type: str) -> str:
    error_messages_en = {
        'pinecone_query_error': "Sorry, I encountered an error while searching for relevant information in the database.",
        'no_relevant_info': "Sorry, I couldn't find any relevant information in my documents for this question.",
        'context_building_error': "Sorry, I couldn't form a valid context from the retrieved information.",
        'llm_generation_error': "Sorry, I encountered an error while generating the answer.",
        'query_preprocessing_error': "Sorry, I encountered an error while processing your question.",
        # 'no_relevant_info_and_suggest_gk' is now handled separately for English output
        'general_knowledge_declined': "Understood. I will stick to information from the provided documents. Is there anything else I can help you with?",
        'general_knowledge_fallback_error': "Sorry, I couldn't find the answer using my general knowledge either. Can I help you with anything else?",
        'repeat_no_history': "I'm sorry, I cannot repeat the answer as there is no previous conversation to refer to. What else can I help you with?"
    }

    english_error = error_messages_en.get(error_type, error_messages_en['llm_generation_error'])

    if lang_code != 'en':
        return translate_error_message(english_error, lang_code)
    else:
        return english_error

# --- Hàm tiền xử lý truy vấn: Sửa lỗi chính tả/ngữ pháp và Dịch sang tiếng Anh ---
def preprocess_query(query_text: str, original_lang_code: str) -> str:
    model = get_gemini_llm_model()

    prompt_fix_and_translate = f"""As a professional language assistant, your task is to review the user's question, correct any spelling or grammatical errors, improve the phrasing if it's unclear or awkward, and then translate the corrected and improved question into English.
    Respond with only the corrected and translated English question. Do not add any other content or commentary.

    [ORIGINAL QUESTION]
    {query_text}

    [CORRECTED AND TRANSLATED ENGLISH QUESTION]
    """

    try:
        response = model.generate_content(prompt_fix_and_translate)
        processed_query = response.text.strip()
        print(f"Original Query: '{query_text}'")
        print(f"Processed Query (Fixed & Translated to English): '{processed_query}'")
        return processed_query
    except Exception as e:
        print(f"Error during query preprocessing with Gemini: {e}")
        print("Falling back to original query for Pinecone search due to preprocessing error.")
        return query_text

# --- Hàm tóm tắt lịch sử hội thoại ---
def summarize_conversation_history(history_list: list) -> str:
    if not history_list:
        return ""

    recent_turns = history_list[-MAX_CONVERSATION_TURNS_FOR_SUMMARIZATION:]

    dialogue_str = ""
    for item in recent_turns:
        if isinstance(item, dict) and item.get('role') == 'user':
            dialogue_str += f"User: {item.get('content', '')}\n"
        elif isinstance(item, dict) and item.get('role') == 'assistant':
            # Remove the hidden tag for summarization
            clean_content = item.get('content', '').replace(AWAITING_GENERAL_KNOWLEDGE_CONFIRMATION_TAG, "").strip()
            dialogue_str += f"Assistant: {clean_content}\n"

    # Note: Token count estimation here is very rough. For production, use a proper tokenizer.
    if len(dialogue_str.split()) > MAX_CONVERSATION_HISTORY_TOKENS:
        model = get_gemini_llm_model()
        prompt_summarize = f"""Summarize the following conversation history concisely to extract key topics and context. This summary will be used to help an assistant understand the ongoing conversation and respond appropriately to the next user query.

        [CONVERSATION HISTORY]
        {dialogue_str}

        [CONCISE SUMMARY]
        """
        try:
            response = model.generate_content(prompt_summarize)
            summary = response.text.strip()
            print(f"Conversation history summarized: {summary[:150]}...")
            return f"\n[PREVIOUS CONVERSATION SUMMARY]\n{summary}\n"
        except Exception as e:
            print(f"Error summarizing history: {e}")
            return "\n[PREVIOUS CONVERSATION CONTEXT UNAVAILABLE]\n"
    else:
        return f"\n[PREVIOUS CONVERSATION]\n{dialogue_str}\n"

# --- Hàm cốt lõi của Chatbot RAG ---
def rag_chatbot(message: str, history: list):
    # Bước 1: Phát hiện ngôn ngữ của truy vấn gốc (luôn ở đầu hàm)
    original_lang_code = detect_language(message)
    print(f"Detected original language: {original_lang_code}")

    # --- Xử lý Vấn đề 01: Meta-query (nhắc lại câu trả lời) ---
    repeat_keywords = {
        'en': ['repeat', 'say again', 'what did you say', 'clarify', 'last answer'],
        'vi': ['nhắc lại', 'lặp lại', 'nói lại', 'câu trả lời trước'],
        'ko': ['다시 말해줘', '반복해줘', '뭐라고 했어']
    }
    user_message_lower = message.strip().lower()

    if original_lang_code in repeat_keywords and any(keyword in user_message_lower for keyword in repeat_keywords[original_lang_code]):
        if history and len(history) >= 2:
            last_bot_message_item = None
            for item in reversed(history):
                if isinstance(item, dict) and item.get('role') == 'assistant':
                    last_bot_message_item = item
                    break

            if last_bot_message_item and last_bot_message_item.get('content'):
                print("Responding to repeat query with previous answer.")
                clean_answer = last_bot_message_item['content'].replace(AWAITING_GENERAL_KNOWLEDGE_CONFIRMATION_TAG, "").strip()
                return clean_answer
            else:
                print("No previous bot response found to repeat.")
                return get_localized_error_message(original_lang_code, 'repeat_no_history')
        else:
            print("No previous conversation to repeat.")
            return get_localized_error_message(original_lang_code, 'repeat_no_history')

    # --- Xử lý Vấn đề 02: Trạng thái chờ xác nhận kiến thức chung ---
    last_bot_response_content = ""
    original_query_for_gk = message # Default to current message

    if history:
        for item in reversed(history):
            if isinstance(item, dict) and item.get('role') == 'assistant':
                last_bot_response_content = item.get('content', '')
                break

        found_bot_response_idx = -1
        # Find index of the last bot response
        for idx, item in enumerate(reversed(history)):
            if isinstance(item, dict) and item.get('role') == 'assistant' and item.get('content') == last_bot_response_content:
                found_bot_response_idx = len(history) - 1 - idx
                break

        # If the last bot response was the GK prompt, use the user's *previous* query
        if found_bot_response_idx > 0 and AWAITING_GENERAL_KNOWLEDGE_CONFIRMATION_TAG in last_bot_response_content:
            user_msg_item = history[found_bot_response_idx - 1]
            if isinstance(user_msg_item, dict) and user_msg_item.get('role') == 'user':
                original_query_for_gk = user_msg_item.get('content', original_query_for_gk)


    is_awaiting_gk_confirmation = AWAITING_GENERAL_KNOWLEDGE_CONFIRMATION_TAG in last_bot_response_content

    if is_awaiting_gk_confirmation:
        # Check user's response to the general knowledge suggestion
        user_message_lower = message.strip().lower()
        if user_message_lower in ["yes", "vâng", "có", "ok", "chấp nhận", "đồng ý"]: # Allow some common non-English affirmatives too for robustness
            print("User accepted general knowledge fallback. Generating answer from LLM's general knowledge.")

            model = get_gemini_llm_model(model_name=LLM_GENERATION_MODEL)
            prompt_gk = f"""You are an intelligent assistant. Answer the following question using your general knowledge.
            Answer in the language of the ORIGINAL USER QUESTION, which was: '{original_lang_code}'.
            IMPORTANT: ALWAYS RETAIN PLACE NAMES, EVENT NAMES, ORGANIZATION NAMES, TIMES, DATES, PHONE NUMBERS, WEBSITES, and SPECIALIZED TERMS in English in the answer.

            [ORIGINAL USER QUESTION]
            {original_query_for_gk}

            [ANSWER]
            """
            try:
                response = model.generate_content(prompt_gk)
                final_answer = response.text if response.candidates and response.candidates[0].content.parts else ""
                if not final_answer:
                    return get_localized_error_message(original_lang_code, 'llm_generation_error')
                return final_answer
            except Exception as e:
                print(f"Error generating GK content: {e}")
                return get_localized_error_message(original_lang_code, 'general_knowledge_fallback_error')
        elif user_message_lower in ["no", "không", "ko", "từ chối"]: # Allow some common non-English negatives
            print("User declined general knowledge fallback.")
            return get_localized_error_message(original_lang_code, 'general_knowledge_declined')
        else:
            print("Unclear response to GK fallback, re-prompting.")
            # Keep the GK prompt in English as requested
            return f"{ENGLISH_GK_CONFIRMATION_PROMPT} {AWAITING_GENERAL_KNOWLEDGE_CONFIRMATION_TAG}"


    # Bắt đầu luồng RAG tiêu chuẩn

    conversation_history_context = summarize_conversation_history(history)

    processed_query_for_pinecone = preprocess_query(message, original_lang_code)
    if not processed_query_for_pinecone:
        return get_localized_error_message(original_lang_code, 'query_preprocessing_error')

    try:
        retrieved_chunks = query_pinecone_index(PINECONE_INDEX_NAME, processed_query_for_pinecone, top_k=3)
        print(f"Retrieved {len(retrieved_chunks)} chunks for processed query: '{processed_query_for_pinecone}'")
    except Exception as e:
        print(f"Error querying Pinecone: {e}")
        return get_localized_error_message(original_lang_code, 'pinecone_query_error')

    avg_score = sum([c['score'] for c in retrieved_chunks]) / len(retrieved_chunks) if retrieved_chunks else 0

    if avg_score < CONFIDENCE_THRESHOLD or not retrieved_chunks:
        print(f"Low confidence (avg_score={avg_score:.2f}) or no chunks retrieved. Suggesting general knowledge fallback.")
        # Directly use the English GK confirmation prompt
        return f"{ENGLISH_GK_CONFIRMATION_PROMPT} {AWAITING_GENERAL_KNOWLEDGE_CONFIRMATION_TAG}"


    context_texts = [chunk["content"] for chunk in retrieved_chunks]
    context_texts = [text for text in context_texts if text != 'N/A' and text.strip()]
    context_str = "\n\n---\n\n".join(context_texts)

    if not context_str.strip():
        return get_localized_error_message(original_lang_code, 'context_building_error')

    model = get_gemini_llm_model(model_name=LLM_GENERATION_MODEL)

    prompt = f"""You are an intelligent assistant specialized in APEC 2025 information.

    {conversation_history_context}

    You must answer the user's question accurately and completely BASED ON the context provided below.
    If the context does not contain enough information to answer, state that you do not know that information.
    Do not fabricate information.

    Answer in the language of the ORIGINAL USER QUESTION.
    For example: If the ORIGINAL USER QUESTION is in Vietnamese, answer in Vietnamese. If it is in English, answer in English. If it is in Korean, answer in Korean.

    IMPORTANT: ALWAYS RETAIN PLACE NAMES, EVENT NAMES, ORGANIZATION NAMES, TIMES, DATES, PHONE NUMBERS, WEBSITES, and SPECIALIZED TERMS in English in the answer.
    Example: "The APEC Economic Leaders’ Meeting will take place in Gyeongju."

    FORMATTING GUIDELINES:
    - For answers containing lists of items (e.g., members, events, detailed information), use bullet points or numbered lists.
    - Bold important keywords, names, dates, and locations using Markdown (e.g., **Example Text**).
    - Ensure the answer is well-structured, easy to read, and uses line breaks appropriately for clarity.
    - If the answer has multiple distinct parts, use subheadings or clear paragraph breaks.

    [CONTEXT]
    {context_str}

    [ORIGINAL USER QUESTION]
    {message}

    [ANSWER]
    """
    print("--- LLM Prompt (Final Generation) ---")
    print(prompt)
    print("-------------------------------------")

    try:
        response = model.generate_content(prompt)
        final_answer = response.text if response.candidates and response.candidates[0].content.parts else ""
        if not final_answer:
            raise ValueError("LLM response was empty or blocked.")
        return final_answer
    except Exception as e:
        print(f"Error generating content with LLM: {e}")
        return get_localized_error_message(original_lang_code, 'llm_generation_error')

# --- Thiết lập Gradio Interface ---
if __name__ == "__main__":
    print("Starting Gradio Chatbot APEC 2025 RAG...")

    demo = gr.ChatInterface(
        fn=rag_chatbot,
        chatbot=gr.Chatbot(height=400, type="messages"),
        textbox=gr.Textbox(placeholder="Hỏi tôi về APEC 2025...", container=False, scale=7),
        title="Chatbot APEC 2025 RAG",
        description="Hãy hỏi tôi bất kỳ điều gì về APEC 2025 dựa trên các tài liệu đã cung cấp. Tôi sẽ cố gắng sửa lỗi chính tả, dịch câu hỏi, và trả lời bằng ngôn ngữ bạn đã hỏi (giữ nguyên tên riêng tiếng Anh).",
        examples=[
            "APEC là gì?",
            "Các thành viên của APEC là những quốc gia nào?",
            "APEC 2025 KOREA có chủ đề và ưu tiên gì?",
            "Thời tiết ở Hàn Quốc vào tháng 7 như thế nào?",
            "Hãy cho tôi biết về Gyeongju East Palace Garden và số điện thoại liên hệ của nó?",
            "Các sự kiện chính của APEC 2025 Korea là gì và diễn ra ở đâu?",
            "APEC 한국의 주요 회의는 무엇입니까?",
            "Bạn có thể nhắc lại câu trả lời được không?"
        ],
        # Changed theme to a brighter one
        theme=gr.themes.Base() # Using Base for a generally brighter and cleaner look
    )

    demo.launch(share=True)