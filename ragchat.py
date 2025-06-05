import streamlit as st
import requests
import json
import time
import os
import tempfile
import re
from bs4 import BeautifulSoup
from docling.document_converter import DocumentConverter
from sentence_transformers import SentenceTransformer
import faiss
import warnings
import logging

# --- Suppress Warnings ---
warnings.filterwarnings("ignore")
logging.getLogger('torch').setLevel(logging.ERROR)
logging.getLogger('streamlit').setLevel(logging.ERROR)

# --- Constants ---
OLLAMA_API = "http://localhost:11434"
LOG_FILE = "chat_log.json"
embedder = SentenceTransformer('all-MiniLM-L6-v2')
st.set_page_config(page_title="AskDoc", page_icon="📄", layout="wide")

# --- Custom UI CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

:root {
    --primary: #4f46e5;
    --primary-hover: #4338ca;
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-tertiary: #334155;
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --user-msg: #4f46e5;
    --assistant-msg: #334155;
    --border-radius: 12px;
    --spacing-xs: 4px;
    --spacing-sm: 8px;
    --spacing-md: 16px;
    --spacing-lg: 24px;
    --spacing-xl: 32px;
}

* {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
}

html, body, [class*="st"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
}

body {
    background-color: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.6;
}

/* Header */
#sticky-title {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    width: 100%;
    background: rgba(15, 23, 42, 0.9);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    font-size: 1.25rem;
    font-weight: 600;
    color: var(--text-primary);
    padding: var(--spacing-md) var(--spacing-xl);
    z-index: 1000;
    border-bottom: 1px solid rgba(255, 255, 255, 0.1);
    display: flex;
    align-items: center;
    justify-content: space-between;
}

#sticky-title::before {
    content: '💬';
    margin-right: var(--spacing-sm);
}

/* Hide default Streamlit header */
[data-testid="stHeader"] {
    display: none !important;
}

/* Chat container */
.stChatFloatingInputContainer {
    background: var(--bg-secondary);
    padding: var(--spacing-md) var(--spacing-xl);
    border-top: 1px solid rgba(255, 255, 255, 0.1);
}

/* Message bubbles */
.stChatMessage {
    max-width: 85%;
    margin-bottom: var(--spacing-md);
    padding: 0;
    border: none;
    background: transparent;
    display: flex;
    flex-direction: column;
    gap: var(--spacing-xs);
}

.stChatMessage.user {
    margin-left: auto;
    align-items: flex-end;
}

.stChatMessage.assistant {
    margin-right: auto;
    align-items: flex-start;
}

/* Message content */
.stChatMessage .message-content {
    padding: var(--spacing-md) var(--spacing-lg);
    border-radius: var(--border-radius);
    font-size: 0.95rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    position: relative;
    word-break: break-word;
}

.stChatMessage.user .message-content {
    background: var(--user-msg);
    color: white;
    border-bottom-right-radius: 4px;
}

.stChatMessage.assistant .message-content {
    background: var(--assistant-msg);
    color: var(--text-primary);
    border-bottom-left-radius: 4px;
}

/* Message metadata */
.message-meta {
    font-size: 0.75rem;
    color: var(--text-secondary);
    margin-top: var(--spacing-xs);
    display: flex;
    align-items: center;
    gap: var(--spacing-sm);
}

/* Input area */
.stTextInput>div>div>input,
.stTextArea>div>textarea {
    background: var(--bg-tertiary) !important;
    color: var(--text-primary) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    border-radius: var(--border-radius) !important;
    padding: var(--spacing-md) var(--spacing-lg) !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
}

.stTextInput>div>div>input:focus,
.stTextArea>div>textarea:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.2) !important;
    outline: none !important;
}

/* Buttons */
.stButton>button {
    background: var(--primary) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--border-radius) !important;
    padding: var(--spacing-sm) var(--spacing-lg) !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
    text-transform: none !important;
    font-size: 0.9rem !important;
}

.stButton>button:hover {
    background: var(--primary-hover) !important;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2);
}

/* Sidebar */
.stSidebar {
    background: var(--bg-secondary) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
}

.stSidebar .stMarkdown h1,
.stSidebar .stMarkdown h2,
.stSidebar .stMarkdown h3 {
    color: var(--text-primary) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: var(--bg-secondary);
}

::-webkit-scrollbar-thumb {
    background: var(--bg-tertiary);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: #4b5563;
}

/* Responsive adjustments */
@media (max-width: 768px) {
    .stChatMessage {
        max-width: 90%;
    }
    
    #sticky-title {
        padding: var(--spacing-sm) var(--spacing-md);
        font-size: 1.1rem;
    }
}

/* Animation for new messages */
@keyframes messageAppear {
    from {
        opacity: 0;
        transform: translateY(10px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.stChatMessage {
    animation: messageAppear 0.3s ease-out forwards;
}

/* Improve code blocks */
pre {
    background: #1e1e1e !important;
    border-radius: 6px !important;
    padding: 1rem !important;
    overflow-x: auto;
}

code {
    font-family: 'Fira Code', 'Cascadia Code', 'Consolas', monospace !important;
}

/* Loading animation */
@keyframes pulse {
    0%, 100% { opacity: 0.6; }
    50% { opacity: 1; }
}

.typing-indicator {
    display: flex;
    gap: 4px;
    padding: 12px 16px;
    background: var(--assistant-msg);
    border-radius: var(--border-radius);
    width: fit-content;
    margin-bottom: var(--spacing-md);
}

.typing-dot {
    width: 8px;
    height: 8px;
    background: var(--text-secondary);
    border-radius: 50%;
    animation: pulse 1.4s infinite ease-in-out;
}

.typing-dot:nth-child(1) { animation-delay: 0s; }
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
    color: #fff;
    border-radius: 6px;
    border: 1px solid #444;
}

.stProgress>div>div {
    background-color: #10a37f;
}

#chat-input {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: #1e1e1e;
    padding: 12px 24px;
    border-top: 1px solid #333;
    z-index: 999;
}

@keyframes blink {
    50% { opacity: 0.3; }
}
</style>
<div id="sticky-title">AskDoc</div>
<div style="height: 65px;"></div>
""", unsafe_allow_html=True)

# --- Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []

# --- Sidebar ---
st.sidebar.title("⚙️ Settings & Tools")

@st.cache_data(show_spinner=False)
def get_available_models():
    try:
        response = requests.get(f"{OLLAMA_API}/api/tags", timeout=5)
        return [model["name"] for model in response.json().get("models", [])]
    except:
        return ["gemma3:1b"]

available_models = get_available_models()
selected_model = st.sidebar.selectbox("Select Model", available_models)

uploaded_files = st.sidebar.file_uploader("📂 Upload Files", type=["pdf", "docx", "txt", "html", "md", "csv"], accept_multiple_files=True)
url_input = st.sidebar.text_input("🌐 Scrape a URL")

if st.sidebar.button("🩹 Clear Chat"):
    st.session_state.messages = []
    with open(LOG_FILE, "w") as f:
        f.write("[]")

# --- File/URL Processing ---
@st.cache_data(show_spinner=False)
def cached_process_file(uploaded_file):
    suffix = "." + uploaded_file.name.split(".")[-1] if "." in uploaded_file.name else ""

    if suffix == ".txt":
        content = uploaded_file.read().decode("utf-8", errors="ignore")
        return content  # ✅ Don't involve DocumentConverter at all

    # Reset file pointer to start before reading again
    uploaded_file.seek(0)
    converter = DocumentConverter()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.read())
        path = temp_file.name
    result = converter.convert(path)
    os.unlink(path)
    return result.document.export_to_markdown()



@st.cache_data(show_spinner=False)
def cached_scrape_url(url):
    response = requests.get(url, timeout=10)
    soup = BeautifulSoup(response.content, "html.parser")
    return soup.get_text(separator=" ", strip=True)

@st.cache_resource
def build_faiss_index_cached(text):
    chunks = [text[i:i + 500] for i in range(0, len(text), 450)]
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, chunks

# --- Load Knowledgebase ---
if uploaded_files or url_input:
    corpus = []
    progress_bar = st.progress(0)
    total = len(uploaded_files) + (1 if url_input else 0)
    done = 0

    for file in uploaded_files:
        try:
            corpus.append(cached_process_file(file))
        except Exception as e:
            st.error(f"Failed: {file.name} — {str(e)}")
        done += 1
        progress_bar.progress(done / total)

    if url_input:
        try:
            corpus.append(cached_scrape_url(url_input))
        except Exception as e:
            st.error(f"Failed to scrape URL: {str(e)}")
        done += 1
        progress_bar.progress(done / total)

    if corpus:
        with st.spinner("Building index..."):
            st.session_state.index, st.session_state.chunks = build_faiss_index_cached("\n".join(corpus))
        st.success("Knowledge base indexed!")
    progress_bar.empty()

# --- View Knowledgebase ---
if st.session_state.chunks:
    with st.expander("📚 View Knowledgebase"):
        for i, chunk in enumerate(st.session_state.chunks):
            st.markdown(f"**Chunk {i+1}:**")
            st.code(chunk[:800] + ("..." if len(chunk) > 800 else ""), language="markdown")

# --- Display Chat ---
if not st.session_state.messages:
    st.markdown("### 💬 Let's chat...")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        content = message["content"]
        if isinstance(content, dict) and content.get("type") == "error":
            st.error(content["content"])
        else:
            st.markdown(content if isinstance(content, str) else content["content"])

# --- API Call ---
def query_ollama_stream(prompt, model):
    payload = {"model": model, "prompt": prompt, "stream": True}
    try:
        with requests.post(f"{OLLAMA_API}/api/generate", json=payload, stream=True, timeout=30) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if line:
                    part = json.loads(line.decode('utf-8'))
                    yield part.get('response', '')
    except requests.RequestException as e:
        yield f"Error: Could not connect to Ollama API - {str(e)}"

# --- Display Chat Messages ---
# Display existing messages
for message in st.session_state.messages:
    role = message["role"]
    content = message["content"]
    
    with st.chat_message(role):
        if role == "user":
            st.markdown(f'<div class="message-content">{content}</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="message-meta">{time.strftime("%I:%M %p")} • You</div>',
                unsafe_allow_html=True
            )
        else:  # assistant
            if isinstance(content, dict) and content.get("type") == "answer":
                st.markdown(f'<div class="message-content">{content["content"]}</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="message-meta">{time.strftime("%I:%M %p")} • Assistant</div>',
                    unsafe_allow_html=True
                )

# --- Input Box ---
prompt = st.chat_input("Type your message...")
if prompt:
    # Add user message to chat
    timestamp = time.time()
    st.session_state.messages.append({
        "role": "user",
        "content": prompt,
        "timestamp": timestamp
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(f'<div class="message-content">{prompt}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="message-meta">{time.strftime("%I:%M %p", time.localtime(timestamp))} • You</div>',
            unsafe_allow_html=True
        )

    # Prepare context
    context = ""
    if st.session_state.index:
        embedding = embedder.encode([prompt])
        D, I = st.session_state.index.search(embedding, k=5)
        context = "\n\n".join([st.session_state.chunks[i] for i in I[0]])

    full_prompt = f"""You are a helpful assistant. Use the following context if relevant to answer the question. If the context doesn't help, use your own knowledge.\n\nContext:\n{context}\n\nQuestion: {prompt}\n\nAnswer:"""

    # Show typing indicator
    with st.chat_message("assistant"):
        typing_placeholder = st.empty()
        typing_placeholder.markdown(
            '<div class="typing-indicator">'
            '<div class="typing-dot"></div>'
            '<div class="typing-dot"></div>'
            '<div class="typing-dot"></div>'
            '</div>',
            unsafe_allow_html=True
        )
        
        # Stream the response
        response_content = ""
        response_box = st.empty()
        
        for token in query_ollama_stream(full_prompt, selected_model):
            # Skip any think tags for now
            if token in ["<think>", "</think>"]:
                continue
                
            response_content += token
            response_box.markdown(
                f'<div class="message-content">{response_content}</div>',
                unsafe_allow_html=True
            )
            time.sleep(0.02)  # Small delay for better streaming effect
        
        # Remove typing indicator and show final response
        typing_placeholder.empty()
        response_box.markdown(
            f'<div class="message-content">{response_content}</div>',
            unsafe_allow_html=True
        )
        
        # Add timestamp
        st.markdown(
            f'<div class="message-meta">{time.strftime("%I:%M %p", time.localtime())} • Assistant</div>',
            unsafe_allow_html=True
        )
    
    # Save assistant's response to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": {
            "type": "answer",
            "content": response_content
        },
        "timestamp": time.time()
    })
    
    # Save to log file
    with open(LOG_FILE, "w") as f:
        json.dump([{"role": msg["role"], "content": msg["content"]} for msg in st.session_state.messages], f, indent=2)
    
    # Rerun to update the UI with the new messages
    st.rerun()
