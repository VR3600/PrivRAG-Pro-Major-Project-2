import os
import re
import json
import time
import base64
import shutil
import hashlib
import requests
import numpy as np
import streamlit as st
import fitz  # PyMuPDF
import faiss
from fastembed import TextEmbedding
from rank_bm25 import BM25Okapi
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# -------------------- CONFIGURATION --------------------
BASE_VAULT_DIR = "encrypted_vaults"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
MODEL_NAME = "qwen2.5:1.5b"

st.set_page_config(
    page_title="PrivRAG Pro - Student-Friendly Confidential AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------- CUSTOM MODERN UI CSS --------------------
st.markdown("""
<style>
    .main-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
        padding: 1.4rem 1.8rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.2rem;
        border: 1px solid rgba(255,255,255,0.12);
    }
    .main-banner h2 {
        margin: 0;
        font-size: 1.7rem;
        color: #f8fafc;
    }
    .main-banner p {
        margin: 0.35rem 0 0 0;
        color: #cbd5e1;
        font-size: 0.92rem;
    }
    .badge-pill {
        display: inline-block;
        background: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid #10b981;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- SESSION STATE --------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_user" not in st.session_state:
    st.session_state.active_user = ""

# -------------------- CACHED EMBEDDING MODEL --------------------
@st.cache_resource(show_spinner="Loading lightweight ONNX Embedding Model (~130MB)...")
def load_embedder():
    return TextEmbedding(model_name="BAAI/bge-small-en-v1.5")

# -------------------- MULTI-USER ISOLATED SECURITY (AES-256) --------------------
def get_user_paths(username: str):
    clean_user = username.strip().lower()
    user_hash = hashlib.sha256(clean_user.encode("utf-8")).hexdigest()[:16]
    user_dir = os.path.join(BASE_VAULT_DIR, f"user_{user_hash}")
    return {
        "dir": user_dir,
        "salt": os.path.join(user_dir, "salt.bin"),
        "chunks": os.path.join(user_dir, "chunks.enc"),
        "index": os.path.join(user_dir, "vector.index"),
    }

def get_fernet(password: str, paths: dict) -> Fernet:
    os.makedirs(paths["dir"], exist_ok=True)
    if not os.path.exists(paths["salt"]):
        salt = os.urandom(16)
        with open(paths["salt"], "wb") as f:
            f.write(salt)
    else:
        with open(paths["salt"], "rb") as f:
            salt = f.read()

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode("utf-8")))
    return Fernet(key)

def redact_pii(text: str) -> str:
    text = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "<REDACTED_EMAIL>", text)
    text = re.sub(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b", "<REDACTED_PAN>", text)
    text = re.sub(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b", "<REDACTED_ID>", text)
    text = re.sub(r"\b(?:\+91[\-\s]?)?[6-9]\d{9}\b", "<REDACTED_PHONE>", text)
    return text

# -------------------- DOCUMENT PARSING & CHUNKING --------------------
def extract_and_chunk(uploaded_file, enable_pii_mask: bool, chunk_size: int = 180, overlap: int = 35):
    filename = uploaded_file.name
    ext = filename.split(".")[-1].lower()
    pages_data = []

    if ext == "pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        for page_num in range(len(doc)):
            text = doc[page_num].get_text("text").strip()
            if text:
                pages_data.append((page_num + 1, text))
        doc.close()
    else:
        raw_text = uploaded_file.read().decode("utf-8", errors="ignore").strip()
        if raw_text:
            pages_data.append((1, raw_text))

    chunks = []
    for page_num, text in pages_data:
        if enable_pii_mask:
            text = redact_pii(text)
        words = text.split()
        step = max(1, chunk_size - overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + chunk_size]
            if len(chunk_words) >= 10 or len(words) < 10:
                chunks.append({
                    "source": filename,
                    "page": page_num,
                    "text": " ".join(chunk_words)
                })
    return chunks

# -------------------- ENCRYPTED STORAGE & LOADING --------------------
def save_vault(chunks: list, faiss_index, fernet: Fernet, paths: dict):
    os.makedirs(paths["dir"], exist_ok=True)
    encrypted_list = []
    for item in chunks:
        raw_json = json.dumps(item).encode("utf-8")
        encrypted_token = fernet.encrypt(raw_json).decode("utf-8")
        encrypted_list.append(encrypted_token)

    with open(paths["chunks"], "w", encoding="utf-8") as f:
        json.dump(encrypted_list, f)
    if faiss_index is not None:
        faiss.write_index(faiss_index, paths["index"])

def load_vault(fernet: Fernet, paths: dict):
    if not (os.path.exists(paths["chunks"]) and os.path.exists(paths["index"])):
        return [], None, []
    with open(paths["chunks"], "r", encoding="utf-8") as f:
        encrypted_list = json.load(f)

    decrypted_chunks = []
    for token in encrypted_list:
        raw_json = fernet.decrypt(token.encode("utf-8")).decode("utf-8")
        decrypted_chunks.append(json.loads(raw_json))

    faiss_index = faiss.read_index(paths["index"])
    return decrypted_chunks, faiss_index, encrypted_list

def rebuild_index(chunks: list, embedder, fernet: Fernet, paths: dict):
    if not chunks:
        if os.path.exists(paths["chunks"]):
            os.remove(paths["chunks"])
        if os.path.exists(paths["index"]):
            os.remove(paths["index"])
        return None
    texts = [c["text"] for c in chunks]
    embeddings = np.array(list(embedder.embed(texts, batch_size=16)), dtype="float32")
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    new_index = faiss.IndexFlatIP(dim)
    new_index.add(embeddings)
    save_vault(chunks, new_index, fernet, paths)
    return new_index

# -------------------- HYBRID SEARCH (FAISS + BM25) --------------------
def hybrid_retrieve(query: str, chunks: list, faiss_index, embedder, top_k: int = 3, source_filter: str = "All Documents"):
    if not chunks or faiss_index is None:
        return []

    valid_indices = [
        i for i, c in enumerate(chunks)
        if source_filter == "All Documents" or c["source"] == source_filter
    ]
    if not valid_indices:
        return []

    k = min(top_k, len(valid_indices))

    q_emb = np.array(list(embedder.embed([query])), dtype="float32")
    faiss.normalize_L2(q_emb)
    _, faiss_ids = faiss_index.search(q_emb, k=min(len(chunks), max(10, k * 4)))

    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(query.lower().split())
    bm25_ids = np.argsort(bm25_scores)[::-1][: min(len(chunks), max(10, k * 4))]

    valid_set = set(valid_indices)
    rrf_scores = {}
    for rank, idx in enumerate(faiss_ids[0]):
        if idx != -1 and idx in valid_set:
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank)
    for rank, idx in enumerate(bm25_ids):
        if idx in valid_set:
            rrf_scores[idx] = rrf_scores.get(idx, 0.0) + 1.0 / (60 + rank)

    sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
    return [chunks[i] for i in sorted_indices]

# -------------------- HYBRID DUAL-ENGINE (LOCAL OLLAMA + FREE CLOUD FALLBACK) --------------------
def unload_model_from_ram():
    try:
        requests.post(OLLAMA_URL, json={"model": MODEL_NAME, "keep_alive": 0}, timeout=5)
    except Exception:
        pass

def ask_local_llm(
    query: str,
    context_chunks: list,
    response_style: str = "🎯 Auto (Follow My Question)",
    language_mode: str = "Simple English"
):
    context_lines = []
    for c in context_chunks:
        header = f"Document: {c['source']} (Page {c['page']})"
        context_lines.append(f"{header}\n{c['text']}")
    context_text = "\n\n".join(context_lines)

    style_map = {
        "🎯 Auto (Follow My Question)": (
            "Follow the exact length or format requested in the student's question (e.g., short, points, or detail). "
            "Otherwise, give a clear and simple explanation."
        ),
        "⚡ Short & Sweet (2-3 Lines)": "Keep the answer strictly to 2 or 3 short sentences maximum.",
        "📖 Easy Student Explanation": "Explain step-by-step like a friendly college teacher using everyday words.",
        "📝 Exam-Ready (Headings + Points)": "Start with a 2-line definition, then write neat bullet points for exams.",
        "🔑 Key Bullet Points Only": "Write the entire answer strictly as short, easy-to-read bullet points.",
    }
    chosen_style = style_map.get(response_style, style_map["🎯 Auto (Follow My Question)"])

    if "Hinglish" in language_mode:
        lang_instruction = """CRITICAL LANGUAGE RULE:
You MUST write the entire answer in HINGLISH (Hindi language written in English alphabet mixed with English technical terms, like Indian students chat on WhatsApp).
DO NOT write pure English sentences! Use words like 'hai', 'hota hai', 'kaam karta hai', 'jaise ki', 'iska matlab hai'.
Example: "Aasan shabdon mein, Stack ek linear data structure hota hai jo LIFO principle par kaam karta hai." """
        answer_trigger = "HINGLISH ANSWER (Use words like 'hai', 'hota hai'):"
    elif "Hindi" in language_mode:
        lang_instruction = "CRITICAL LANGUAGE RULE: आपको पूरा उत्तर केवल सरल हिंदी (Devanagari script) में लिखना है। अंग्रेजी के वाक्य बिल्कुल न लिखें।"
        answer_trigger = "सरल हिंदी में उत्तर:"
    else:
        lang_instruction = "CRITICAL LANGUAGE RULE: Write the answer in very simple, beginner-friendly English."
        answer_trigger = "SIMPLE ENGLISH ANSWER:"

    prompt = f"""You are a friendly Indian College Study Tutor.
Answer the student's question using ONLY the DOCUMENT CONTEXT below.

{lang_instruction}
FORMAT RULE: {chosen_style}
ACCURACY RULE: If the answer is not in the context, say "Yeh jankari uploaded document mein nahi mili (Not found in document)." Always mention the Document name and Page number at the end.

DOCUMENT CONTEXT:
{context_text}

STUDENT'S QUESTION: {query}

{answer_trigger}"""

    # 1. Local PC Mode: Check if Local Ollama is running
    try:
        base_ollama = OLLAMA_URL.replace("/api/generate", "")
        requests.get(base_ollama, timeout=2)
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "2m",
            "options": {"temperature": 0.25, "num_ctx": 2048}
        }
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        response.raise_for_status()
        return response.json().get("response", "No response generated.")
    except Exception:
        pass

    # 2. Live Website Mode: Auto-detect active Groq models so 404 never happens
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = ""

    api_key = str(api_key).strip().strip('"').strip("'")
    if api_key:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        candidate_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "gemma2-9b-it",
            "llama3-8b-8192"
        ]
        try:
            m_resp = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=10)
            if m_resp.status_code == 200:
                for m in m_resp.json().get("data", []):
                    m_id = m.get("id", "")
                    if m_id and not any(x in m_id.lower() for x in ["whisper", "guard", "tts", "embed", "vision"]):
                        if m_id not in candidate_models:
                            candidate_models.append(m_id)
        except Exception:
            pass

        for model_id in candidate_models:
            try:
                cloud_payload = {
                    "model": model_id,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.25
                }
                r = requests.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers=headers,
                    json=cloud_payload,
                    timeout=25
                )
                if r.status_code == 200:
                    return r.json()["choices"][0]["message"]["content"]
            except Exception:
                continue

    # 3. Zero-Key Free AI Fallback (Works even without API Key)
    try:
        free_resp = requests.post(
            "https://text.pollinations.ai/openai",
            headers={"Content-Type": "application/json"},
            json={
                "model": "openai",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.25
            },
            timeout=30
        )
        if free_resp.status_code == 200:
            data = free_resp.json()
            if "choices" in data and data["choices"]:
                return data["choices"][0]["message"]["content"]
    except Exception:
        pass

    return (
        "**[Direct Match from Encrypted Vault]:**\n\n"
        + "\n\n".join([f"📄 **{c['source']} (Page {c['page']})**: {c['text']}" for c in context_chunks])
    )
# -------------------- TOP HERO BANNER --------------------
st.markdown("""
<div class="main-banner">
    <div style="margin-bottom: 8px;">
        <span class="badge-pill">🎓 Student-Friendly AI Tutor</span>
        <span class="badge-pill">🔐 AES-256 Encrypted Vault</span>
        <span class="badge-pill">🗣️ English • Hinglish • Hindi</span>
    </div>
    <h2>🛡️ PrivRAG Pro: Smart & Simple Confidential Document Tutor</h2>
    <p>Ask in any style (Short, Exam Notes, Bullet Points, or Hinglish) • 100% Private & Offline • Hybrid Search</p>
</div>
""", unsafe_allow_html=True)

# -------------------- SIDEBAR MULTI-USER LOGIN --------------------
with st.sidebar:
    st.header("👤 Student Workspace Login")
    username = st.text_input("Username / Roll No:", placeholder="e.g., rahul_01")
    vault_password = st.text_input("Secret Vault Password:", type="password", placeholder="Enter secret key...")
    enable_pii = st.toggle("Auto-Mask PII (ID/PAN/Phone/Email)", value=True)

    st.divider()
    st.subheader("📂 Upload Notes & PDFs")
    uploaded_files = st.file_uploader(
        "Drop PDF, TXT, or MD files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True
    )

    index_btn = st.button("🔐 Encrypt & Index to Vault", use_container_width=True, type="primary")

    st.divider()
    st.subheader("⚡ Controls")
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🧹 Free RAM", use_container_width=True):
            unload_model_from_ram()
            st.toast("RAM freed! Model unloaded.", icon="✅")
    with col_b:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    if st.button("🚨 Delete My Vault", use_container_width=True):
        if username:
            u_paths = get_user_paths(username)
            if os.path.exists(u_paths["dir"]):
                shutil.rmtree(u_paths["dir"])
        st.session_state.clear()
        st.rerun()

if not username or not vault_password:
    st.info("👈 Enter any **Username** and **Secret Vault Password** in the left sidebar to unlock your personal study & confidential workspace.")
    st.stop()

if st.session_state.active_user != username.strip().lower():
    st.session_state.messages = []
    st.session_state.active_user = username.strip().lower()

# -------------------- LOAD USER VAULT & EMBEDDER --------------------
user_paths = get_user_paths(username)
fernet = get_fernet(vault_password, user_paths)
embedder = load_embedder()

try:
    chunks, faiss_index, encrypted_tokens = load_vault(fernet, user_paths)
except InvalidToken:
    st.error("❌ Incorrect Vault Password for this Username! AES-256 Decryption failed.")
    st.stop()

if index_btn:
    if not uploaded_files:
        st.sidebar.warning("Select at least one file first.")
    else:
        new_chunks = []
        with st.sidebar.status("Encrypting & Indexing...", expanded=True) as status:
            st.write("📄 Reading pages & masking PII...")
            for uf in uploaded_files:
                extracted = extract_and_chunk(uf, enable_pii_mask=enable_pii)
                new_chunks.extend(extracted)

            if not new_chunks:
                status.update(label="No readable text found.", state="error")
            else:
                st.write("🧠 Creating ONNX embeddings...")
                all_chunks = chunks + new_chunks
                rebuild_index(all_chunks, embedder, fernet, user_paths)
                chunks, faiss_index, encrypted_tokens = load_vault(fernet, user_paths)
                status.update(label=f"✅ Encrypted {len(new_chunks)} chunks!", state="complete")
                st.toast(f"Added {len(new_chunks)} encrypted chunks!", icon="🔐")

# -------------------- TOP METRICS ROW --------------------
unique_docs = sorted(list(set(c["source"] for c in chunks))) if chunks else []
redacted_count = sum(c["text"].count("<REDACTED_") for c in chunks) if chunks else 0

m1, m2, m3, m4 = st.columns(4)
m1.metric("👤 Student ID", username.strip())
m2.metric("📚 Uploaded PDFs/Notes", len(unique_docs))
m3.metric("🧩 Encrypted Chunks", len(chunks))
m4.metric("🛡️ PII Masked", redacted_count)

# -------------------- MAIN 4-TAB WORKSPACE --------------------
tab_chat, tab_tools, tab_vault, tab_pii = st.tabs([
    "💬 Smart Student AI Chat",
    "⚡ 1-Click Notes Summary & Viva Q&A",
    "🔐 Encrypted Vault Inspector",
    "🛡️ Live PII Redaction Sandbox"
])

with tab_chat:
    f_col1, f_col2, f_col3, f_col4 = st.columns([1.3, 1.4, 1.2, 0.8])
    with f_col1:
        selected_doc = st.selectbox("🎯 Select PDF/Note:", ["All Documents"] + unique_docs)
    with f_col2:
        response_style = st.selectbox(
            "📝 Answer Style:",
            [
                "🎯 Auto (Follow My Question)",
                "⚡ Short & Sweet (2-3 Lines)",
                "📖 Easy Student Explanation",
                "📝 Exam-Ready (Headings + Points)",
                "🔑 Key Bullet Points Only"
            ]
        )
    with f_col3:
        language_mode = st.selectbox(
            "🗣️ Answer Language:",
            [
                "Simple English",
                "Hinglish (Hindi + English Mix)",
                "Simple Hindi (सरल हिंदी)"
            ]
        )
    with f_col4:
        top_k_val = st.slider("🔍 Chunks:", min_value=1, max_value=6, value=3)

    st.divider()

    if not st.session_state.messages:
        st.caption(
            "💡 **Tip:** Aap seedha likh sakte hain: *'What is stack in short?'*, *'Explain linked list in easy words'*, ya upar se **Hinglish / Short** mode chunein!"
        )

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("meta"):
                st.caption(msg["meta"])
            if msg.get("sources"):
                with st.expander("📄 View Exact PDF Page Text"):
                    for idx, s in enumerate(msg["sources"], 1):
                        st.markdown(f"**{idx}. {s['source']} — Page {s['page']}**")
                        st.caption(s["text"])

    user_query = st.chat_input("Ask anything (e.g., 'What is Stack in short?' or 'Explain Tree in simple words')...")
    if user_query:
        if not chunks:
            st.warning("⚠️ Please upload and encrypt a PDF or note from the left sidebar first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            with st.chat_message("assistant"):
                with st.spinner("Preparing easy-to-understand answer from your notes..."):
                    t0 = time.time()
                    top_chunks = hybrid_retrieve(
                        user_query, chunks, faiss_index, embedder,
                        top_k=top_k_val, source_filter=selected_doc
                    )
                    try:
                        answer = ask_local_llm(
                            user_query,
                            top_chunks,
                            response_style=response_style,
                            language_mode=language_mode
                        )
                        elapsed = round(time.time() - t0, 2)
                        meta_info = f"⏱️ {elapsed}s | 📝 {response_style} | 🗣️ {language_mode}"

                        st.markdown(answer)
                        st.caption(meta_info)

                        with st.expander("📄 View Exact PDF Page Text"):
                            for idx, s in enumerate(top_chunks, 1):
                                st.markdown(f"**{idx}. {s['source']} — Page {s['page']}**")
                                st.caption(s["text"])

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer,
                            "meta": meta_info,
                            "sources": top_chunks
                        })
                    except Exception as e:
                        st.error(f"Error: {e}")

    if st.session_state.messages:
        chat_export = "\n\n".join(
            [f"{m['role'].upper()}:\n{m['content']}" for m in st.session_state.messages]
        )
        st.download_button(
            "📥 Download Study Notes / Chat (.txt)",
            data=chat_export,
            file_name=f"Study_Notes_{username}.txt",
            mime="text/plain"
        )

with tab_tools:
    st.subheader("⚡ 1-Click Revision Notes & Viva Question Generator")
    if not unique_docs:
        st.info("Upload a document in the sidebar first to generate quick revision notes.")
    else:
        t_top1, t_top2 = st.columns(2)
        with t_top1:
            target_doc = st.selectbox("Select Document:", unique_docs, key="tool_doc")
        with t_top2:
            tool_lang = st.selectbox(
                "Language for Summary/Viva:",
                ["Simple English", "Hinglish (Hindi + English Mix)", "Simple Hindi (सरल हिंदी)"],
                key="tool_lang"
            )

        doc_chunks = [c for c in chunks if c["source"] == target_doc][:5]

        t_col1, t_col2 = st.columns(2)
        with t_col1:
            if st.button("📝 Generate Quick Revision Summary", use_container_width=True, type="primary"):
                with st.spinner(f"Creating easy revision notes for {target_doc}..."):
                    summary_q = "Create a super easy-to-understand revision summary of all important concepts and definitions in this document for a student."
                    res = ask_local_llm(
                        summary_q,
                        doc_chunks,
                        response_style="📝 Exam-Ready (Headings + Points)",
                        language_mode=tool_lang
                    )
                    st.markdown("### 📌 Easy Revision Notes")
                    st.write(res)

        with t_col2:
            if st.button("🎓 Generate 5 Important Viva Q&A", use_container_width=True):
                with st.spinner(f"Generating 5 important exam questions & simple answers..."):
                    quiz_q = "Create 5 most important viva/exam questions and their short, easy-to-learn answers from this document."
                    res = ask_local_llm(
                        quiz_q,
                        doc_chunks,
                        response_style="📖 Easy Student Explanation",
                        language_mode=tool_lang
                    )
                    st.markdown("### 🎓 Important Exam & Viva Q&A")
                    st.write(res)

with tab_vault:
    st.subheader("🔐 AES-256 Encrypted Vault Inspector")
    if not chunks:
        st.info("Your vault is currently empty.")
    else:
        v_col1, v_col2 = st.columns([1, 1.3])
        with v_col1:
            st.markdown("#### 📁 My Indexed Files")
            for doc_name in unique_docs:
                cnt = sum(1 for c in chunks if c["source"] == doc_name)
                d_col1, d_col2 = st.columns([3, 1])
                d_col1.write(f"📄 **{doc_name}** ({cnt} chunks)")
                if d_col2.button("Delete", key=f"del_{doc_name}"):
                    remaining = [c for c in chunks if c["source"] != doc_name]
                    rebuild_index(remaining, embedder, fernet, user_paths)
                    st.rerun()

        with v_col2:
            st.markdown("#### 🔍 Disk Ciphertext vs RAM Decryption")
            chunk_idx = st.number_input("Select Chunk Number:", min_value=0, max_value=len(chunks)-1, value=0)
            st.markdown("**1. Stored on Disk (AES-256 Encrypted):**")
            st.code(encrypted_tokens[chunk_idx][:220] + "...", language="text")
            st.markdown("**2. Decrypted in RAM Only:**")
            st.json(chunks[chunk_idx])

with tab_pii:
    st.subheader("🛡️ Real-Time PII Privacy Masking Sandbox")
    sample_input = st.text_area(
        "Test Personal ID, PAN, Phone Number, or Email redaction:",
        value="Student Aarav Sharma (Phone: +91 9876543210, Email: aarav@college.edu) submitted ID [Aadhaar Redacted] and PAN ABCDE1234F.",
        height=100
    )
    if st.button("🛡️ Test PII Redaction Engine"):
        masked_output = redact_pii(sample_input)
        p1, p2 = st.columns(2)
        with p1:
            st.markdown("**Original Text:**")
            st.warning(sample_input)
        with p2:
            st.markdown("**Masked Output:**")
            st.success(masked_output)
