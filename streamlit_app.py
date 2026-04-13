"""
RAG System — Streamlit Web Interface
"""
import os
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page Config ──
st.set_page_config(
    page_title="RAG System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Clean Light Theme ──
st.markdown("""
<style>
    /* ─── Main background ─── */
    .stApp {
        background: #f8fafc;
    }

    /* ─── Sidebar ─── */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] * {
        color: #334155 !important;
    }
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: #0f172a !important;
    }

    /* ─── Hide footer ─── */
    footer { visibility: hidden; }

    /* ─── Buttons ─── */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
    }

    /* ─── Hero section ─── */
    .hero {
        text-align: center;
        padding: 32px 20px 8px;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hero-sub {
        color: #64748b;
        font-size: 1rem;
        margin-top: 6px;
    }

    /* ─── Pipeline visualization ─── */
    .pipeline-bar {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        margin: 24px auto 28px;
        max-width: 900px;
        flex-wrap: wrap;
    }
    .pipe-step {
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 10px 16px;
        border-radius: 10px;
        font-size: 0.82rem;
        font-weight: 600;
        white-space: nowrap;
    }
    .pipe-step .step-icon {
        font-size: 1.1rem;
    }
    .pipe-arrow {
        color: #cbd5e1;
        font-size: 1.1rem;
        margin: 0 2px;
    }
    .pipe-ingest { background: #ecfdf5; color: #065f46; }
    .pipe-query  { background: #eff6ff; color: #1e40af; }
    .pipe-db     { background: #fef3c7; color: #92400e; border: 1.5px solid #fcd34d; }

    /* ─── Stat cards ─── */
    .stat-row {
        display: flex;
        gap: 16px;
        margin: 0 auto 28px;
        max-width: 900px;
        justify-content: center;
        flex-wrap: wrap;
    }
    .stat-card {
        flex: 1;
        min-width: 160px;
        max-width: 210px;
        padding: 18px 20px;
        border-radius: 14px;
        text-align: center;
        border: 1px solid;
    }
    .stat-card .stat-value {
        font-size: 1.8rem;
        font-weight: 800;
        line-height: 1.2;
    }
    .stat-card .stat-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 4px;
        opacity: 0.8;
    }
    .stat-green  { background: #ecfdf5; border-color: #a7f3d0; }
    .stat-green .stat-value { color: #059669; }
    .stat-green .stat-label { color: #065f46; }

    .stat-blue   { background: #eff6ff; border-color: #bfdbfe; }
    .stat-blue .stat-value { color: #2563eb; }
    .stat-blue .stat-label { color: #1e40af; }

    .stat-purple { background: #f5f3ff; border-color: #ddd6fe; }
    .stat-purple .stat-value { color: #7c3aed; }
    .stat-purple .stat-label { color: #5b21b6; }

    .stat-amber  { background: #fffbeb; border-color: #fde68a; }
    .stat-amber .stat-value { color: #d97706; }
    .stat-amber .stat-label { color: #92400e; }

    /* ─── Chat area ─── */
    .chat-divider {
        height: 1px;
        background: #e2e8f0;
        max-width: 900px;
        margin: 0 auto 20px;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 14px !important;
        max-width: 900px;
        margin-left: auto;
        margin-right: auto;
    }

    /* ─── Source cards ─── */
    .src-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 3px solid #6366f1;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.84rem;
        color: #475569;
        line-height: 1.55;
    }
    .src-card strong {
        color: #4f46e5;
    }
    .src-card .src-file {
        color: #94a3b8;
        font-size: 0.75rem;
    }

    /* ─── Empty state ─── */
    .empty-state {
        text-align: center;
        padding: 60px 20px;
        max-width: 560px;
        margin: 0 auto;
    }
    .empty-state .empty-icon {
        font-size: 3.5rem;
        margin-bottom: 16px;
    }
    .empty-state h3 {
        color: #1e293b;
        font-size: 1.3rem;
        margin-bottom: 8px;
    }
    .empty-state p {
        color: #64748b;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    .example-chip {
        display: inline-block;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 20px;
        padding: 8px 18px;
        margin: 6px 4px;
        font-size: 0.82rem;
        color: #475569;
        cursor: default;
        transition: all 0.2s;
    }
    .example-chip:hover {
        background: #e2e8f0;
        border-color: #cbd5e1;
    }

    /* ─── Sidebar file list ─── */
    .file-item {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 8px 12px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        margin-bottom: 6px;
        font-size: 0.85rem;
    }
    .file-item .file-icon { font-size: 1.1rem; }
    .file-item .file-name { font-weight: 600; color: #1e293b; flex: 1; }
    .file-item .file-size { color: #94a3b8; font-size: 0.75rem; }

    /* ─── Sidebar divider ─── */
    .side-divider {
        height: 1px;
        background: #e2e8f0;
        margin: 18px 0;
    }

    /* ─── Sidebar DB status ─── */
    .db-status {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 10px 14px;
        border-radius: 10px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .db-ok {
        background: #ecfdf5;
        color: #059669;
        border: 1px solid #a7f3d0;
    }
    .db-empty {
        background: #fef3c7;
        color: #92400e;
        border: 1px solid #fde68a;
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──
@st.cache_resource(show_spinner=False)
def load_pipeline(model_name, vectordb_path, data_path, provider="openrouter"):
    """Load RAG pipeline (cached across reruns)."""
    from src.pipeline import RAGPipeline
    pipeline = RAGPipeline(
        data_path=data_path,
        vectordb_path=vectordb_path,
        llm_model=model_name,
        provider=provider,
    )
    return pipeline


def get_vectordb_stats(vectordb_path):
    """Get vector database statistics."""
    if not os.path.exists(vectordb_path):
        return {"exists": False, "vectors": 0}
    try:
        from src.embedder import get_embedding_model, load_vectordb
        emb = get_embedding_model()
        vdb = load_vectordb(emb, vectordb_path)
        count = vdb._collection.count()
        return {"exists": True, "vectors": count}
    except Exception:
        return {"exists": True, "vectors": 0}


def get_data_files(data_path):
    """List document files in data folder."""
    if not os.path.exists(data_path):
        return []
    return [f for f in os.listdir(data_path) if f.lower().endswith((".pdf", ".txt"))]


# ── Session State Init ──
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None


# ── Sidebar ──
data_path = "./data"
files = get_data_files(data_path)
stats = get_vectordb_stats("./storage/shared_vectors")

with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown('<div class="side-divider"></div>', unsafe_allow_html=True)

    # Provider selection
    provider_options = ["OpenRouter (Cloud)", "Ollama (Local)"]
    selected_provider_label = st.selectbox(
        "🔌 LLM Provider",
        options=provider_options,
        index=0,
        help="OpenRouter = cloud (needs API key), Ollama = local (no internet)",
    )
    if "Ollama" in selected_provider_label:
        selected_provider = "ollama"
    else:
        selected_provider = "openrouter"

    # Provider status
    if selected_provider == "ollama":
        from src.generator import is_ollama_available, list_ollama_models
        ollama_online = is_ollama_available()
        if ollama_online:
            st.success("✅ Ollama is running")
            ollama_models = list_ollama_models()
        else:
            st.error("❌ Ollama is offline — start it first")
            ollama_models = []
    else:  # openrouter
        or_key = os.environ.get("OPENROUTER_API_KEY", "")
        if or_key:
            st.success("✅ OPENROUTER_API_KEY is set")
        else:
            st.warning("⚠️ Set OPENROUTER_API_KEY in .env")

    # Model selection — dynamic based on provider
    if selected_provider == "openrouter":
        model_options = {
            "GLM-4.5-Air (Best Balance)": "z-ai/glm-4.5-air:free",
            "Step-3.5-Flash (Fast)": "stepfun/step-3.5-flash:free",
            "GPT-OSS-120B (Strong)": "openai/gpt-oss-120b:free",
            "Trinity-Mini (Efficient)": "arcee-ai/trinity-mini:free",
        }
        selected_model_label = st.selectbox(
            "🤖 LLM Model",
            options=list(model_options.keys()),
            index=0,
            help="All models are FREE via OpenRouter",
        )
        selected_model = model_options[selected_model_label]
    else:  # ollama
        if ollama_models:
            selected_model = st.selectbox(
                "🤖 Ollama Model",
                options=ollama_models,
                index=0,
                help="Models pulled locally via `ollama pull <model>`",
            )
        else:
            st.warning("No models found. Run: `ollama pull llama3.2`")
            selected_model = "llama3.2"

    st.markdown('<div class="side-divider"></div>', unsafe_allow_html=True)

    # Chunking settings
    st.markdown("### 📐 Chunking")
    chunk_size = st.slider("Chunk Size", 200, 2000, 1000, 100, help="Max characters per chunk")
    chunk_overlap = st.slider("Overlap", 0, 500, 100, 50, help="Overlap between chunks")
    top_k = st.slider("Top-K", 1, 10, 3, 1, help="Results returned per query")

    st.markdown('<div class="side-divider"></div>', unsafe_allow_html=True)

    # Documents
    st.markdown("### 📁 Documents")
    if files:
        for f in files:
            icon = "📄" if f.endswith(".pdf") else "📝"
            size_kb = os.path.getsize(os.path.join(data_path, f)) / 1024
            st.markdown(
                f'<div class="file-item">'
                f'<span class="file-icon">{icon}</span>'
                f'<span class="file-name">{f}</span>'
                f'<span class="file-size">{size_kb:.1f} KB</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("No documents in `data/`")

    st.markdown('<div class="side-divider"></div>', unsafe_allow_html=True)

    # Vector DB
    st.markdown("### 🗄️ Vector DB")
    if stats["exists"] and stats["vectors"] > 0:
        st.markdown(
            f'<div class="db-status db-ok">✅ {stats["vectors"]} vectors stored</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="db-status db-empty">⚠️ Empty — ingest first</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<div class="side-divider"></div>', unsafe_allow_html=True)

    # Buttons
    col_a, col_b = st.columns(2)
    with col_a:
        ingest_clicked = st.button("📥 Ingest", use_container_width=True, type="primary")
    with col_b:
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)

    if ingest_clicked:
        if not files:
            st.error("No documents found!")
        else:
            with st.spinner("Processing documents..."):
                pipeline = load_pipeline(selected_model, "./storage/shared_vectors", data_path, selected_provider)
                success = pipeline.ingest(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                if success:
                    st.session_state.pipeline = pipeline
                    st.session_state.ingested = True
                    st.session_state.rag_ready = False
                    st.success("✅ Done!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("❌ Failed!")

    if clear_clicked:
        st.session_state.messages = []
        st.rerun()

    st.markdown('<div class="side-divider"></div>', unsafe_allow_html=True)
    st.caption("Built with LangChain · ChromaDB")
    st.caption("OpenRouter · Ollama")


# ═════════════════════════════════════
# ── Main Content ──
# ═════════════════════════════════════

# Hero
st.markdown(
    '<div class="hero">'
    '<div class="hero-title">🔍 RAG System</div>'
    '<div class="hero-sub">Ask questions about your documents — answers grounded in your data</div>'
    '</div>',
    unsafe_allow_html=True,
)

# Pipeline flow
st.markdown(
    '<div class="pipeline-bar">'
    '<span class="pipe-step pipe-ingest"><span class="step-icon">📂</span> Load</span>'
    '<span class="pipe-arrow">→</span>'
    '<span class="pipe-step pipe-ingest"><span class="step-icon">✂️</span> Chunk</span>'
    '<span class="pipe-arrow">→</span>'
    '<span class="pipe-step pipe-ingest"><span class="step-icon">🧮</span> Embed</span>'
    '<span class="pipe-arrow">→</span>'
    '<span class="pipe-step pipe-db"><span class="step-icon">🗄️</span> VectorDB</span>'
    '<span class="pipe-arrow">→</span>'
    '<span class="pipe-step pipe-query"><span class="step-icon">🔎</span> Retrieve</span>'
    '<span class="pipe-arrow">→</span>'
    '<span class="pipe-step pipe-query"><span class="step-icon">🤖</span> LLM</span>'
    '<span class="pipe-arrow">→</span>'
    '<span class="pipe-step pipe-query"><span class="step-icon">💬</span> Answer</span>'
    '</div>',
    unsafe_allow_html=True,
)

# Stats
st.markdown(
    f'<div class="stat-row">'
    f'<div class="stat-card stat-green"><div class="stat-value">{len(files)}</div><div class="stat-label">Documents</div></div>'
    f'<div class="stat-card stat-blue"><div class="stat-value">{stats["vectors"]}</div><div class="stat-label">Vectors</div></div>'
    f'<div class="stat-card stat-purple"><div class="stat-value">{chunk_size}</div><div class="stat-label">Chunk Size</div></div>'
    f'<div class="stat-card stat-amber"><div class="stat-value">{top_k}</div><div class="stat-label">Top-K</div></div>'
    f'</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="chat-divider"></div>', unsafe_allow_html=True)

# ── Chat Interface ──
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
            with st.expander(f"📚 {len(msg['sources'])} Source Chunks", expanded=False):
                for i, src in enumerate(msg["sources"]):
                    st.markdown(
                        f'<div class="src-card">'
                        f'<strong>Chunk {i+1}</strong> '
                        f'<span class="src-file">— {src.get("source", "Unknown")}</span><br>'
                        f'{src.get("preview", "")}'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
            if "time" in msg:
                st.caption(f"⏱️ {msg['time']:.1f}s")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if stats["vectors"] == 0 and not st.session_state.ingested:
        st.warning("⚠️ Please **Ingest Documents** first using the sidebar button.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching documents..."):
                start_time = time.time()

                if st.session_state.pipeline is None:
                    st.session_state.pipeline = load_pipeline(selected_model, "./storage/shared_vectors", data_path, selected_provider)

                pipeline = st.session_state.pipeline

                if not st.session_state.rag_ready:
                    pipeline.llm_model = selected_model
                    pipeline.provider = selected_provider
                    success = pipeline.setup_qa(top_k=top_k)
                    if success:
                        st.session_state.rag_ready = True
                    else:
                        st.error("❌ Failed to set up. Ingest documents first.")
                        st.stop()

                answer, sources = pipeline.query(prompt)
                elapsed = time.time() - start_time

            st.markdown(answer)

            source_data = []
            for src in sources:
                meta = src.metadata if hasattr(src, "metadata") else {}
                source_name = os.path.basename(meta.get("source", "Unknown"))
                preview = src.page_content[:200].replace("\n", " ") if hasattr(src, "page_content") else ""
                source_data.append({"source": source_name, "preview": preview + "..."})

            if source_data:
                with st.expander(f"📚 {len(source_data)} Source Chunks", expanded=False):
                    for i, src in enumerate(source_data):
                        st.markdown(
                            f'<div class="src-card">'
                            f'<strong>Chunk {i+1}</strong> '
                            f'<span class="src-file">— {src["source"]}</span><br>'
                            f'{src["preview"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

            st.caption(f"⏱️ {elapsed:.1f}s")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": source_data,
            "time": elapsed,
        })

# ── Empty State ──
if not st.session_state.messages:
    st.markdown(
        '<div class="empty-state">'
        '<div class="empty-icon">💬</div>'
        '<h3>Start a Conversation</h3>'
        '<p>Ask any question about your ingested documents. '
        'The AI searches through your data and provides grounded answers.</p>'
        '<div style="margin-top: 20px;">'
        '<span class="example-chip">💡 What is retrieval augmented generation?</span>'
        '<span class="example-chip">💡 How do transformers work?</span>'
        '<span class="example-chip">💡 What are the types of machine learning?</span>'
        '</div>'
        '</div>',
        unsafe_allow_html=True,
    )