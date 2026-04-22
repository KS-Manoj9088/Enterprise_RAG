"""
========================================
Steps 4, 5: LLM → Response
========================================
Sends relevant data + query to LLM and generates response.

Supports TWO providers:
  1. OpenRouter API (cloud, FREE models) — needs OPENROUTER_API_KEY
  2. Ollama (local) — needs Ollama installed from https://ollama.com

Supported FREE OpenRouter models:
  - z-ai/glm-4.5-air:free       → BEST BALANCE (agents + chat)
  - stepfun/step-3.5-flash:free  → fast + good reasoning
  - openai/gpt-oss-120b:free     → strong reasoning
  - arcee-ai/trinity-mini:free   → efficient + long context

Supported Ollama models (pull with `ollama pull <model>`):
  - llama3.2  → Meta Llama 3.2 (3B, fast)
  - llama3.1  → Meta Llama 3.1 (8B, balanced)
  - mistral   → Mistral 7B
  - phi3      → Microsoft Phi-3
  - gemma2    → Google Gemma 2

"""
import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load .env file automatically (override=True picks up key changes)
load_dotenv(override=True)

# Import strict prompt from prompts folder
from prompts.strict.strict_qa import STRICT_PROMPT

# Active prompt template (change this to switch prompt styles)
# Options:
#   from prompts.strict.strict_qa import STRICT_PROMPT
#   from prompts.strict.strict_cited import STRICT_CITED_PROMPT
#   from prompts.conversational.friendly import CONVERSATIONAL_PROMPT
#   from prompts.conversational.eli5 import ELI5_PROMPT
#   from prompts.analytical.summarizer import SUMMARY_PROMPT
#   from prompts.analytical.comparison import COMPARISON_PROMPT
#   from prompts.analytical.step_by_step import STEP_BY_STEP_PROMPT
#   from prompts.domain.technical import TECHNICAL_PROMPT
#   from prompts.domain.academic import ACADEMIC_PROMPT
#   from prompts.domain.legal import LEGAL_PROMPT
PROMPT_TEMPLATE = STRICT_PROMPT

# ── Default Ollama models (shown even if Ollama is offline) ──
DEFAULT_OLLAMA_MODELS = [
    "llama3.2",
    "llama3.1",
    "mistral",
    "phi3",
    "gemma2",
]


def get_ollama_base_url() -> str:
    """Return the Ollama server base URL (configurable via env var)."""
    return os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def is_ollama_available() -> bool:
    """Check if Ollama server is running and reachable."""
    try:
        resp = httpx.get(f"{get_ollama_base_url()}/api/tags", timeout=3)
        return resp.status_code == 200
    except Exception:
        return False


def list_ollama_models() -> list[str]:
    """
    Fetch the list of locally-available Ollama models.

    Returns:
        List of model name strings, e.g. ['llama3.2', 'mistral']
    """
    try:
        resp = httpx.get(f"{get_ollama_base_url()}/api/tags", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        pass
    return []


def get_openrouter_llm(model_name: str = "z-ai/glm-4.5-air:free", temperature: float = 0.3):
    """
    Load the LLM using OpenRouter API (FREE models).

    Args:
        model_name: OpenRouter model name
        temperature: Creativity (0=focused, 1=creative)

    Returns:
        ChatOpenAI instance configured for OpenRouter
    """
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise ValueError(
            "OPENROUTER_API_KEY not set!\n"
            "1. Get free key at: https://openrouter.ai/keys\n"
            "2. Set it: $env:OPENROUTER_API_KEY='sk-or-...'\n"
        )

    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        openai_api_key=api_key,
        openai_api_base="https://openrouter.ai/api/v1",
    )
    print(f"[4] LLM loaded: {model_name} (via OpenRouter, temperature={temperature})")
    return llm


def get_ollama_llm(model_name: str = "llama3.2", temperature: float = 0.3):
    """
    Load the LLM using a locally-running Ollama server.

    Prerequisites:
        1. Install Ollama: https://ollama.com
        2. Pull a model: ollama pull llama3.2
        3. Ollama server must be running (starts automatically after install)

    Args:
        model_name: Ollama model name (e.g. 'llama3.2', 'mistral')
        temperature: Creativity (0=focused, 1=creative)

    Returns:
        ChatOllama instance
    """
    from langchain_ollama import ChatOllama

    base_url = get_ollama_base_url()

    if not is_ollama_available():
        raise ValueError(
            f"Ollama server is not reachable at {base_url}!\n"
            "1. Install Ollama from: https://ollama.com\n"
            "2. Make sure the Ollama service is running.\n"
            f"3. Verify with: curl {base_url}/api/tags\n"
        )

    llm = ChatOllama(
        model=model_name,
        temperature=temperature,
        base_url=base_url,
    )
    print(f"[4] LLM loaded: {model_name} (via Ollama @ {base_url}, temperature={temperature})")
    return llm


def get_llm(model_name="z-ai/glm-4.5-air:free", temperature=0.3, provider="openrouter"):
    """
    Load the LLM from the selected provider.

    Args:
        model_name: Model identifier for the chosen provider
        temperature: Creativity (0=focused, 1=creative)
        provider: 'openrouter' | 'ollama'

    Returns:
        LangChain chat model instance
    """
    if provider == "ollama":
        return get_ollama_llm(model_name, temperature)
    else:
        return get_openrouter_llm(model_name, temperature)


def format_docs(docs):
    """Format retrieved documents into a single context string."""
    return "\n\n".join(doc.page_content for doc in docs)


def deduplicate_docs(docs):
    """Remove duplicate chunks that have identical page content."""
    seen = set()
    unique = []
    for doc in docs:
        content = doc.page_content.strip()
        if content not in seen:
            seen.add(content)
            unique.append(doc)
    if len(unique) < len(docs):
        print(f"[Dedup] Removed {len(docs) - len(unique)} duplicate chunks ({len(unique)} unique remaining)")
    return unique


def create_rag_chain(llm, retriever, prompt_template: str | None = None):
    """
    Create the complete RAG chain connecting retriever → LLM.

    Architecture Mapping:
        Full chain: Query(1) → Embed(2) → Retrieve(3) → LLM(4) → Response(5)

    Args:
        llm: Language model
        retriever: Vector database retriever
        prompt_template: Optional override for the prompt template string.
                         Defaults to the module-level PROMPT_TEMPLATE.

    Returns:
        LCEL RAG chain
    """
    template = prompt_template if prompt_template is not None else PROMPT_TEMPLATE
    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    rag_chain = (
        {"context": retriever | deduplicate_docs | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    print("[4+5] RAG Chain created: Retriever -> LLM -> Response")
    return rag_chain


def generate_response(rag_chain, query: str, retriever=None):
    """
    Generate a response using the full RAG pipeline.

    Architecture Mapping:
        4 → Relevant data + Query sent to LLM
        5 → Response generated

    Args:
        rag_chain: Complete RAG chain
        query: User's question
        retriever: Optional retriever to fetch source docs separately

    Returns:
        Tuple of (answer_text, source_documents)
    """
    print(f"\n{'='*60}")
    print(f"[1] Query: {query}")

    answer = rag_chain.invoke(query)

    # Fetch source docs separately if retriever is provided
    raw_sources = retriever.invoke(query) if retriever else []
    sources = deduplicate_docs(raw_sources)

    print(f"[5] Response generated ({len(answer)} chars, {len(sources)} unique sources)")
    print(f"{'='*60}\n")

    return answer, sources


def direct_generate(llm, prompt_template: str, retriever, retrieval_query: str, user_question: str):
    """
    Generate a response by separating the retrieval query from the LLM question.

    This is essential for summarization: the vector search needs a broad query
    (e.g. 'key topics main ideas') to retrieve content, while the LLM prompt
    still receives the user's original question.

    Args:
        llm: Language model instance
        prompt_template: Prompt template string with {context} and {question}
        retriever: Vector DB retriever
        retrieval_query: Query used ONLY for vector similarity search
        user_question: Question shown to the LLM in the prompt

    Returns:
        Tuple of (answer_text, source_documents)
    """
    print(f"\n{'='*60}")
    print(f"[1] User question : {user_question}")
    print(f"[2] Retrieval query: {retrieval_query}")

    # Step 2+3: retrieve with the broad query
    raw_docs = retriever.invoke(retrieval_query)
    docs = deduplicate_docs(raw_docs)
    print(f"[3] Retrieved {len(docs)} unique chunks (from {len(raw_docs)} total)")

    if not docs:
        msg = "No relevant content was found in your documents for this question."
        print(f"[!] {msg}")
        return msg, []

    context = format_docs(docs)

    # Step 4+5: call LLM with the context and the ORIGINAL user question
    template = prompt_template if prompt_template else PROMPT_TEMPLATE
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": user_question})

    print(f"[5] Response generated ({len(answer)} chars, {len(docs)} sources)")
    print(f"{'='*60}\n")

    return answer, docs