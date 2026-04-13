# 🔍 RAG System — Enterprise Edition

A complete, production-ready **Retrieval Augmented Generation** system with **Google OAuth**, **per-user document management**, **OCR support**, and a **React frontend**. Ask questions about your documents — answers are grounded in your data, never hallucinated.

**100% free** — open-source embeddings (HuggingFace) + free LLMs via OpenRouter or run models locally with Ollama.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Features](#features)
3. [Tech Stack](#tech-stack)
4. [Project Structure](#project-structure)
5. [Prerequisites](#prerequisites)
6. [Setup Instructions](#setup-instructions)
   - [Step 1: Clone & Install Python Dependencies](#step-1-clone--install-python-dependencies)
   - [Step 2: Get OpenRouter API Key (Free)](#step-2-get-openrouter-api-key-free)
   - [Step 3: Set Up Google OAuth](#step-3-set-up-google-oauth)
   - [Step 4: Configure Environment Variables](#step-4-configure-environment-variables)
   - [Step 5: Install Frontend Dependencies](#step-5-install-frontend-dependencies)
   - [Step 6: (Optional) Install Tesseract for OCR](#step-6-optional-install-tesseract-for-ocr)
7. [Running the Application](#running-the-application)
8. [How It Works](#how-it-works)
9. [API Reference](#api-reference)
10. [Source Code Breakdown](#source-code-breakdown)
    - [RAG Core (src/)](#rag-core-src)
    - [Backend (backend/)](#backend-backend)
    - [Frontend (frontend/src/)](#frontend-frontendsrc)
    - [Prompt Templates (prompts/)](#prompt-templates-prompts)
11. [Configuration & Customization](#configuration--customization)
12. [Troubleshooting](#troubleshooting)
13. [Legacy Interfaces](#legacy-interfaces)

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        ENTERPRISE RAG SYSTEM                            │
│                                                                          │
│   ┌──────────────────────┐          ┌──────────────────────────┐         │
│   │   React Frontend     │  REST    │   FastAPI Backend         │         │
│   │   (Vite :5173)       │◄────────►│   (Uvicorn :8000)        │         │
│   │                      │   API    │                          │         │
│   │  • Google Sign-In    │          │  • Google OAuth verify   │         │
│   │  • Document Upload   │          │  • JWT Authentication    │         │
│   │  • Chat Interface    │          │  • Per-user storage      │         │
│   │  • Settings Panel    │          │  • OCR processing        │         │
│   └──────────────────────┘          │  • RAG pipeline          │         │
│                                     └─────────┬────────────────┘         │
│                                               │                          │
│   ┌───────────────────────────────────────────┼──────────────────────┐   │
│   │              RAG ENGINE (src/)             │                      │   │
│   │                                           ▼                      │   │
│   │  ┌────────┐  ┌─────────┐  ┌───────┐  ┌────────┐                 │   │
│   │  │ A.Load │─►│B.Extract│─►│C.Chunk│─►│D.Embed │                 │   │
│   │  │ PDF/TXT│  │  Text   │  │ 1000  │  │ HF     │                 │   │
│   │  └────────┘  └─────────┘  │ chars │  │ model  │                 │   │
│   │                           └───────┘  └───┬────┘                 │   │
│   │                                          ▼                      │   │
│   │                                  ┌──────────────┐               │   │
│   │                                  │  ChromaDB    │               │   │
│   │                                  │ (per-user)   │               │   │
│   │                                  └──────┬───────┘               │   │
│   │                                         │                       │   │
│   │  ┌────────┐  ┌─────────┐  ┌──────────┐ │                       │   │
│   │  │1.Query │─►│2.Embed  │─►│3.Retrieve│◄┘                       │   │
│   │  │        │  │  Query  │  │  Top-K   │                          │   │
│   │  └────────┘  └─────────┘  └────┬─────┘                          │   │
│   │                                ▼                                │   │
│   │                         ┌────────────┐                          │   │
│   │                         │ 4. LLM     │ (OpenRouter cloud / Ollama local) │   │
│   │                         └─────┬──────┘                          │   │
│   │                               ▼                                 │   │
│   │                         ┌────────────┐                          │   │
│   │                         │ 5.Response │ (grounded in docs)       │   │
│   │                         └────────────┘                          │   │
│   └──────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Google OAuth** | Sign in with Google — no passwords to manage |
| **Per-user Isolation** | Each user gets their own document storage & vector database |
| **Document Upload** | Upload PDF, TXT, or image files — directly in the browser |
| **50 MB Limit** | Files over 50 MB are rejected with a clear error message |
| **OCR Support** | Extract text from scanned PDFs and images using Tesseract |
| **Smart Fallback** | If PDF text extraction yields < 50 chars, auto-falls back to OCR |
| **Multiple LLMs** | Switch between OpenRouter cloud models (free) or run locally with Ollama |
| **Strict Prompt** | LLM only answers from your documents — no hallucination |
| **9 Prompt Templates** | Strict, conversational, analytical, and domain-specific styles |
| **Source Citations** | Every answer shows which document chunks were used |
| **React Frontend** | Modern UI with Tailwind CSS — sidebar, chat, file management |
| **REST API** | Full FastAPI backend with Swagger docs at `/docs` |
| **Streamlit UI** | Legacy Streamlit interface still available |

---

## Tech Stack

| Layer | Technology | Role |
|-------|-----------|------|
| **Frontend** | React 19, Vite, Tailwind CSS | User interface |
| **Auth** | Google OAuth 2.0, JWT (python-jose) | Authentication |
| **Backend** | FastAPI, Uvicorn, SQLAlchemy | REST API server |
| **Database** | SQLite | User & document metadata |
| **RAG Core** | LangChain (LCEL pattern) | Pipeline orchestration |
| **Embeddings** | HuggingFace `all-MiniLM-L6-v2` | Text → vectors (local, free) |
| **Vector DB** | ChromaDB | Similarity search storage |
| **LLM** | OpenRouter (cloud, free) / Ollama (local) | Answer generation |
| **OCR** | Tesseract + pdf2image | Scanned document extraction |
| **PDF** | PyPDF | Regular PDF text extraction |

---

## Project Structure

```
ldsrag/
│
├── .env                        # API keys & secrets (DO NOT COMMIT)
├── .env.example                # Template for .env
├── .gitignore                  # Git ignore rules
├── requirements.txt            # All Python dependencies
├── README.md                   # This file
├── run.py                      # Backend launcher (python run.py)
├── app.py                      # CLI entry point (legacy)
├── streamlit_app.py            # Streamlit UI (legacy)
│
├── backend/                    # ── FastAPI Backend ──
│   ├── __init__.py
│   ├── main.py                 #   App factory, CORS, routers, startup
│   ├── auth.py                 #   Google OAuth verification + JWT
│   ├── models.py               #   SQLAlchemy models (User, Document)
│   ├── database.py             #   SQLite engine + session
│   ├── ocr.py                  #   Tesseract OCR processing
│   └── routes/
│       ├── __init__.py
│       ├── auth.py             #   POST /api/auth/google, GET /api/auth/me
│       ├── documents.py        #   POST /api/documents/upload, GET, DELETE
│       └── rag.py              #   POST /api/rag/ingest, POST /api/rag/query
│
├── frontend/                   # ── React Frontend ──
│   ├── .env                    #   VITE_GOOGLE_CLIENT_ID
│   ├── .env.example            #   Template
│   ├── index.html              #   HTML entry
│   ├── package.json            #   npm dependencies
│   ├── vite.config.js          #   Vite config with API proxy
│   ├── tailwind.config.js      #   Tailwind theme
│   ├── postcss.config.js       #   PostCSS config
│   └── src/
│       ├── main.jsx            #   React root + GoogleOAuthProvider
│       ├── App.jsx             #   Routes + auth guards
│       ├── api.js              #   Axios client + JWT interceptor
│       ├── index.css           #   Global styles + Tailwind
│       ├── context/
│       │   └── AuthContext.jsx #   Auth state management
│       ├── pages/
│       │   ├── Login.jsx       #   Google Sign-In page
│       │   └── Dashboard.jsx   #   Main app layout
│       └── components/
│           ├── Navbar.jsx      #   Top bar with user info
│           ├── Sidebar.jsx     #   Upload, docs, settings, ingest
│           └── ChatPanel.jsx   #   Chat messages, sources, input
│
├── src/                        # ── RAG Engine ──
│   ├── __init__.py
│   ├── loader.py               #   Steps A+B: Load PDF/TXT files
│   ├── chunker.py              #   Step C: Split text into chunks
│   ├── embedder.py             #   Step D: HuggingFace embed + ChromaDB
│   ├── retriever.py            #   Steps 1-3: Query → similarity search
│   ├── generator.py            #   Steps 4-5: OpenRouter LLM → response
│   └── pipeline.py             #   RAGPipeline orchestrator class
│
├── prompts/                    # ── 9 Prompt Templates ──
│   ├── strict/
│   │   ├── strict_qa.py        #   Default — document-grounded only
│   │   └── strict_cited.py     #   With citation markers
│   ├── conversational/
│   │   ├── friendly.py         #   Friendly tone
│   │   └── eli5.py             #   Explain Like I'm 5
│   ├── analytical/
│   │   ├── summarizer.py       #   Summarization
│   │   ├── comparison.py       #   Compare & contrast
│   │   └── step_by_step.py     #   Step-by-step reasoning
│   └── domain/
│       ├── technical.py        #   Technical documentation
│       ├── academic.py         #   Academic tone
│       └── legal.py            #   Legal document style
│
├── data/                       # ── Sample Documents ──
│   ├── sample_document.txt
│   ├── machine_learning_guide.txt
│   └── ai_complete_reference.pdf
│
├── storage/                    # ── All Persistent Data (git-ignored) ──
│   ├── db/                     #   SQLite database (rag.db)
│   ├── uploads/                #   Per-user uploaded files
│   │   └── <user-uuid>/
│   ├── vectors/                #   Per-user vector databases
│   │   └── <user-uuid>/
│   └── shared_vectors/         #   Shared vector DB (legacy/CLI)
│
├── tests/                      # ── Test Suite ──
│   ├── __init__.py
│   └── test_pipeline.py        #   RAG pipeline unit tests
│
├── docs/                       # ── Documentation ──
│   └── README.md
│
└── .streamlit/                 # Streamlit config
    └── config.toml
```

---

## Prerequisites

| Requirement | Version | Check Command |
|-------------|---------|---------------|
| **Python** | 3.10+ | `python --version` |
| **Node.js** | 18+ | `node --version` |
| **npm** | 9+ | `npm --version` |
| **Git** | Any | `git --version` |
| Tesseract *(optional, for OCR)* | 5.0+ | `tesseract --version` |

---

## Setup Instructions

### Step 1: Clone & Install Python Dependencies

```bash
# Clone the repository
git clone https://github.com/Sriram-098/SummarizerUsingRag.git
cd SummarizerUsingRag

# Create a virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install all Python packages
pip install -r requirements.txt
```

This installs:
- **RAG core**: LangChain, ChromaDB, HuggingFace, PyPDF, sentence-transformers
- **Backend**: FastAPI, Uvicorn, SQLAlchemy, python-jose, google-auth
- **OCR**: pytesseract, pdf2image, Pillow

### Step 2: Get OpenRouter API Key (Free)

1. Go to [https://openrouter.ai](https://openrouter.ai)
2. Sign up (free, no credit card needed)
3. Go to [https://openrouter.ai/keys](https://openrouter.ai/keys)
4. Click **Create Key**
5. Copy the key — it starts with `sk-or-v1-...`

### Step 3: Set Up Google OAuth

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project (or select an existing one)
3. Navigate to **APIs & Services** → **OAuth consent screen**
   - Choose **External**
   - Fill in app name, user support email, developer email
   - Add scope: `email`, `profile`, `openid`
   - Add your email as a test user
   - Click **Publish** or keep in testing mode
4. Navigate to **APIs & Services** → **Credentials**
5. Click **+ CREATE CREDENTIALS** → **OAuth client ID**
   - Application type: **Web application**
   - Name: `RAG Enterprise`
   - Authorized JavaScript origins: `http://localhost:5173`
   - Authorized redirect URIs: `http://localhost:5173`
6. Click **CREATE**
7. Copy the **Client ID** and **Client Secret**

> ⚠️ Use exactly `http://localhost:5173` — no trailing slash, `http://` not `https://`.

### Step 4: Configure Environment Variables

**Backend** — create `.env` in the project root:

```env
# OpenRouter API Key (from Step 2)
OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Google OAuth (from Step 3)
GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
GOOGLE_CLIENT_SECRET=your-client-secret

# JWT Secret (change this in production!)
JWT_SECRET=some-random-secret-string
```

**Frontend** — create `frontend/.env`:

```env
VITE_GOOGLE_CLIENT_ID=your-client-id.apps.googleusercontent.com
VITE_API_URL=http://localhost:8000
```

> ⚠️ The `GOOGLE_CLIENT_ID` must be **identical** in both `.env` files.

### Step 5: Install Frontend Dependencies

```bash
cd frontend
npm install
cd ..
```

### Step 6: (Optional) Install Tesseract for OCR

OCR is optional but enables text extraction from scanned PDFs and images.

**Windows:**
1. Download from [UB Mannheim Tesseract](https://github.com/UB-Mannheim/tesseract/wiki)
2. Run the installer
3. Add to PATH: `C:\Program Files\Tesseract-OCR`
4. Verify: `tesseract --version`

**Linux:**
```bash
sudo apt install tesseract-ocr
```

**Mac:**
```bash
brew install tesseract
```

The app detects Tesseract automatically. If not installed, the OCR checkbox simply won't appear in the sidebar.

---

## Running the Application

### Quick Start

Open **two terminals**, both with the virtual environment activated:

**Terminal 1 — Backend (FastAPI):**
```bash
# Windows
venv\Scripts\uvicorn.exe backend.main:app --reload --port 8000

# Linux/Mac
uvicorn backend.main:app --reload --port 8000
```

You should see:
```
✅ Database initialized
✅ Enterprise RAG API ready
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Terminal 2 — Frontend (React):**
```bash
cd frontend
npm run dev
```

You should see:
```
VITE v6.x.x  ready in xxx ms

➜  Local:   http://localhost:5173/
```

### Access Points

| URL | What |
|-----|------|
| `http://localhost:5173` | **React UI** — main application |
| `http://localhost:8000/docs` | **Swagger UI** — interactive API docs |
| `http://localhost:8000/api/health` | Health check endpoint |

### User Workflow

1. Open `http://localhost:5173`
2. Click **Sign in with Google** → authenticate with your Google account
3. Upload your PDF/TXT documents using the **Upload Document** button in the sidebar
4. Click **Ingest Documents** to process them into vectors
5. Type a question in the chat input — get grounded answers with source citations

### What "Ingest" Means

Ingestion is the one-time data preparation step:
- **Load** → reads your uploaded files
- **Extract** → pulls text from PDFs/TXT
- **Chunk** → splits text into 1000-char pieces (configurable)
- **Embed** → converts each chunk into a 384-dimensional vector
- **Store** → saves vectors into your private ChromaDB

After ingestion, your vectors persist on disk. You only need to re-ingest when you add or remove documents.

---

## How It Works

### Data Preparation (Steps A → D) — happens during "Ingest"

| Step | What Happens | Code File | Tool |
|------|-------------|-----------|------|
| **A. Raw Data** | Load PDF/TXT files from user's upload folder | `src/loader.py` | PyPDF, TextLoader |
| **B. Extract** | Extract text content (+ OCR fallback for scanned docs) | `src/loader.py`, `backend/ocr.py` | LangChain Loaders, Tesseract |
| **C. Chunk** | Split text into 1000-char pieces with 100-char overlap | `src/chunker.py` | RecursiveCharacterTextSplitter |
| **D. Embed + Store** | Convert chunks to 384-dim vectors, store in ChromaDB | `src/embedder.py` | HuggingFace `all-MiniLM-L6-v2` |

### Query Pipeline (Steps 1 → 5) — happens on each question

| Step | What Happens | Code File | Tool |
|------|-------------|-----------|------|
| **1. Query** | User asks a question in chat | `src/retriever.py` | — |
| **2. Embed Query** | Question converted to 384-dim vector | `src/retriever.py` | Same HuggingFace model |
| **3. Retrieve** | Find top-K most similar chunks via cosine similarity | `src/retriever.py` | ChromaDB |
| **Step 4 — LLM** | Send context + question to LLM with strict prompt | `src/generator.py` | OpenRouter *or* Ollama |
| **5. Response** | LLM answers from documents only — no hallucination | `src/generator.py` | — |

### Authentication Flow

```
User clicks "Sign in with Google"
    ↓
Google returns an ID token (JWT signed by Google)
    ↓
Frontend sends token to POST /api/auth/google
    ↓
Backend verifies token with Google's public keys
    ↓
Backend creates/finds user in SQLite database
    ↓
Backend issues its own JWT (valid 72 hours)
    ↓
Frontend stores JWT in localStorage
    ↓
Every API call includes: Authorization: Bearer <jwt>
    ↓
Backend validates JWT → extracts user → scopes data to that user
```

### Per-User Isolation

Each authenticated user gets:
- `storage/uploads/<user-uuid>/` — uploaded files stored here
- `storage/vectors/<user-uuid>/` — private ChromaDB instance
- Database records in `storage/db/rag.db` — document metadata

Users cannot see or query each other's documents.

---

## API Reference

All endpoints (except auth) require `Authorization: Bearer <jwt>` header.

### Authentication

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/api/auth/google` | `{ "credential": "<google-id-token>" }` | Verify Google token, return JWT |
| `GET` | `/api/auth/me` | — | Get current user info |

### Documents

| Method | Endpoint | Body / Params | Description |
|--------|----------|---------------|-------------|
| `POST` | `/api/documents/upload?use_ocr=false` | `multipart/form-data` (file) | Upload a document (max 50 MB) |
| `GET` | `/api/documents/` | — | List user's documents |
| `DELETE` | `/api/documents/{doc_id}` | — | Delete a document |

### RAG

| Method | Endpoint | Body | Description |
|--------|----------|------|-------------|
| `POST` | `/api/rag/ingest` | `{ "chunk_size": 1000, "chunk_overlap": 100, "model": "...", "provider": "openrouter" }` | Ingest all uploaded docs into vector DB |
| `POST` | `/api/rag/query` | `{ "question": "...", "top_k": 3, "model": "...", "provider": "openrouter" }` | Query the RAG pipeline |
| `GET` | `/api/rag/stats` | — | Get vector DB stats |
| `GET` | `/api/rag/providers` | — | List available providers and their models |

### System

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/ocr/status` | Check if Tesseract OCR is available |

Full interactive API docs available at: **http://localhost:8000/docs**

---

## Source Code Breakdown

### RAG Core (`src/`)

#### `src/loader.py` — Steps A + B: Load & Extract

Loads all PDF and TXT files from a given directory. Uses LangChain's `PyPDFLoader` for PDFs and `TextLoader` for text files. Returns a list of `Document` objects with extracted text and metadata.

```python
documents = load_documents("./data")
# → [Document(page_content="...", metadata={"source": "file.pdf"}), ...]
```

#### `src/chunker.py` — Step C: Chunking

Splits documents into smaller overlapping pieces using `RecursiveCharacterTextSplitter`. Default: 1000 chars per chunk, 100 chars overlap. Splits on paragraph breaks → sentence breaks → word breaks → character breaks.

```python
chunks = chunk_documents(documents, chunk_size=1000, chunk_overlap=100)
# → [Document(page_content="chunk text...", metadata={"source": "file.pdf"}), ...]
```

#### `src/embedder.py` — Step D: Embed + Store

Loads the `sentence-transformers/all-MiniLM-L6-v2` model (runs locally on CPU, no API key needed). Converts each chunk into a 384-dimensional vector. Stores vectors in ChromaDB with automatic persistence to disk.

```python
embeddings = get_embedding_model()
vectordb = store_in_vectordb(chunks, embeddings, "./storage/shared_vectors")
# Later:
vectordb = load_vectordb(embeddings, "./storage/shared_vectors")
```

#### `src/retriever.py` — Steps 1, 2, 3: Query → Retrieve

Creates a retriever from ChromaDB that performs cosine similarity search. When a question comes in, it's automatically embedded with the same model and matched against stored vectors.

```python
retriever = get_retriever(vectordb, top_k=3)
docs = retriever.invoke("What is machine learning?")
# → [Document(...), Document(...), Document(...)]
```

#### `src/generator.py` — Steps 4, 5: LLM → Response

Connects to OpenRouter's API (OpenAI-compatible) to access free LLMs, or to a local **Ollama** server for fully offline inference. Uses LangChain's LCEL (LangChain Expression Language) to build the chain:

```python
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

Imports the strict prompt from `prompts/strict/strict_qa.py` to ensure the LLM only answers from document context.

#### `src/pipeline.py` — RAGPipeline Orchestrator

The `RAGPipeline` class ties everything together:

```python
rag = RAGPipeline(
    data_path="./data",
    llm_model="z-ai/glm-4.5-air:free",
    provider="openrouter"  # or "ollama"
)
rag.ingest(chunk_size=1000, chunk_overlap=100)  # A → B → C → D
rag.setup_qa(top_k=3)                           # Connect retriever + LLM
answer, sources = rag.query("What is...?")       # 1 → 2 → 3 → 4 → 5
```

---

### Backend (`backend/`)

#### `backend/main.py` — FastAPI Application

Creates the FastAPI app with:
- CORS middleware (allows React dev server at `:5173`)
- Three route groups: auth, documents, RAG
- Startup event to initialize the SQLite database
- Health check and OCR status endpoints

#### `backend/auth.py` — Google OAuth + JWT

- `verify_google_token(token)` → uses `google-auth` library to verify the ID token against Google's public keys and your Client ID
- `create_jwt(user_id, email)` → issues a JWT valid for 72 hours
- `decode_jwt(token)` → validates and decodes JWT
- `get_current_user(credentials, db)` → FastAPI dependency that extracts the authenticated user from the Bearer token

#### `backend/models.py` — Database Models

Two SQLAlchemy models:

| Model | Fields |
|-------|--------|
| **User** | id (UUID), email (unique), name, picture, created_at |
| **Document** | id (UUID), user_id (FK), filename, original_name, size_bytes, content_type, ocr_used, chunk_count, uploaded_at |

#### `backend/database.py` — SQLite Setup

Creates the SQLite engine at `storage/db/rag.db` with SQLAlchemy. Provides `get_db()` dependency for FastAPI route injection.

#### `backend/ocr.py` — OCR Processing

- `is_tesseract_available()` → checks if Tesseract is installed
- `ocr_pdf(pdf_path)` → converts PDF pages to images at 300 DPI, runs Tesseract on each page
- `ocr_image(image_path)` → runs Tesseract on a single image (PNG, JPG, TIFF, BMP)
- `extract_text_with_ocr(file_path)` → smart extraction: tries regular text first, falls back to OCR if < 50 chars extracted

#### `backend/routes/auth.py` — Auth Endpoints

- `POST /api/auth/google` → receives Google ID token from frontend, verifies it against Google servers, creates user in database if first-time, returns JWT
- `GET /api/auth/me` → returns currently authenticated user's profile

#### `backend/routes/documents.py` — Document Management

- `POST /api/documents/upload` → accepts file upload (max 50 MB), validates extension, saves to `storage/uploads/<uuid>/`, optionally runs OCR, stores metadata in SQLite
- `GET /api/documents/` → lists all documents belonging to the authenticated user
- `DELETE /api/documents/{id}` → removes file from disk and database record

#### `backend/routes/rag.py` — RAG Pipeline API

- `POST /api/rag/ingest` → runs A → B → C → D on the user's documents, stores vectors in `storage/vectors/<uuid>/`
- `POST /api/rag/query` → runs 1 → 2 → 3 → 4 → 5 against the user's private vector DB, returns answer + source chunks with preview
- `GET /api/rag/stats` → returns vector count for the authenticated user
- `GET /api/rag/providers` → returns available providers (OpenRouter, Ollama) with their models and status
- Maintains an in-memory cache of per-user pipelines for fast repeated queries

---

### Frontend (`frontend/src/`)

#### `main.jsx` — React Entry Point

Wraps the app in `GoogleOAuthProvider` (Client ID from env), `BrowserRouter`, and `AuthProvider`. Initializes react-hot-toast for notifications.

#### `App.jsx` — Router & Auth Guards

Two routes:
- `/` → Login page (redirects to `/app` if already authenticated)
- `/app` → Dashboard (protected — redirects to `/` if not authenticated)

Uses a `ProtectedRoute` wrapper that checks auth state.

#### `context/AuthContext.jsx` — Auth State Management

React context providing:
- `user` — current user object or null
- `loading` — true while checking existing session
- `login(credential)` — calls POST `/api/auth/google`, stores JWT + user in localStorage
- `logout()` — clears localStorage, resets state

On mount, checks for existing JWT in localStorage and validates it with GET `/api/auth/me`.

#### `api.js` — Axios API Client

Pre-configured Axios instance with:
- Base URL from `VITE_API_URL` environment variable
- **Request interceptor**: attaches JWT token from localStorage to every request
- **Response interceptor**: on 401 → auto-clears storage and redirects to login
- Exported functions: `googleLogin`, `uploadDocument`, `listDocuments`, `deleteDocument`, `ingestDocuments`, `queryRag`, `getRagStats`, `getOcrStatus`, `getProviders`

#### `pages/Login.jsx` — Sign-In Page

Split-screen layout:
- **Left panel**: branding, app title, 6 feature cards with icons (hidden on mobile)
- **Right panel**: Google Sign-In button using `@react-oauth/google` component + privacy notice

#### `pages/Dashboard.jsx` — Main Application Layout

Three-section layout: Navbar (top) + Sidebar (left) + ChatPanel (main area). Manages state for documents, vector stats, OCR availability, model selection, and chunk settings. Fetches data from API on component mount.

#### `components/Navbar.jsx` — Top Navigation Bar

Displays: sidebar toggle button, app name with "ENTERPRISE" badge, user avatar from Google profile, user name, logout button.

#### `components/Sidebar.jsx` — Left Panel

Contains six sections:
1. **Upload button** — file input accepting PDF, TXT, PNG, JPG, TIFF, BMP (supports multiple files)
2. **OCR toggle** — checkbox to enable OCR (only shown when Tesseract is detected on the server)
3. **Documents list** — filename, file size, delete button on hover. **Rejects files over 50 MB** with toast error before uploading
4. **Vector DB status** — green badge (vectors stored) or amber badge (empty)
5. **Ingest button** — triggers POST `/api/rag/ingest`
6. **Settings** — LLM provider dropdown (OpenRouter / Ollama), LLM model dropdown (populated from selected provider), chunk size slider (200–2000), overlap slider (0–500), top-K slider (1–10). Shows provider status badge (API key set / Ollama online).

#### `components/ChatPanel.jsx` — Chat Interface

Contains:
- **Stats bar** — colored badges showing document count, vector count, top-K, current model
- **Message list** — user messages (indigo, right-aligned), assistant messages (white card, left-aligned with bot avatar)
- **Source expander** — collapsible panel showing source chunk filename and text preview
- **Response time** — displayed below each assistant message
- **Typing indicator** — animated dots while waiting for API response
- **Empty state** — "Start a Conversation" prompt with clickable example question chips
- **Input area** — auto-growing textarea with Enter-to-send, Send button

---

### Prompt Templates (`prompts/`)

| Template | File | Style |
|----------|------|-------|
| **Strict Q&A** *(default)* | `prompts/strict/strict_qa.py` | Only answers from documents, says "not found" otherwise |
| **Strict Cited** | `prompts/strict/strict_cited.py` | Same + adds [Source: filename] citations |
| **Friendly** | `prompts/conversational/friendly.py` | Warm, conversational tone |
| **ELI5** | `prompts/conversational/eli5.py` | Explain Like I'm 5 |
| **Summarizer** | `prompts/analytical/summarizer.py` | Concise summaries |
| **Comparison** | `prompts/analytical/comparison.py` | Compare & contrast format |
| **Step-by-Step** | `prompts/analytical/step_by_step.py` | Numbered step reasoning |
| **Technical** | `prompts/domain/technical.py` | Technical documentation style |
| **Academic** | `prompts/domain/academic.py` | Scholarly tone |
| **Legal** | `prompts/domain/legal.py` | Legal document phrasing |

To switch prompts, edit the import in `src/generator.py`:
```python
from prompts.conversational.friendly import CONVERSATIONAL_PROMPT
PROMPT_TEMPLATE = CONVERSATIONAL_PROMPT
```

---

## Configuration & Customization

### LLM Providers

| Provider | Type | Key Required | Notes |
|----------|------|-------------|-------|
| **OpenRouter** | Cloud | `OPENROUTER_API_KEY` | Free models available — no credit card needed |
| **Ollama** | Local | None | Install from [ollama.com](https://ollama.com), no internet required |

Switch providers using the **LLM Provider** dropdown in the sidebar.

### Available OpenRouter Models (all free)

| Model | ID | Best For |
|-------|----|----------|
| GLM-4.5-Air | `z-ai/glm-4.5-air:free` | Best balance (default) |
| Step-3.5-Flash | `stepfun/step-3.5-flash:free` | Fast responses |
| GPT-OSS-120B | `openai/gpt-oss-120b:free` | Strong reasoning |
| Trinity-Mini | `arcee-ai/trinity-mini:free` | Efficient + long context |

### Available Ollama Models (local)

First install Ollama from [https://ollama.com](https://ollama.com), then pull a model:

```bash
ollama pull llama3.2   # recommended (3B, fast)
ollama pull llama3.1   # 8B, balanced
ollama pull mistral    # Mistral 7B
ollama pull phi3       # Microsoft Phi-3
```

Once Ollama is running (`ollama serve`), select **Ollama (Local)** in the sidebar — available models are listed automatically.

### Chunking Settings

| Setting | Default | Range | Effect |
|---------|---------|-------|--------|
| Chunk Size | 1000 chars | 200 – 2000 | Larger = more context per result |
| Chunk Overlap | 100 chars | 0 – 500 | More = smoother boundaries |
| Top-K Results | 3 | 1 – 10 | More = broader context for LLM |

Adjustable via sidebar sliders. Changes apply on next ingest (chunk settings) or next query (top-K).

### Supported File Types

| Type | Extension | OCR Support |
|------|-----------|-------------|
| PDF | `.pdf` | ✅ (auto-fallback if text < 50 chars) |
| Text | `.txt` | N/A |
| PNG | `.png` | ✅ |
| JPEG | `.jpg`, `.jpeg` | ✅ |
| TIFF | `.tiff` | ✅ |
| BMP | `.bmp` | ✅ |

---

## Troubleshooting

### "Access blocked: no registered origin" on Google Sign-In

Your Google OAuth client doesn't have the correct JavaScript origin. Go to **Google Cloud Console → Credentials → your OAuth client** and add:
- Authorized JavaScript origins: `http://localhost:5173`
- Authorized redirect URIs: `http://localhost:5173`

Wait 1–2 minutes after saving for Google to propagate.

### "deleted_client" error

You're using a Client ID from a deleted OAuth client. Create a **new Web application** OAuth client and update both `.env` files.

### CORS errors in browser console

Make sure the FastAPI backend is running on port 8000. The CORS middleware allows `http://localhost:5173`. If using a different port, update `backend/main.py`.

### `OPENROUTER_API_KEY not set`

Ensure `.env` exists in the project root with your key. Restart the backend after editing `.env`.

### Ollama not showing in sidebar / showing offline

1. Install Ollama from [https://ollama.com](https://ollama.com)
2. Start the server: `ollama serve` (or it starts automatically on Windows)
3. Pull at least one model: `ollama pull llama3.2`
4. The sidebar will show **Ollama (Local)** as available with your pulled models listed

### OCR checkbox not showing in sidebar

Tesseract is not installed on your system. See [Step 6](#step-6-optional-install-tesseract-for-ocr). The app checks Tesseract availability via the `/api/ocr/status` endpoint.

### `torch` / `torchvision` warnings on startup

Non-critical warnings from the sentence-transformers library file watcher. Everything works normally.

### Upload fails with 413 error

File exceeds the 50 MB limit. The frontend also checks client-side before uploading. Compress or split the document.

### "No documents uploaded" when trying to ingest

Upload documents first using the sidebar upload button, then click Ingest.

### Frontend `.env` changes not taking effect

Vite caches environment variables at build time. Stop the dev server (`Ctrl+C`) and restart: `npm run dev`.

### Google Sign-In popup closes immediately

- Check browser console for errors
- Verify `VITE_GOOGLE_CLIENT_ID` in `frontend/.env` matches your OAuth client
- Make sure you selected **Web application** type (not Desktop) when creating the OAuth client
- Ensure `http://localhost:5173` is in authorized origins (no trailing slash)

---

## Legacy Interfaces

### Streamlit UI

The original Streamlit web interface is still available:

```bash
# Windows
venv\Scripts\streamlit.exe run streamlit_app.py

# Linux/Mac
streamlit run streamlit_app.py
```

Opens at `http://localhost:8501`. Uses the shared `vectordb/` directory (not per-user isolated).

### CLI

The command-line interface is still available:

```bash
python app.py
```

Select provider at startup: `1` for OpenRouter, `2` for Ollama. Commands: `ingest`, `query`, `exit`. Uses the shared `data/` folder and `vectordb/`.

---

## License

This project is for educational purposes. Uses free tiers of OpenRouter and Google OAuth.