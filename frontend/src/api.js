import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || ''

const api = axios.create({
  baseURL: API_BASE,
  headers: { 'Content-Type': 'application/json' },
})

// Attach JWT token to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('rag_token')
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})

// Handle 401 → logout
api.interceptors.response.use(
  (res) => res,
  (err) => {
    if (err.response?.status === 401) {
      localStorage.removeItem('rag_token')
      localStorage.removeItem('rag_user')
      window.location.href = '/'
    }
    return Promise.reject(err)
  },
)

// ── Auth ──
export const googleLogin = (credential) =>
  api.post('/api/auth/google', { credential })

export const getMe = () => api.get('/api/auth/me')

// ── Documents ──
export const uploadDocument = (file, useOcr = false) => {
  const formData = new FormData()
  formData.append('file', file)
  return api.post(`/api/documents/upload?use_ocr=${useOcr}`, formData, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export const listDocuments = () => api.get('/api/documents/')

export const deleteDocument = (docId) => api.delete(`/api/documents/${docId}`)

// ── RAG ──
export const ingestDocuments = (chunkSize = 1000, chunkOverlap = 100, model = 'z-ai/glm-4.5-air:free', provider = 'openrouter') =>
  api.post('/api/rag/ingest', {
    chunk_size: chunkSize,
    chunk_overlap: chunkOverlap,
    model,
    provider,
  })

export const queryRag = (question, topK = 3, model = 'z-ai/glm-4.5-air:free', provider = 'openrouter', promptStyle = 'auto') =>
  api.post('/api/rag/query', { question, top_k: topK, model, provider, prompt_style: promptStyle })

export const getRagStats = () => api.get('/api/rag/stats')

// ── LLM Providers ──
export const getProviders = () => api.get('/api/rag/providers')

// ── OCR ──
export const getOcrStatus = () => api.get('/api/ocr/status')

// ── Health ──
export const getHealth = () => api.get('/api/health')

export default api