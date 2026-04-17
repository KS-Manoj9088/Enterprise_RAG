import { useState, useRef } from 'react'
import toast from 'react-hot-toast'
import {
  Upload,
  FileText,
  Trash2,
  Database,
  Settings,
  RefreshCw,
  ScanLine,
  File,
} from 'lucide-react'
import {
  uploadDocument,
  deleteDocument,
  ingestDocuments,
} from '../api'

const MAX_SIZE = 50 * 1024 * 1024 // 50 MB

// Fallback models when the API hasn't loaded yet
const FALLBACK_MODELS = {
  openrouter: {
    'z-ai/glm-4.5-air:free': 'GLM-4.5-Air (Best Balance)',
    'stepfun/step-3.5-flash:free': 'Step-3.5-Flash (Fast)',
    'openai/gpt-oss-120b:free': 'GPT-OSS-120B (Strong)',
    'arcee-ai/trinity-mini:free': 'Trinity-Mini (Efficient)',
  },
  ollama: {},
}

const DEFAULT_MODEL = {
  openrouter: 'z-ai/glm-4.5-air:free',
  ollama: 'llama3.2',
}

export default function Sidebar({
  documents,
  stats,
  ocrAvailable,
  model,
  setModel,
  provider,
  setProvider,
  providersData,
  chunkSize,
  setChunkSize,
  chunkOverlap,
  setChunkOverlap,
  topK,
  setTopK,
  promptStyle,
  setPromptStyle,
  onRefresh,
}) {
  const fileInputRef = useRef(null)
  const [uploading, setUploading] = useState(false)
  const [ingesting, setIngesting] = useState(false)
  const [useOcr, setUseOcr] = useState(false)

  const handleUpload = async (e) => {
    const files = e.target.files
    if (!files?.length) return

    for (const file of files) {
      if (file.size > MAX_SIZE) {
        toast.error(`"${file.name}" exceeds 50 MB limit (${(file.size / 1024 / 1024).toFixed(1)} MB)`)
        continue
      }

      setUploading(true)
      try {
        await uploadDocument(file, useOcr)
        toast.success(`Uploaded: ${file.name}`)
      } catch (err) {
        toast.error(err.response?.data?.detail || `Upload failed: ${file.name}`)
      }
    }
    setUploading(false)
    fileInputRef.current.value = ''
    onRefresh()
  }

  const handleDelete = async (docId, name) => {
    if (!window.confirm(`Delete "${name}"?`)) return
    try {
      await deleteDocument(docId)
      toast.success('Deleted')
      onRefresh()
    } catch (err) {
      toast.error('Delete failed')
    }
  }

  const handleIngest = async () => {
    if (!documents.length) {
      toast.error('Upload documents first')
      return
    }
    setIngesting(true)
    try {
      const res = await ingestDocuments(chunkSize, chunkOverlap, model, provider)
      toast.success(`Ingested: ${res.data.chunks_created} chunks → ${res.data.vectors_stored} vectors`)
      onRefresh()
    } catch (err) {
      toast.error(err.response?.data?.detail || 'Ingestion failed')
    }
    setIngesting(false)
  }

  const formatSize = (bytes) => {
    if (bytes >= 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`
    return `${(bytes / 1024).toFixed(1)} KB`
  }

  return (
    <aside className="w-80 bg-white border-r border-slate-200 flex flex-col overflow-y-auto shrink-0">
      {/* Upload */}
      <div className="p-4 border-b border-slate-100">
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt,.png,.jpg,.jpeg,.tiff,.bmp"
          multiple
          className="hidden"
          onChange={handleUpload}
        />
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={uploading}
          className="w-full flex items-center justify-center gap-2 bg-brand-600 hover:bg-brand-700 disabled:bg-brand-300 text-white rounded-xl py-2.5 text-sm font-semibold transition"
        >
          {uploading ? (
            <RefreshCw size={16} className="animate-spin" />
          ) : (
            <Upload size={16} />
          )}
          {uploading ? 'Uploading...' : 'Upload Document'}
        </button>
        <p className="text-[11px] text-slate-400 text-center mt-1.5">
          PDF, TXT, Images — Max 50 MB
        </p>

        {ocrAvailable && (
          <label className="flex items-center gap-2 mt-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useOcr}
              onChange={(e) => setUseOcr(e.target.checked)}
              className="rounded border-slate-300 text-brand-600 focus:ring-brand-500"
            />
            <ScanLine size={14} className="text-amber-500" />
            <span className="text-xs text-slate-600">Enable OCR for scanned docs</span>
          </label>
        )}
      </div>

      {/* Documents */}
      <div className="p-4 border-b border-slate-100">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-1.5">
          <FileText size={12} /> Documents ({documents.length})
        </h3>
        {documents.length === 0 ? (
          <p className="text-xs text-slate-400 italic">No documents uploaded yet</p>
        ) : (
          <div className="space-y-1.5 max-h-64 overflow-y-auto pr-1">
            {documents.map((doc) => (
              <div
                key={doc.id}
                className="group flex items-start gap-2 px-2.5 py-2 rounded-lg hover:bg-slate-50 transition"
              >
                <File size={14} className="text-slate-400 shrink-0 mt-0.5" />
                <div className="flex-1 min-w-0">
                  <p
                    className="text-xs font-medium text-slate-700 break-all leading-snug"
                    title={doc.filename}
                  >
                    {doc.filename}
                  </p>
                  <p className="text-[10px] text-slate-400 mt-0.5">
                    {formatSize(doc.size_bytes)}
                    {doc.ocr_used && ' · OCR'}
                  </p>
                </div>
                <button
                  onClick={() => handleDelete(doc.id, doc.filename)}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-red-50 text-slate-300 hover:text-red-500 transition shrink-0"
                >
                  <Trash2 size={12} />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Vector DB */}
      <div className="p-4 border-b border-slate-100">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1.5">
          <Database size={12} /> Vector DB
        </h3>
        <div
          className={`rounded-lg px-3 py-2 text-xs font-semibold flex items-center gap-2 ${
            stats.vectors > 0
              ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
              : 'bg-amber-50 text-amber-700 border border-amber-200'
          }`}
        >
          {stats.vectors > 0 ? '✅' : '⚠️'}
          {stats.vectors > 0 ? `${stats.vectors} vectors stored` : 'Empty — ingest first'}
        </div>

        <button
          onClick={handleIngest}
          disabled={ingesting || !documents.length}
          className="mt-2 w-full flex items-center justify-center gap-2 bg-emerald-600 hover:bg-emerald-700 disabled:bg-slate-200 disabled:text-slate-400 text-white rounded-lg py-2 text-xs font-semibold transition"
        >
          {ingesting ? (
            <RefreshCw size={14} className="animate-spin" />
          ) : (
            <RefreshCw size={14} />
          )}
          {ingesting ? 'Ingesting...' : 'Ingest Documents'}
        </button>
      </div>

      {/* Settings */}
      <div className="p-4">
        <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3 flex items-center gap-1.5">
          <Settings size={12} /> Settings
        </h3>

        {/* Provider */}
        <label className="block mb-3">
          <span className="text-[11px] font-medium text-slate-500">LLM Provider</span>
          <select
            value={provider}
            onChange={(e) => {
              const newProvider = e.target.value
              setProvider(newProvider)
              // Reset to first model of new provider
              const prov = providersData.find((p) => p.id === newProvider)
              const models = Object.keys(prov?.models || FALLBACK_MODELS[newProvider] || {})
              setModel(models[0] || DEFAULT_MODEL[newProvider] || '')
            }}
            className="mt-1 block w-full rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs text-slate-700 focus:border-brand-400 focus:ring-1 focus:ring-brand-200 outline-none"
          >
            {providersData.length > 0
              ? providersData.map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.name}
                  </option>
                ))
              : (
                <>
                  <option value="openrouter">OpenRouter (Cloud)</option>
                  <option value="ollama">Ollama (Local)</option>
                </>
              )}
          </select>
        </label>

        {/* Provider status banner */}
        {(() => {
          const prov = providersData.find((p) => p.id === provider)
          if (!prov) return null
          if (provider === 'ollama') {
            return (
              <div className={`mb-3 rounded-lg px-2.5 py-1.5 text-[11px] font-semibold flex items-center gap-1.5 ${
                prov.available
                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                  : 'bg-red-50 text-red-600 border border-red-200'
              }`}>
                {prov.available ? '● Ollama connected' : '● Ollama offline — start it first'}
              </div>
            )
          }
          if (prov.requires_key) {
            return (
              <div className={`mb-3 rounded-lg px-2.5 py-1.5 text-[11px] font-semibold flex items-center gap-1.5 ${
                prov.key_set
                  ? 'bg-emerald-50 text-emerald-700 border border-emerald-200'
                  : 'bg-amber-50 text-amber-700 border border-amber-200'
              }`}>
                {prov.key_set
                  ? `● API key set`
                  : `⚠ Set ${prov.key_env} in .env`}
              </div>
            )
          }
          return null
        })()}

        {/* Model */}
        <label className="block mb-3">
          <span className="text-[11px] font-medium text-slate-500">LLM Model</span>
          {(() => {
            const prov = providersData.find((p) => p.id === provider)
            const models = prov?.models
              ? Object.entries(prov.models)
              : Object.entries(FALLBACK_MODELS[provider] || {})
            const disabled = provider === 'ollama' && (!prov?.available || models.length === 0)
            return (
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                disabled={disabled}
                className="mt-1 block w-full rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs text-slate-700 focus:border-brand-400 focus:ring-1 focus:ring-brand-200 outline-none disabled:opacity-50"
              >
                {models.length > 0 ? (
                  models.map(([value, label]) => (
                    <option key={value} value={value}>{label}</option>
                  ))
                ) : (
                  <option value="">No models — run: ollama pull llama3.2</option>
                )}
              </select>
            )
          })()}
        </label>

        {/* Chunk size */}
        <label className="block mb-2">
          <div className="flex justify-between">
            <span className="text-[11px] font-medium text-slate-500">Chunk Size</span>
            <span className="text-[11px] text-brand-600 font-semibold">{chunkSize}</span>
          </div>
          <input
            type="range"
            min={200}
            max={2000}
            step={100}
            value={chunkSize}
            onChange={(e) => setChunkSize(+e.target.value)}
            className="mt-1 w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-brand-600"
          />
        </label>

        {/* Overlap */}
        <label className="block mb-2">
          <div className="flex justify-between">
            <span className="text-[11px] font-medium text-slate-500">Overlap</span>
            <span className="text-[11px] text-brand-600 font-semibold">{chunkOverlap}</span>
          </div>
          <input
            type="range"
            min={0}
            max={500}
            step={50}
            value={chunkOverlap}
            onChange={(e) => setChunkOverlap(+e.target.value)}
            className="mt-1 w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-brand-600"
          />
        </label>

        {/* Top-K */}
        <label className="block mb-3">
          <div className="flex justify-between">
            <span className="text-[11px] font-medium text-slate-500">Top-K</span>
            <span className="text-[11px] text-brand-600 font-semibold">{topK}</span>
          </div>
          <input
            type="range"
            min={1}
            max={10}
            step={1}
            value={topK}
            onChange={(e) => setTopK(+e.target.value)}
            className="mt-1 w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-brand-600"
          />
        </label>

        {/* Response Style */}
        <label className="block">
          <span className="text-[11px] font-medium text-slate-500">Response Style</span>
          <select
            value={promptStyle}
            onChange={(e) => setPromptStyle(e.target.value)}
            className="mt-1 block w-full rounded-lg border border-slate-200 bg-slate-50 px-2.5 py-1.5 text-xs text-slate-700 focus:border-brand-400 focus:ring-1 focus:ring-brand-200 outline-none"
          >
            <option value="strict">Strict Q&amp;A</option>
            <option value="general">General</option>
          </select>
        </label>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-slate-100 mt-auto">
        <p className="text-[10px] text-slate-400 text-center">
          LangChain · ChromaDB
          <br />
          OpenRouter · Ollama
        </p>
      </div>
    </aside>
  )
}
