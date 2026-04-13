import { useState, useRef, useEffect } from 'react'
import { queryRag } from '../api'
import toast from 'react-hot-toast'
import {
  Send,
  Bot,
  User,
  FileText,
  Clock,
  MessageSquare,
  ChevronDown,
  ChevronRight,
  Sparkles,
} from 'lucide-react'

export default function ChatPanel({ stats, model, provider, topK, documents }) {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const endRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  const handleSend = async () => {
    const question = input.trim()
    if (!question) return

    if (stats.vectors === 0) {
      toast.error('Please ingest documents first')
      return
    }

    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: question }])
    setLoading(true)

    try {
      const res = await queryRag(question, topK, model, provider)
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: res.data.answer,
          sources: res.data.sources,
          time: res.data.time_seconds,
        },
      ])
    } catch (err) {
      const detail = err.response?.data?.detail || 'Query failed'
      const status = err.response?.status
      // Friendly toast based on error type
      let toastMsg
      if (status === 429) {
        toastMsg = '⏳ Quota exhausted — switch to OpenRouter or wait for reset'
      } else if (status === 400 && detail.toLowerCase().includes('authentication')) {
        toastMsg = '🔑 API key invalid — check your .env file'
      } else if (status === 400 && detail.toLowerCase().includes('not found')) {
        toastMsg = '⚠️ Model not found — select a different model'
      } else {
        toastMsg = detail.length > 120 ? detail.slice(0, 117) + '...' : detail
      }
      toast.error(toastMsg, { duration: 6000 })
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: `❌ ${detail}`, error: true },
      ])
    }
    setLoading(false)
    inputRef.current?.focus()
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const examples = [
    'What is retrieval augmented generation?',
    'How do transformers work?',
    'What are the types of machine learning?',
  ]

  return (
    <main className="flex-1 flex flex-col min-w-0">
      {/* Stats bar */}
      <div className="bg-white border-b border-slate-100 px-6 py-2.5 flex items-center gap-6 shrink-0">
        <StatBadge label="Documents" value={documents.length} color="emerald" />
        <StatBadge label="Vectors" value={stats.vectors} color="blue" />
        <StatBadge label="Top-K" value={topK} color="purple" />
        <div className="ml-auto flex items-center gap-1.5 text-[11px] text-slate-400">
          <Sparkles size={12} />
          <span className="hidden sm:inline">
            {provider === 'ollama'
              ? `${model} (Local)`
              : model.split('/')[1]?.replace(':free', '')}
          </span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-6">
        {messages.length === 0 && !loading ? (
          <EmptyState examples={examples} onSelect={(q) => { setInput(q); inputRef.current?.focus() }} />
        ) : (
          <div className="max-w-3xl mx-auto space-y-6">
            {messages.map((msg, i) => (
              <Message key={i} msg={msg} />
            ))}
            {loading && <TypingIndicator />}
            <div ref={endRef} />
          </div>
        )}
      </div>

      {/* Input */}
      <div className="bg-white border-t border-slate-200 p-4 shrink-0">
        <div className="max-w-3xl mx-auto flex gap-2">
          <textarea
            ref={inputRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask a question about your documents..."
            rows={1}
            className="flex-1 resize-none rounded-xl border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-800 placeholder-slate-400 focus:border-brand-400 focus:ring-2 focus:ring-brand-100 outline-none transition"
          />
          <button
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="px-4 rounded-xl bg-brand-600 hover:bg-brand-700 disabled:bg-slate-200 text-white disabled:text-slate-400 transition flex items-center gap-1.5"
          >
            <Send size={16} />
          </button>
        </div>
      </div>
    </main>
  )
}

/* ── Sub-components ── */

function StatBadge({ label, value, color }) {
  const styles = {
    emerald: 'bg-emerald-50 text-emerald-700',
    blue: 'bg-blue-50 text-blue-700',
    purple: 'bg-purple-50 text-purple-700',
    amber: 'bg-amber-50 text-amber-700',
  }
  return (
    <div className={`px-3 py-1 rounded-full text-xs font-semibold ${styles[color]}`}>
      {value} <span className="font-normal opacity-70">{label}</span>
    </div>
  )
}

function Message({ msg }) {
  const [sourcesOpen, setSourcesOpen] = useState(false)
  const isUser = msg.role === 'user'

  return (
    <div className={`flex gap-3 animate-fade-in ${isUser ? 'justify-end' : ''}`}>
      {!isUser && (
        <div className="w-8 h-8 rounded-full bg-brand-100 text-brand-600 flex items-center justify-center shrink-0 mt-0.5">
          <Bot size={16} />
        </div>
      )}

      <div className={`max-w-[80%] ${isUser ? 'order-first' : ''}`}>
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? 'bg-brand-600 text-white rounded-tr-md'
              : msg.error
              ? 'bg-red-50 text-red-700 border border-red-200 rounded-tl-md'
              : 'bg-white text-slate-700 border border-slate-200 shadow-sm rounded-tl-md'
          }`}
        >
          <p className="whitespace-pre-wrap">{msg.content}</p>
        </div>

        {/* Sources */}
        {msg.sources?.length > 0 && (
          <div className="mt-1.5">
            <button
              onClick={() => setSourcesOpen(!sourcesOpen)}
              className="flex items-center gap-1 text-[11px] text-slate-400 hover:text-brand-600 transition"
            >
              {sourcesOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              <FileText size={11} />
              {msg.sources.length} source{msg.sources.length > 1 ? 's' : ''}
            </button>
            {sourcesOpen && (
              <div className="mt-1.5 space-y-1.5">
                {msg.sources.map((src, i) => (
                  <div
                    key={i}
                    className="bg-slate-50 border border-slate-100 border-l-2 border-l-brand-400 rounded-lg px-3 py-2 text-xs text-slate-600"
                  >
                    <span className="font-semibold text-brand-600">Chunk {i + 1}</span>
                    <span className="text-slate-400"> — {src.filename}</span>
                    <p className="mt-1 text-slate-500 leading-relaxed">{src.preview}</p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Time */}
        {msg.time && (
          <div className="flex items-center gap-1 mt-1 text-[10px] text-slate-400">
            <Clock size={10} />
            {msg.time.toFixed(1)}s
          </div>
        )}
      </div>

      {isUser && (
        <div className="w-8 h-8 rounded-full bg-slate-200 text-slate-600 flex items-center justify-center shrink-0 mt-0.5">
          <User size={16} />
        </div>
      )}
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="flex gap-3 animate-fade-in">
      <div className="w-8 h-8 rounded-full bg-brand-100 text-brand-600 flex items-center justify-center shrink-0">
        <Bot size={16} />
      </div>
      <div className="bg-white border border-slate-200 shadow-sm rounded-2xl rounded-tl-md px-4 py-3 flex items-center gap-1.5">
        <span className="typing-dot" />
        <span className="typing-dot" />
        <span className="typing-dot" />
      </div>
    </div>
  )
}

function EmptyState({ examples, onSelect }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center px-4">
      <div className="text-5xl mb-4">💬</div>
      <h3 className="text-xl font-bold text-slate-800 mb-2">Start a Conversation</h3>
      <p className="text-sm text-slate-500 max-w-md mb-8">
        Ask any question about your uploaded documents. The AI searches through your data
        and provides grounded answers with source citations.
      </p>
      <div className="flex flex-wrap justify-center gap-2">
        {examples.map((q) => (
          <button
            key={q}
            onClick={() => onSelect(q)}
            className="px-4 py-2 bg-white border border-slate-200 rounded-full text-xs text-slate-600 hover:border-brand-300 hover:text-brand-600 hover:shadow-sm transition"
          >
            💡 {q}
          </button>
        ))}
      </div>
    </div>
  )
}
