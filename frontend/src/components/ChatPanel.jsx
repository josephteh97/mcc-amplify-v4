import React, { useState, useEffect, useRef, useCallback } from 'react';

// Stable user-id for this browser session.
// crypto.randomUUID() requires HTTPS; fall back for HTTP LAN access.
const USER_ID = (() => {
  try { return crypto.randomUUID(); }
  catch {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
      const r = (Math.random() * 16) | 0;
      return (c === 'x' ? r : (r & 0x3) | 0x8).toString(16);
    });
  }
})();

// Short label shown on the selector buttons
const MODEL_SHORT = {
  qwen3_vl:  'Qwen3-VL',
  gemma3_it: 'Gemma 3',
};

const ChatPanel = ({ jobId }) => {
  const [messages, setMessages]           = useState([]);
  const [input, setInput]                 = useState('');
  const [connected, setConnected]         = useState(false);
  const [typing, setTyping]               = useState(false);
  const [models, setModels]               = useState([]);
  const [selectedModel, setSelectedModel] = useState(null);

  const wsRef          = useRef(null);
  const messagesEndRef = useRef(null);
  const reconnectTimer = useRef(null);

  const scrollBottom = () =>
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });

  useEffect(() => { scrollBottom(); }, [messages]);

  // ── Fetch available models on mount ────────────────────────────────────────
  useEffect(() => {
    fetch('/api/chat/models')
      .then(r => r.json())
      .then(data => {
        setModels(data.models || []);
        setSelectedModel(data.default || 'qwen3_vl');
      })
      .catch(() => {
        // Fallback if the endpoint is unreachable during startup
        setModels([
          { backend: 'qwen3_vl',  display_name: 'qwen3-vl:2b',       provider: 'Ollama', available: true },
          { backend: 'gemma3_it', display_name: 'gemma3:4b-it-qat',  provider: 'Ollama', available: true },
        ]);
        setSelectedModel('qwen3_vl');
      });
  }, []);

  // ── WebSocket connection ────────────────────────────────────────────────────
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const ws    = new WebSocket(`${proto}://${window.location.host}/ws/chat/${USER_ID}`);
    wsRef.current = ws;

    ws.onopen = () => {
      setConnected(true);
      clearTimeout(reconnectTimer.current);
    };

    ws.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        if (data.type === 'agent_message') {
          setTyping(false);
          setMessages(prev => [
            ...prev,
            { id: Date.now(), text: data.message, sender: 'bot' },
          ]);
        }
      } catch { /* ignore malformed frames */ }
    };

    ws.onclose = () => {
      setConnected(false);
      reconnectTimer.current = setTimeout(connect, 3000);
    };

    ws.onerror = () => ws.close();
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  // ── Send message ───────────────────────────────────────────────────────────
  const handleSend = () => {
    const text = input.trim();
    if (!text || !connected) return;

    setMessages(prev => [...prev, { id: Date.now(), text, sender: 'user' }]);
    setInput('');
    setTyping(true);

    wsRef.current.send(JSON.stringify({
      type:    'user_message',
      message: text,
      context: {
        job_id: jobId || null,
        model:  selectedModel,   // tells backend which LLM to use for this message
      },
    }));
  };

  // Short label for the active model
  const activeLabel = MODEL_SHORT[selectedModel] || selectedModel || '…';

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col h-full bg-white rounded-2xl shadow-2xl overflow-hidden border border-slate-200">

      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-blue-600 px-3 py-2.5 flex items-center justify-between shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-base">🤖</span>
          <h3 className="text-xs font-bold text-white">AI Supervisor</h3>
        </div>

        <div className="flex items-center gap-1.5">
          {/* Model selector — compact pill toggle */}
          {models.length > 0 && (
            <div className="flex items-center bg-black/25 rounded-md p-0.5 gap-0.5">
              {models.map(m => (
                <button
                  key={m.backend}
                  onClick={() => m.available && setSelectedModel(m.backend)}
                  disabled={!m.available}
                  title={m.available ? m.display_name : `${m.display_name} — key not configured`}
                  className={[
                    'text-[9px] font-bold px-2 py-0.5 rounded transition-all',
                    selectedModel === m.backend
                      ? 'bg-white/25 text-white shadow-sm'
                      : 'text-white/50 hover:text-white/75',
                    !m.available ? 'opacity-30 cursor-not-allowed' : 'cursor-pointer',
                  ].join(' ')}
                >
                  {MODEL_SHORT[m.backend] || m.backend}
                </button>
              ))}
            </div>
          )}

          {/* Connection + active model badge */}
          <span className={`text-[9px] font-semibold px-2 py-0.5 rounded-full whitespace-nowrap ${
            connected
              ? 'bg-emerald-400/30 text-emerald-100'
              : 'bg-red-400/30 text-red-100'
          }`}>
            {connected ? `${activeLabel} · live` : 'reconnecting…'}
          </span>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-3 space-y-3 bg-slate-50">
        {messages.length === 0 && (
          <p className="text-center text-slate-400 text-xs pt-6">
            {connected
              ? `Ask ${activeLabel} anything about your floor plan…`
              : 'Connecting to AI…'}
          </p>
        )}

        {messages.map(msg => (
          <div
            key={msg.id}
            className={`flex items-end gap-2 ${msg.sender === 'user' ? 'flex-row-reverse' : ''}`}
          >
            <div className={`w-6 h-6 rounded-full flex items-center justify-center shrink-0 text-xs
              ${msg.sender === 'user' ? 'bg-indigo-600 text-white' : 'bg-emerald-500 text-white'}`}
            >
              {msg.sender === 'user' ? '👤' : '🤖'}
            </div>
            <div className={`max-w-[80%] px-3 py-2 rounded-2xl text-xs leading-relaxed whitespace-pre-wrap
              ${msg.sender === 'user'
                ? 'bg-indigo-600 text-white rounded-br-none'
                : 'bg-white border border-slate-200 text-slate-700 rounded-bl-none shadow-sm'
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}

        {/* Typing indicator */}
        {typing && (
          <div className="flex items-end gap-2">
            <div className="w-6 h-6 rounded-full bg-emerald-500 flex items-center justify-center text-xs text-white shrink-0">🤖</div>
            <div className="bg-white border border-slate-200 rounded-2xl rounded-bl-none px-3 py-2 shadow-sm flex gap-1 items-center">
              <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:0ms]" />
              <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:150ms]" />
              <span className="w-1.5 h-1.5 bg-slate-400 rounded-full animate-bounce [animation-delay:300ms]" />
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <form
        onSubmit={e => { e.preventDefault(); handleSend(); }}
        className="flex items-center gap-2 p-3 bg-white border-t border-slate-100 shrink-0"
      >
        <input
          type="text"
          value={input}
          onChange={e => setInput(e.target.value)}
          placeholder={connected ? `Ask ${activeLabel} about your floor plan…` : 'Connecting…'}
          disabled={!connected}
          className="flex-1 bg-slate-100 rounded-lg px-3 py-2 text-xs outline-none focus:ring-2 focus:ring-indigo-400 disabled:opacity-50"
        />
        <button
          type="submit"
          disabled={!input.trim() || !connected}
          className="w-8 h-8 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white rounded-lg flex items-center justify-center transition-colors text-sm"
        >
          ➤
        </button>
      </form>
    </div>
  );
};

export default ChatPanel;
