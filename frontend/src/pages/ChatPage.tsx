import React, { useRef, useState } from 'react';
import { sendChatMessage } from '../services/chatService';
import type { ChatResponse } from '../services/chatService';
import PlotlyChart from '../components/PlotlyChart';

type HistoryEntry = {
  role: 'user' | 'assistant';
  content: string;
};

export default function ChatPage() {
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState('');
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [working, setWorking] = useState(false);
  const [error, setError] = useState('');
  const [lastViz, setLastViz] = useState<ChatResponse['visualization']>(null);
  const suggested = useRef<string[]>([]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setFile(e.target.files?.[0] ?? null);
    setError('');
  };

  const handleSend = async () => {
    if (!file) {
      setError('Please select a dataset file first.');
      return;
    }
    if (!message.trim()) {
      setError('Please enter a question.');
      return;
    }
    setWorking(true);
    setError('');
    const userEntry: HistoryEntry = { role: 'user', content: message };
    setHistory((h) => [...h, userEntry]);

    try {
      const res = await sendChatMessage(file, message);
      const assistantEntry: HistoryEntry = { role: 'assistant', content: res.message };
      setHistory((h) => [...h, assistantEntry]);
      setLastViz(res.visualization);
      suggested.current = res.suggested_questions ?? [];
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Chat request failed');
    } finally {
      setWorking(false);
      setMessage('');
    }
  };

  return (
    <div className="page-stack chat-page">
      <header>
        <h1>Chat</h1>
        <p className="muted">
          Upload a dataset and ask questions about it. Example: "What was the highest sales value?"
        </p>
      </header>

      <section className="card">
        <div className="field">
          <label htmlFor="chat-file">Dataset file</label>
          <input
            id="chat-file"
            type="file"
            accept=".csv,.xlsx,.xls"
            onChange={handleFileChange}
            disabled={working}
          />
        </div>
        {error ? <div className="status-error">{error}</div> : null}
        <div className="chat-input-row">
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Ask a question about the dataset…"
            disabled={working}
            onKeyDown={(e) => e.key === 'Enter' && !working && handleSend()}
            style={{ flex: 1, minWidth: 0 }}
          />
          <button className="primary-btn" onClick={handleSend} disabled={working} style={{ width: 'auto' }}>
            {working ? 'Thinking…' : 'Send'}
          </button>
        </div>
        {suggested.current.length ? (
          <div className="chip-row">
            {suggested.current.map((q) => (
              <button key={q} className="action-btn" onClick={() => setMessage(q)} type="button">
                {q}
              </button>
            ))}
          </div>
        ) : null}
      </section>

      {lastViz ? (
        <section className="card">
          <h2>Visualization</h2>
          <PlotlyChart data={lastViz.data as any} layout={lastViz.layout} />
        </section>
      ) : null}

      <section className="card">
        <h2>Conversation</h2>
        {history.length === 0 ? (
          <p className="muted">No messages yet.</p>
        ) : (
          <ul className="chat-history">
            {history.map((entry, i) => (
              <li key={i} className={`chat-bubble chat-${entry.role}`}>
                <strong>{entry.role === 'user' ? 'You' : 'Assistant'}:</strong> {entry.content}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
