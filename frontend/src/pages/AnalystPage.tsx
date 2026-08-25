import React, { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { PageContainer } from '../components/layout/PageContainer';
import ChatMessage from '../components/analyst/ChatMessage';
import AnalystComposer from '../components/analyst/AnalystComposer';
import ConversationHistory from '../components/analyst/ConversationHistory';
import ErrorState from '../components/ui/ErrorState';
import Spinner from '../components/ui/Spinner';
import { useDataset } from '../context/DatasetContext';
import { useNotification } from '../context/NotificationContext';
import { sendConversationalMessage } from '../services/chatService';
import type { AnalystSession, ChatMessage as ChatMessageType } from '../types';
import { IconDatabase, IconBrain, IconTrendUp, IconLightbulb } from '../components/ui/Icons';

const EXAMPLE_PROMPTS = [
  'Analyze my data and identify key performance drivers',
  'Why did the primary metric change recently?',
  'Find unusual anomalies or correlation patterns',
  'What if key volume increases by 10%?',
  'Provide a summary report of findings',
];

const LOCAL_SESSIONS_KEY = 'auto_analyst_chat_sessions';

export default function AnalystPage() {
  const { profile, fileName } = useDataset();
  const { notify } = useNotification();

  const datasetName = fileName || profile?.dataset_name;

  // Session state
  const [sessions, setSessions] = useState<AnalystSession[]>(() => {
    try {
      const raw = localStorage.getItem(LOCAL_SESSIONS_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch {
      return [];
    }
  });

  const [currentSessionId, setCurrentSessionId] = useState<string>(() => {
    return `session_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
  });

  const [messages, setMessages] = useState<ChatMessageType[]>([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load messages when currentSessionId changes
  useEffect(() => {
    const found = sessions.find((s) => s.session_id === currentSessionId);
    if (found) {
      setMessages(found.messages);
    } else {
      setMessages([]);
    }
  }, [currentSessionId]);

  // Persist sessions to localStorage
  const saveSessionMessages = (newMessages: ChatMessageType[]) => {
    setMessages(newMessages);
    setSessions((prev) => {
      const now = new Date().toISOString();
      const existingIdx = prev.findIndex((s) => s.session_id === currentSessionId);
      let updated: AnalystSession[];

      if (existingIdx >= 0) {
        updated = prev.map((s, idx) =>
          idx === existingIdx
            ? { ...s, messages: newMessages, updated_at: now, dataset_name: datasetName || s.dataset_name }
            : s,
        );
      } else {
        const newSession: AnalystSession = {
          session_id: currentSessionId,
          dataset_name: datasetName,
          created_at: now,
          updated_at: now,
          messages: newMessages,
        };
        updated = [newSession, ...prev];
      }

      try {
        localStorage.setItem(LOCAL_SESSIONS_KEY, JSON.stringify(updated));
      } catch {
        // storage overflow fallback
      }
      return updated;
    });
  };

  // Auto-scroll to latest message
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const handleStartNewSession = () => {
    const newId = `session_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
    setCurrentSessionId(newId);
    setMessages([]);
    setInputMessage('');
    setError('');
    notify('Started a new analysis session.', 'info');
  };

  const handleSendMessage = async (textToSend?: string) => {
    const text = (textToSend || inputMessage).trim();
    if (!text || loading) return;

    setError('');
    setInputMessage('');

    const userMsg: ChatMessageType = {
      id: `usr_${Date.now()}`,
      role: 'user',
      content: text,
      timestamp: new Date().toISOString(),
    };

    const nextMessages = [...messages, userMsg];
    saveSessionMessages(nextMessages);
    setLoading(true);

    try {
      const apiResp = await sendConversationalMessage(
        text,
        currentSessionId,
        profile?.preview,
        datasetName,
      );

      const assistantMsg: ChatMessageType = {
        id: `asst_${Date.now()}`,
        role: 'assistant',
        content: apiResp.response,
        evidence: apiResp.evidence,
        metadata: apiResp.metadata,
        timestamp: apiResp.created_at || new Date().toISOString(),
      };

      saveSessionMessages([...nextMessages, assistantMsg]);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis request failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <PageContainer className="analyst-page" style={{ display: 'flex', flexDirection: 'column', minHeight: 'calc(100vh - 120px)' }}>
      {/* Top Context Bar */}
      <div
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          backgroundColor: 'rgba(255, 255, 255, 0.95)',
          border: '1px solid rgba(226, 232, 240, 0.9)',
          borderRadius: '14px',
          padding: '0.65rem 1rem',
          marginBottom: '1rem',
          boxShadow: '0 2px 10px rgba(15, 23, 42, 0.03)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <div
            style={{
              width: '32px',
              height: '32px',
              borderRadius: '8px',
              backgroundColor: 'rgba(99, 102, 241, 0.1)',
              color: 'var(--primary)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <IconDatabase size={16} />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <span style={{ fontSize: '0.88rem', fontWeight: 600, color: 'var(--ink)' }}>
                {profile ? `Analyzing: ${datasetName}` : 'No Dataset Selected'}
              </span>
              <span
                style={{
                  fontSize: '0.68rem',
                  fontWeight: 700,
                  textTransform: 'uppercase',
                  padding: '0.1rem 0.4rem',
                  borderRadius: '4px',
                  backgroundColor: '#ecfdf5',
                  color: '#059669',
                }}
              >
                ● Ready
              </span>
            </div>
            <p className="muted" style={{ margin: 0, fontSize: '0.74rem' }}>
              {profile ? `${profile.rows.toLocaleString()} rows · ${profile.columns} columns loaded` : 'Using system default reasoning context'}
            </p>
          </div>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          <Link
            to="/datasets"
            className="action-btn"
            style={{ padding: '0.3rem 0.65rem', fontSize: '0.78rem', textDecoration: 'none' }}
          >
            {profile ? 'Switch Dataset' : 'Select Dataset'}
          </Link>
          <button
            type="button"
            onClick={handleStartNewSession}
            className="ghost-text-btn"
            style={{ fontSize: '0.78rem' }}
          >
            + New Analysis
          </button>
        </div>
      </div>

      {/* Main Conversation Layout */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr minmax(200px, 240px)', gap: '1.25rem', flex: 1, alignItems: 'start' }}>
        {/* Chat Stream & Composer */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem', minHeight: '480px' }}>
          <div
            style={{
              flex: 1,
              minHeight: '380px',
              padding: '0.5rem 0',
              display: 'flex',
              flexDirection: 'column',
              justifyContent: messages.length === 0 ? 'center' : 'flex-start',
            }}
          >
            {messages.length === 0 ? (
              <div style={{ textAlign: 'center', padding: '2rem 1rem', maxWidth: '540px', margin: '0 auto' }}>
                <div
                  style={{
                    width: '48px',
                    height: '48px',
                    borderRadius: '14px',
                    backgroundColor: 'rgba(99, 102, 241, 0.1)',
                    color: 'var(--primary)',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    margin: '0 auto 1rem',
                  }}
                >
                  <IconBrain size={26} />
                </div>
                <h2 style={{ margin: '0 0 0.4rem', fontSize: '1.25rem', fontWeight: 700 }}>
                  Conversational AI Data Analyst
                </h2>
                <p className="muted" style={{ margin: '0 0 1.5rem', fontSize: '0.88rem' }}>
                  {profile
                    ? `Ready to explore ${datasetName}. Ask questions in natural language and receive verifiable, evidence-backed insights.`
                    : 'Ask questions in natural language and let the multi-agent AI decompose intent, formulate plans, and compute answers.'}
                </p>

                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem' }}>
                  <p className="muted" style={{ fontSize: '0.76rem', fontWeight: 600, textTransform: 'uppercase' }}>
                    Try a query:
                  </p>
                  {EXAMPLE_PROMPTS.map((prompt) => (
                    <button
                      key={prompt}
                      type="button"
                      onClick={() => handleSendMessage(prompt)}
                      className="analyst-chip"
                      style={{ textAlign: 'left', padding: '0.5rem 0.85rem' }}
                    >
                      ⚡ {prompt}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                {messages.map((msg) => (
                  <ChatMessage key={msg.id} message={msg} />
                ))}

                {/* Subtle thinking state */}
                {loading && (
                  <div style={{ display: 'flex', gap: '0.65rem', alignItems: 'center', margin: '1rem 0', color: 'var(--primary)' }}>
                    <Spinner size={18} />
                    <span style={{ fontSize: '0.86rem', fontWeight: 500 }}>
                      Analyzing your data with verifiable multi-agent reasoning…
                    </span>
                  </div>
                )}

                <div ref={messagesEndRef} />
              </div>
            )}
          </div>

          {error && (
            <ErrorState
              message={error}
              onRetry={() => {
                const lastUserMsg = [...messages].reverse().find((m) => m.role === 'user');
                if (lastUserMsg) handleSendMessage(lastUserMsg.content);
              }}
            />
          )}

          {/* Composer */}
          <div style={{ position: 'sticky', bottom: '0.5rem' }}>
            <AnalystComposer
              value={inputMessage}
              onChange={setInputMessage}
              onSubmit={() => handleSendMessage()}
              disabled={loading}
              placeholder={
                profile
                  ? `Ask anything about ${datasetName} (e.g. "Analyze top revenue drivers", "Why?", "What if pricing increases by 10%?")...`
                  : 'Ask your data anything (e.g. "Analyze performance", "Why?", "Simulate what happens if volume drops")...'
              }
            />
          </div>
        </div>

        {/* History sidebar (desktop) */}
        <div className="glass-card glass-card--padded" style={{ alignSelf: 'stretch' }}>
          <ConversationHistory
            sessions={sessions}
            currentSessionId={currentSessionId}
            onSelectSession={(sid) => setCurrentSessionId(sid)}
            onNewSession={handleStartNewSession}
          />
        </div>
      </div>
    </PageContainer>
  );
}
