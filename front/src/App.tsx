import React, { useState, useRef, useEffect } from 'react';
import { Send, Upload, FileText, Loader2, BookOpen, Bot, Zap } from 'lucide-react';
import { ingestDocument, sendChatQuery, type ChatResponse, type SourceInfo, type ChatMode } from './api';
import './App.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceInfo[];
  mode?: ChatMode; // 답변이 생성된 모드 표시
}

function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [chatMode, setChatMode] = useState<ChatMode>('simple'); // 채팅 모드 상태

  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      const result = await ingestDocument(file);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: `✅ 문서 "${result.filename}"가 성공적으로 업로드되었습니다. (${result.chunks_count} chunks)`,
        },
      ]);
    } catch (error) {
      console.error('Upload failed:', error);
      alert('문서 업로드 중 오류가 발생했습니다.');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = input;
    const currentMode = chatMode;

    setInput('');
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      // 선택된 모드로 API 호출
      const response: ChatResponse = await sendChatQuery(userMessage, currentMode);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: response.answer,
          sources: response.sources,
          mode: currentMode,
        },
      ]);
    } catch (error) {
      console.error('Chat failed:', error);
      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: '죄송합니다. 답변을 생성하는 중 오류가 발생했습니다.',
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1>📑 RAG Document Chat</h1>
        <div className="header-actions">
          {/* 모드 선택 토글 */}
          <div className="mode-toggle">
            <button
              className={`mode-btn ${chatMode === 'simple' ? 'active' : ''}`}
              onClick={() => setChatMode('simple')}
              title="빠른 응답, 단순 검색"
            >
              <Zap size={16} /> Simple
            </button>
            <button
              className={`mode-btn ${chatMode === 'agentic' ? 'active' : ''}`}
              onClick={() => setChatMode('agentic')}
              title="심층 추론, 에이전트 검색"
            >
              <Bot size={16} /> Agentic
            </button>
          </div>

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileUpload}
            accept=".pdf"
            style={{ display: 'none' }}
          />
          <button
            className="upload-btn"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
          >
            {isUploading ? (
              <Loader2 className="icon spin" />
            ) : (
              <Upload className="icon" />
            )}
            Upload PDF
          </button>
        </div>
      </header>

      <main className="chat-container">
        {messages.length === 0 ? (
          <div className="empty-state">
            <FileText className="empty-icon" />
            <h2>문서를 업로드하고 질문해 보세요!</h2>
            <p>PDF 파일을 업로드하면 AI가 내용을 분석하여 답변해 드립니다.</p>
            <div className="mode-info">
              <span className="badge simple"><Zap size={14}/> Simple Mode</span>: 빠른 검색과 답변
              <span className="badge agentic"><Bot size={14}/> Agentic Mode</span>: 에이전트 기반 심층 분석
            </div>
          </div>
        ) : (
          <div className="messages-list">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.role}`}>
                <div className="message-content">
                  {msg.role === 'assistant' && msg.mode && (
                    <div className={`mode-badge ${msg.mode}`}>
                      {msg.mode === 'simple' ? <Zap size={12}/> : <Bot size={12}/>}
                      {msg.mode === 'simple' ? 'Simple' : 'Agentic'}
                    </div>
                  )}
                  <p>{msg.content}</p>
                  
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="sources-section">
                      <h4><BookOpen className="icon-small" /> 참고 문헌</h4>
                      <div className="sources-list">
                        {msg.sources.map((source, idx) => (
                          <div key={idx} className="source-item">
                            <span className="source-title">{source.source} (p.{source.page})</span>
                            <p className="source-preview">{source.content}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="message assistant">
                <div className="message-content loading">
                  <Loader2 className="icon spin" /> 
                  {chatMode === 'agentic' ? '에이전트가 생각 중입니다...' : '답변 생성 중...'}
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </main>

      <form className="input-area" onSubmit={handleSubmit}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={chatMode === 'agentic' ? "복잡한 질문도 가능합니다..." : "빠르게 질문하세요..."}
          disabled={isLoading}
        />
        <button type="submit" disabled={!input.trim() || isLoading}>
          <Send className="icon" />
        </button>
      </form>
    </div>
  );
}

export default App;
