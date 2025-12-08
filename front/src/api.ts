import axios from 'axios';

// API 클라이언트 설정
const api = axios.create({
  baseURL: '/api', // Proxy를 통해 백엔드로 전달됨
  headers: {
    'Content-Type': 'application/json',
  },
});

export interface SourceInfo {
  source: string;
  page: number;
  content: string;
}

export interface ChatResponse {
  answer: string;
  sources: SourceInfo[];
}

export interface IngestResponse {
  status: string;
  filename: string;
  chunks_count: number;
  message: string;
}

// 문서 업로드 API
export const ingestDocument = async (file: File): Promise<IngestResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<IngestResponse>('/ingest', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
};

// 채팅 질의 API
export type ChatMode = 'simple' | 'agentic';

export const sendChatQuery = async (query: string, mode: ChatMode = 'simple'): Promise<ChatResponse> => {
  const response = await api.post<ChatResponse>(`/chat/${mode}`, { query });
  return response.data;
};
