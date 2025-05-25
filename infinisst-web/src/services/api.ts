import axios from 'axios';
import type { AxiosInstance } from 'axios';

const axiosInstance: AxiosInstance = axios.create({
  baseURL: process.env.REACT_APP_API_URL || 'http://localhost:8001',
  timeout: 60000, // 60 second timeout for all requests
});

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

export interface Translation {
  time: number;
  text: string;
}

export interface InitSessionResponse {
  session_id: string;
  queued: boolean;
  queue_position: number;
  initializing?: boolean;
  error?: string;
}

export interface QueueStatusResponse {
  session_id: string;
  status: 'active' | 'initializing' | 'queued' | 'not_found';
  queued: boolean;
  queue_position: number;
  error?: string;
}

class ApiService {
  private clientId: string;

  constructor() {
    // Generate a unique client ID for this browser tab
    this.clientId = this.generateClientId();
  }

  private generateClientId(): string {
    return 'client_' + Math.random().toString(36).substring(2, 15) + Math.random().toString(36).substring(2, 15);
  }

  // Initialize a translation session
  async initSession(
    agentType: string = 'InfiniSST',
    languagePair: string = 'English -> Chinese',
    latencyMultiplier: number = 2
  ): Promise<InitSessionResponse> {
    try {
      const params = new URLSearchParams({
        agent_type: agentType,
        language_pair: languagePair,
        latency_multiplier: latencyMultiplier.toString(),
        client_id: this.clientId
      });

      const response = await axiosInstance.post(`/init?${params}`);
      return response.data;
    } catch (error) {
      console.error('Error initializing session:', error);
      throw error;
    }
  }

  // Check queue status for a session
  async checkQueueStatus(sessionId: string): Promise<QueueStatusResponse> {
    try {
      const response = await axiosInstance.get(`/queue_status/${sessionId}`);
      return response.data;
    } catch (error) {
      console.error('Error checking queue status:', error);
      throw error;
    }
  }

  // Send ping to keep session alive
  async sendPing(sessionId: string): Promise<void> {
    try {
      const params = new URLSearchParams({
        session_id: sessionId
      });

      await axiosInstance.post(`/ping?${params}`);
    } catch (error) {
      console.error('Error sending ping:', error);
      // Don't throw error for ping failures
    }
  }

  // Upload audio file for translation
  async uploadFile(sessionId: string, file: File): Promise<any> {
    try {
      const formData = new FormData();
      formData.append('file', file);
      formData.append('session_id', sessionId);

      const response = await axiosInstance.post(`/upload`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data;
    } catch (error) {
      console.error('Error uploading file:', error);
      throw error;
    }
  }

  // Update latency multiplier
  async updateLatency(sessionId: string, latencyMultiplier: number): Promise<any> {
    try {
      const params = new URLSearchParams({
        session_id: sessionId,
        latency_multiplier: latencyMultiplier.toString()
      });

      const response = await axiosInstance.post(`/update_latency?${params}`);
      return response.data;
    } catch (error) {
      console.error('Error updating latency:', error);
      throw error;
    }
  }

  // Reset translation state
  async resetTranslation(sessionId: string): Promise<any> {
    try {
      const params = new URLSearchParams({
        session_id: sessionId
      });

      const response = await axiosInstance.post(`/reset_translation?${params}`);
      return response.data;
    } catch (error) {
      console.error('Error resetting translation:', error);
      throw error;
    }
  }

  // Delete session
  async deleteSession(sessionId: string): Promise<any> {
    try {
      const params = new URLSearchParams({
        session_id: sessionId
      });

      const response = await axiosInstance.post(`/delete_session?${params}`);
      return response.data;
    } catch (error) {
      console.error('Error deleting session:', error);
      // Don't throw error for delete failures
      return { success: false, error: error };
    }
  }

  // Create WebSocket connection
  createWebSocket(sessionId: string): WebSocket {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const host = API_BASE_URL.replace(/^https?:\/\//, '');
    const wsUrl = `${protocol}//${host}/wss/${sessionId}`;
    
    return new WebSocket(wsUrl);
  }

  // Store session ID in localStorage
  storeSessionId(sessionId: string): void {
    localStorage.setItem(`translationSessionId_${this.clientId}`, sessionId);
  }

  // Get stored session ID
  getStoredSessionId(): string | null {
    return localStorage.getItem(`translationSessionId_${this.clientId}`);
  }

  // Remove stored session ID
  removeStoredSessionId(): void {
    localStorage.removeItem(`translationSessionId_${this.clientId}`);
  }

  // Store pending delete sessions
  storePendingDeleteSession(sessionId: string): void {
    const pendingSessions = JSON.parse(
      localStorage.getItem(`pendingDeleteSessions_${this.clientId}`) || '{}'
    );
    pendingSessions[sessionId] = Date.now();
    localStorage.setItem(`pendingDeleteSessions_${this.clientId}`, JSON.stringify(pendingSessions));
  }

  // Get pending delete sessions
  getPendingDeleteSessions(): Record<string, number> {
    return JSON.parse(
      localStorage.getItem(`pendingDeleteSessions_${this.clientId}`) || '{}'
    );
  }

  // Remove pending delete session
  removePendingDeleteSession(sessionId: string): void {
    const pendingSessions = this.getPendingDeleteSessions();
    delete pendingSessions[sessionId];
    localStorage.setItem(`pendingDeleteSessions_${this.clientId}`, JSON.stringify(pendingSessions));
  }

  // Get client ID
  getClientId(): string {
    return this.clientId;
  }
}

const apiService = new ApiService();
export default apiService; 