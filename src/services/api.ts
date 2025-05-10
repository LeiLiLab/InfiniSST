import axios from 'axios';
import { TranslationSession } from '../types';

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || '';

class ApiService {
  private async request<T>(method: string, endpoint: string, data?: any): Promise<T> {
    try {
      const response = await axios({
        method,
        url: `${API_BASE_URL}${endpoint}`,
        data,
      });
      return response.data;
    } catch (error) {
      if (axios.isAxiosError(error)) {
        throw new Error(error.response?.data?.error || error.message);
      }
      throw error;
    }
  }

  public async initializeSession(
    agentType: string,
    languagePair: string,
    latencyMultiplier: number,
    clientId: string
  ): Promise<TranslationSession> {
    return this.request<TranslationSession>('POST', '/init', {
      agent_type: agentType,
      language_pair: languagePair,
      latency_multiplier: latencyMultiplier,
      client_id: clientId,
    });
  }

  public async updateLatency(sessionId: string, latencyMultiplier: number): Promise<void> {
    return this.request('POST', '/update_latency', {
      session_id: sessionId,
      latency_multiplier: latencyMultiplier,
    });
  }

  public async resetTranslation(sessionId: string): Promise<void> {
    return this.request('POST', '/reset_translation', {
      session_id: sessionId,
    });
  }

  public async deleteSession(sessionId: string): Promise<void> {
    return this.request('POST', '/delete_session', {
      session_id: sessionId,
    });
  }

  public async pingSession(sessionId: string): Promise<void> {
    return this.request('POST', '/ping', {
      session_id: sessionId,
    });
  }

  public async getQueueStatus(sessionId: string): Promise<{
    status: 'active' | 'queued' | 'initializing' | 'not_found';
    queue_position?: number;
  }> {
    return this.request('GET', `/queue_status/${sessionId}`);
  }
}

export const apiService = new ApiService(); 