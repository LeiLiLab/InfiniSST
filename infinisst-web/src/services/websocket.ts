import { TranslationState } from '../types';

class WebSocketService {
  private ws: WebSocket | null = null;
  private sessionId: string | null = null;
  private onMessageCallback: ((text: string) => void) | null = null;
  private onErrorCallback: ((error: string) => void) | null = null;
  private onStatusCallback: ((status: string) => void) | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectTimeout = 1000;

  constructor() {
    this.handleMessage = this.handleMessage.bind(this);
    this.handleError = this.handleError.bind(this);
    this.handleClose = this.handleClose.bind(this);
  }

  public connect(sessionId: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.sessionId = sessionId;
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${protocol}//${window.location.host}/wss/${sessionId}`;

      try {
        this.ws = new WebSocket(wsUrl);
        this.ws.onmessage = this.handleMessage;
        this.ws.onerror = this.handleError;
        this.ws.onclose = this.handleClose;

        this.ws.onopen = () => {
          this.reconnectAttempts = 0;
          resolve();
        };
      } catch (error) {
        reject(error);
      }
    });
  }

  public disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  public sendAudioData(data: ArrayBuffer): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    }
  }

  public setOnMessageCallback(callback: (text: string) => void): void {
    this.onMessageCallback = callback;
  }

  public setOnErrorCallback(callback: (error: string) => void): void {
    this.onErrorCallback = callback;
  }

  public setOnStatusCallback(callback: (status: string) => void): void {
    this.onStatusCallback = callback;
  }

  private handleMessage(event: MessageEvent): void {
    const message = event.data;

    if (typeof message === 'string') {
      if (message.startsWith('INITIALIZING:')) {
        this.onStatusCallback?.(message.substring('INITIALIZING:'.length).trim());
      } else if (message.startsWith('READY:')) {
        this.onStatusCallback?.('Model is ready. Processing audio...');
      } else if (message.startsWith('ERROR:')) {
        this.onErrorCallback?.(message);
      } else {
        this.onMessageCallback?.(message);
      }
    }
  }

  private handleError(error: Event): void {
    this.onErrorCallback?.(`WebSocket error: ${error}`);
  }

  private handleClose(event: CloseEvent): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      setTimeout(() => {
        if (this.sessionId) {
          this.connect(this.sessionId).catch((error) => {
            this.onErrorCallback?.(`Failed to reconnect: ${error}`);
          });
        }
      }, this.reconnectTimeout * this.reconnectAttempts);
    } else {
      this.onErrorCallback?.('WebSocket connection closed and max reconnection attempts reached');
    }
  }
}

export const websocketService = new WebSocketService(); 