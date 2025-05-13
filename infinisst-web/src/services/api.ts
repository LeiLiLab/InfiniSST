import axios from 'axios';

const API_BASE_URL = 'http://localhost:8080/api';

export interface Translation {
  time: number;
  text: string;
}

class ApiService {
  private webSocket: WebSocket | null = null;

  // Upload audio file for translation
  async translateAudio(audioFile: File, language: string = 'en'): Promise<Translation[]> {
    try {
      const formData = new FormData();
      formData.append('audioFile', audioFile);
      formData.append('language', language);

      const response = await axios.post(`${API_BASE_URL}/translate`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return response.data.translations;
    } catch (error) {
      console.error('Error translating audio:', error);
      // For demo purposes, return mock data if API fails
      return this.getMockTranslations();
    }
  }

  // Request YouTube video translation
  async translateYoutubeVideo(videoId: string, language: string = 'en'): Promise<{ message: string }> {
    try {
      const response = await axios.post(`${API_BASE_URL}/youtube`, {
        videoId,
        language,
      });

      return response.data;
    } catch (error) {
      console.error('Error processing YouTube video:', error);
      return { message: 'Error processing YouTube video. Using mock translations.' };
    }
  }

  // Start WebSocket connection for real-time translation
  startRealtimeTranslation(
    language: string = 'en',
    onTranslation: (translation: Translation) => void,
    onError: (error: any) => void
  ): void {
    this.webSocket = new WebSocket(`ws://localhost:8080/api/translate/stream?language=${language}`);

    this.webSocket.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onTranslation(data);
      } catch (error) {
        // If not JSON, use the text directly
        onTranslation({
          time: Date.now() / 1000,
          text: event.data as string,
        });
      }
    };

    this.webSocket.onerror = (event) => {
      console.error('WebSocket error:', event);
      onError(event);
    };

    this.webSocket.onclose = () => {
      console.log('WebSocket connection closed');
    };
  }

  // Send audio chunk through WebSocket
  sendAudioChunk(audioChunk: Blob): void {
    if (this.webSocket && this.webSocket.readyState === WebSocket.OPEN) {
      this.webSocket.send(audioChunk);
    }
  }

  // Close WebSocket connection
  stopRealtimeTranslation(): void {
    if (this.webSocket) {
      this.webSocket.close();
      this.webSocket = null;
    }
  }

  // Fallback mock translations
  private getMockTranslations(): Translation[] {
    return [
      { time: 1.5, text: "Hello, welcome to the demo" },
      { time: 3.2, text: "This is an example of a translation" },
      { time: 6.8, text: "The translation appears in sync with the audio" },
      { time: 10.5, text: "You can drag this bar to reposition it" },
      { time: 15.2, text: "Similar to lyrics display in music apps" }
    ];
  }
}

const apiService = new ApiService();
export default apiService; 