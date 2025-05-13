package api

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/gorilla/websocket"

	"github.com/luojiaxuan/infinisst/internal/service"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins for demo
	},
}

type Handler struct {
	translationService *service.TranslationService
}

func NewHandler(translationService *service.TranslationService) *Handler {
	return &Handler{
		translationService: translationService,
	}
}

type TranslateRequest struct {
	AudioData []byte `json:"audioData"`
	Language  string `json:"language"`
}

type TranslateResponse struct {
	Translations []service.Translation `json:"translations"`
	Error        string                `json:"error,omitempty"`
}

type YoutubeRequest struct {
	VideoID  string `json:"videoId"`
	Language string `json:"language"`
}

// TranslateAudio handles audio file uploads for translation
func (h *Handler) TranslateAudio(c *gin.Context) {
	// Get multipart form file
	file, _, err := c.Request.FormFile("audioFile")
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "No audio file provided"})
		return
	}
	defer file.Close()

	// Read file content
	audioData, err := io.ReadAll(file)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to read audio file"})
		return
	}

	// Get target language (default to English)
	language := c.DefaultPostForm("language", "en")

	// Process the translation
	translations, err := h.translationService.TranslateAudio(audioData, language)
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	c.JSON(http.StatusOK, TranslateResponse{
		Translations: translations,
	})
}

// StreamTranslation handles WebSocket connections for real-time translation
func (h *Handler) StreamTranslation(c *gin.Context) {
	// Upgrade to WebSocket
	conn, err := upgrader.Upgrade(c.Writer, c.Request, nil)
	if err != nil {
		log.Printf("Failed to upgrade to WebSocket: %v", err)
		return
	}
	defer conn.Close()

	// Get language from query parameter (default to English)
	language := c.DefaultQuery("language", "en")

	// Start websocket communication
	for {
		// Read message from client
		messageType, p, err := conn.ReadMessage()
		if err != nil {
			log.Printf("Error reading WebSocket message: %v", err)
			break
		}

		// Process audio chunk
		translation, err := h.translationService.TranslateAudioChunk(p, language)
		if err != nil {
			// Send error back to client
			errorMsg := map[string]string{"error": err.Error()}
			if err := conn.WriteJSON(errorMsg); err != nil {
				log.Printf("Error sending error message: %v", err)
				break
			}
			continue
		}

		// Send translation back to client
		if err := conn.WriteMessage(messageType, []byte(translation.Text)); err != nil {
			log.Printf("Error writing WebSocket message: %v", err)
			break
		}
	}
}

// TranslateYoutubeVideo handles YouTube video translation requests
func (h *Handler) TranslateYoutubeVideo(c *gin.Context) {
	var req YoutubeRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Invalid request"})
		return
	}

	// Start a goroutine to process the translation (this would be a long-running task)
	go func() {
		// In a real implementation, we would download the YouTube audio and process it
		log.Printf("Processing YouTube video ID: %s", req.VideoID)
		// Simulate processing time
		time.Sleep(2 * time.Second)
	}()

	// Immediately return a response to indicate processing has started
	c.JSON(http.StatusAccepted, gin.H{
		"message": "Translation processing started",
		"videoId": req.VideoID,
	})
} 