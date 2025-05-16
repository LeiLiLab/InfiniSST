package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/gorilla/mux"
	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	ReadBufferSize:  1024,
	WriteBufferSize: 1024,
	CheckOrigin: func(r *http.Request) bool {
		return true // Allow all origins for now (in production, you should restrict this)
	},
}

// InitHandler handles the /init endpoint to create a new translation session
func (sm *SessionManager) InitHandler(w http.ResponseWriter, r *http.Request) {
	var req InitRequest
	
	// Parse request JSON
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request format", http.StatusBadRequest)
		return
	}
	
	// Validate request
	if req.AgentType == "" {
		req.AgentType = "InfiniSST" // Default to InfiniSST
	}
	
	if req.LanguagePair == "" {
		req.LanguagePair = "English -> Chinese" // Default language pair
	}
	
	if req.LatencyMultiplier <= 0 {
		req.LatencyMultiplier = 2 // Default latency multiplier
	}
	
	// Generate a unique session ID
	timestamp := time.Now().UnixNano() / int64(time.Millisecond)
	clientSuffix := ""
	if req.ClientID != "" {
		clientSuffix = "_" + req.ClientID
	} else {
		clientSuffix = "_" + strconv.FormatInt(timestamp, 10)
	}
	sessionID := fmt.Sprintf("%s_%s_%d%s", req.AgentType, strings.Replace(req.LanguagePair, " -> ", "_", 1), sm.sessionCount(), clientSuffix)
	
	// Create the worker process
	cmd, stdin, stdout, err := CreateWorkerProcess(req.AgentType, req.LanguagePair, req.LatencyMultiplier)
	if err != nil {
		log.Printf("Failed to create worker process: %v", err)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(InitResponse{
			Error: fmt.Sprintf("Failed to initialize session: %v", err),
		})
		return
	}
	
	// Create the session
	session := &TranslationSession{
		ID:               sessionID,
		AgentType:        req.AgentType,
		LanguagePair:     req.LanguagePair,
		LatencyMultiplier: req.LatencyMultiplier,
		Process:          cmd,
		Stdin:            stdin,
		Stdout:           stdout,
		LastActivity:     time.Now(),
		IsReady:          false, // Will be set to true when the worker signals readiness
	}
	
	// Add the session to the manager
	sm.AddSession(session)
	
	// Start a goroutine to wait for ready signal from worker
	go func() {
		// In a real implementation, we'd read a ready signal from the worker
		// For the MVP, we'll just wait a fixed time
		time.Sleep(5 * time.Second)
		session.IsReady = true
		log.Printf("Session %s is ready", sessionID)
	}()
	
	// Return the session ID
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(InitResponse{
		SessionID:    sessionID,
		Queued:       false,
		QueuePosition: 0,
		Initializing: true,
	})
}

// WebSocketHandler handles WebSocket connections for streaming audio and receiving translations
func (sm *SessionManager) WebSocketHandler(w http.ResponseWriter, r *http.Request) {
	// Get session ID from URL
	vars := mux.Vars(r)
	sessionID := vars["session_id"]
	
	// Check if session exists
	session, exists := sm.GetSession(sessionID)
	if !exists {
		http.Error(w, "Invalid session ID", http.StatusBadRequest)
		return
	}
	
	// Update activity timestamp
	session.LastActivity = time.Now()
	
	// Upgrade HTTP connection to WebSocket
	conn, err := upgrader.Upgrade(w, r, nil)
	if err != nil {
		log.Printf("Failed to upgrade to WebSocket: %v", err)
		return
	}
	defer conn.Close()
	
	// Wait for worker process to be ready
	if !session.IsReady {
		// Send initializing message
		if err := conn.WriteMessage(websocket.TextMessage, []byte("INITIALIZING: Worker process is starting, please wait...")); err != nil {
			log.Printf("Failed to send initializing message: %v", err)
			return
		}
		
		// Wait for ready state (with timeout)
		startTime := time.Now()
		for !session.IsReady && time.Since(startTime) < 60*time.Second {
			time.Sleep(500 * time.Millisecond)
		}
		
		if !session.IsReady {
			conn.WriteMessage(websocket.TextMessage, []byte("ERROR: Worker process initialization timeout"))
			conn.Close()
			return
		}
		
		// Send ready message
		if err := conn.WriteMessage(websocket.TextMessage, []byte("READY: Worker process is ready")); err != nil {
			log.Printf("Failed to send ready message: %v", err)
			return
		}
	}
	
	// Start a goroutine to read translations from the worker and send them to the client
	outputChan, errorChan := ReadTranslationOutput(session.Stdout)
	
	// Channel to signal when the WebSocket connection is closed
	done := make(chan struct{})
	
	// Start a goroutine for sending translations to client
	go func() {
		defer close(done)
		
		for {
			select {
			case translation, ok := <-outputChan:
				if !ok {
					// outputChan closed, worker process ended
					return
				}
				
				if err := conn.WriteMessage(websocket.TextMessage, []byte(translation)); err != nil {
					log.Printf("Error sending translation to client: %v", err)
					return
				}
				
			case err, ok := <-errorChan:
				if !ok {
					// errorChan closed
					return
				}
				
				log.Printf("Error from worker process: %v", err)
				if err := conn.WriteMessage(websocket.TextMessage, []byte(fmt.Sprintf("ERROR: %v", err))); err != nil {
					log.Printf("Error sending error message to client: %v", err)
					return
				}
			}
		}
	}()
	
	// Process incoming WebSocket messages (audio data)
	for {
		messageType, p, err := conn.ReadMessage()
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket error: %v", err)
			}
			break
		}
		
		// Update activity timestamp
		session.LastActivity = time.Now()
		
		// Process different types of messages
		switch messageType {
		case websocket.BinaryMessage:
			// Audio data
			if err := SendAudioChunk(session.Stdin, p, false); err != nil {
				log.Printf("Error sending audio chunk to worker: %v", err)
				conn.WriteMessage(websocket.TextMessage, []byte(fmt.Sprintf("ERROR: %v", err)))
				break
			}
			
		case websocket.TextMessage:
			// Text commands
			cmd := string(p)
			if cmd == "RESET" {
				if err := ResetTranslation(session.Stdin); err != nil {
					log.Printf("Error resetting translation: %v", err)
					conn.WriteMessage(websocket.TextMessage, []byte(fmt.Sprintf("ERROR: %v", err)))
				}
			} else if strings.HasPrefix(cmd, "UPDATE_LATENCY:") {
				parts := strings.Split(cmd, ":")
				if len(parts) != 2 {
					conn.WriteMessage(websocket.TextMessage, []byte("ERROR: Invalid latency update format"))
					continue
				}
				
				multiplier, err := strconv.Atoi(parts[1])
				if err != nil {
					conn.WriteMessage(websocket.TextMessage, []byte(fmt.Sprintf("ERROR: Invalid latency multiplier: %v", err)))
					continue
				}
				
				if err := UpdateLatency(session.Stdin, multiplier); err != nil {
					log.Printf("Error updating latency: %v", err)
					conn.WriteMessage(websocket.TextMessage, []byte(fmt.Sprintf("ERROR: %v", err)))
				}
				
				// Update session's latency multiplier
				session.LatencyMultiplier = multiplier
			}
		}
	}
	
	// WebSocket connection closed, but don't clean up the session yet
	// Let the cleanup goroutine handle it based on inactivity
	log.Printf("WebSocket connection closed for session %s", sessionID)
}

// UpdateLatencyHandler handles requests to update the latency multiplier
func (sm *SessionManager) UpdateLatencyHandler(w http.ResponseWriter, r *http.Request) {
	var req UpdateLatencyRequest
	
	// Parse request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request format", http.StatusBadRequest)
		return
	}
	
	// Check if session exists
	session, exists := sm.GetSession(req.SessionID)
	if !exists {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(GenericResponse{
			Success: false,
			Error:   "Invalid session ID",
		})
		return
	}
	
	// Make sure session is ready
	if !session.IsReady {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(GenericResponse{
			Success: false,
			Error:   "Worker process not ready, try again later",
		})
		return
	}
	
	// Update latency multiplier
	if err := UpdateLatency(session.Stdin, req.LatencyMultiplier); err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(GenericResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to update latency: %v", err),
		})
		return
	}
	
	// Update session's latency multiplier
	session.LatencyMultiplier = req.LatencyMultiplier
	session.LastActivity = time.Now()
	
	// Return success
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(GenericResponse{
		Success: true,
	})
}

// ResetHandler handles requests to reset the translation state
func (sm *SessionManager) ResetHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		SessionID string `json:"session_id"`
	}
	
	// Parse request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request format", http.StatusBadRequest)
		return
	}
	
	// Check if session exists
	session, exists := sm.GetSession(req.SessionID)
	if !exists {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(GenericResponse{
			Success: false,
			Error:   "Invalid session ID",
		})
		return
	}
	
	// Make sure session is ready
	if !session.IsReady {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(GenericResponse{
			Success: false,
			Error:   "Worker process not ready, try again later",
		})
		return
	}
	
	// Reset translation state
	if err := ResetTranslation(session.Stdin); err != nil {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(GenericResponse{
			Success: false,
			Error:   fmt.Sprintf("Failed to reset translation: %v", err),
		})
		return
	}
	
	// Update session activity timestamp
	session.LastActivity = time.Now()
	
	// Return success
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(GenericResponse{
		Success: true,
		Message: "Translation state reset successfully",
	})
}

// DeleteSessionHandler handles requests to delete a session
func (sm *SessionManager) DeleteSessionHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		SessionID string `json:"session_id"`
	}
	
	// Parse request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request format", http.StatusBadRequest)
		return
	}
	
	// Check if session exists
	session, exists := sm.GetSession(req.SessionID)
	if !exists {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(GenericResponse{
			Success: false,
			Error:   "Invalid session ID",
		})
		return
	}
	
	// Clean up worker process
	CleanupWorkerProcess(session.Process, session.Stdin)
	
	// Remove session from manager
	sm.RemoveSession(req.SessionID)
	
	// Return success
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(GenericResponse{
		Success: true,
	})
}

// PingHandler handles ping requests to keep a session alive
func (sm *SessionManager) PingHandler(w http.ResponseWriter, r *http.Request) {
	var req struct {
		SessionID string `json:"session_id"`
	}
	
	// Parse request
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, "Invalid request format", http.StatusBadRequest)
		return
	}
	
	// Check if session exists
	session, exists := sm.GetSession(req.SessionID)
	if !exists {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(GenericResponse{
			Success: false,
			Error:   "Invalid session ID",
		})
		return
	}
	
	// Update activity timestamp
	session.LastActivity = time.Now()
	
	// Return success
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(GenericResponse{
		Success: true,
	})
}

// sessionCount returns the number of active sessions
func (sm *SessionManager) sessionCount() int {
	sm.sessionMutex.RLock()
	defer sm.sessionMutex.RUnlock()
	return len(sm.sessions)
} 