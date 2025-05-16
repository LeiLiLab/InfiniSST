package main

import (
	"log"
	"net/http"
	"time"
	
	"github.com/gorilla/mux"
)

const (
	CLEANUP_INTERVAL = 30 * time.Second
	SESSION_TIMEOUT = 15 * time.Minute
)

func main() {
	// Create session manager
	sessionManager := NewSessionManager()
	
	// Create router
	router := mux.NewRouter()
	
	// Static file server for frontend
	fs := http.FileServer(http.Dir("../infinisst-web/build"))
	router.PathPrefix("/static/").Handler(http.StripPrefix("/static/", fs))
	router.PathPrefix("/assets/").Handler(http.StripPrefix("/assets/", fs))
	
	// API routes
	apiRouter := router.PathPrefix("/api").Subrouter()
	apiRouter.HandleFunc("/init", sessionManager.InitHandler).Methods("POST")
	apiRouter.HandleFunc("/update_latency", sessionManager.UpdateLatencyHandler).Methods("POST")
	apiRouter.HandleFunc("/reset_translation", sessionManager.ResetHandler).Methods("POST")
	apiRouter.HandleFunc("/delete_session", sessionManager.DeleteSessionHandler).Methods("POST")
	apiRouter.HandleFunc("/ping", sessionManager.PingHandler).Methods("POST")
	
	// WebSocket endpoint
	apiRouter.HandleFunc("/ws/{session_id}", sessionManager.WebSocketHandler)
	
	// Add a basic health check endpoint
	router.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("OK"))
	}).Methods("GET")
	
	// Serve index.html for all other routes (for SPA)
	router.PathPrefix("/").HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.ServeFile(w, r, "../infinisst-web/build/index.html")
	})
	
	// Start the session cleanup goroutine
	go cleanupInactiveSessions(sessionManager)
	
	// Start the server
	log.Println("Starting server on :8081")
	if err := http.ListenAndServe(":8081", router); err != nil {
		log.Fatalf("Failed to start server: %v", err)
	}
}

// cleanupInactiveSessions periodically checks for and cleans up inactive sessions
func cleanupInactiveSessions(sessionManager *SessionManager) {
	ticker := time.NewTicker(CLEANUP_INTERVAL)
	defer ticker.Stop()
	
	for range ticker.C {
		now := time.Now()
		var sessionsToRemove []string
		
		// Get a snapshot of the sessions
		sessionManager.sessionMutex.RLock()
		sessions := make(map[string]*TranslationSession)
		for id, session := range sessionManager.sessions {
			sessions[id] = session
		}
		sessionManager.sessionMutex.RUnlock()
		
		// Check for inactive sessions
		for id, session := range sessions {
			if now.Sub(session.LastActivity) > SESSION_TIMEOUT {
				log.Printf("Session %s inactive for %v, cleaning up", id, SESSION_TIMEOUT)
				sessionsToRemove = append(sessionsToRemove, id)
			}
		}
		
		// Remove inactive sessions
		for _, id := range sessionsToRemove {
			if session, exists := sessionManager.GetSession(id); exists {
				// Clean up resources
				CleanupWorkerProcess(session.Process, session.Stdin)
				sessionManager.RemoveSession(id)
				log.Printf("Removed inactive session %s", id)
			}
		}
		
		// Log active sessions count
		log.Printf("Active sessions: %d", len(sessionManager.sessions))
	}
} 