package main

import (
	"io"
	"os/exec"
	"sync"
	"time"
)

// TranslationSession represents an active translation session
type TranslationSession struct {
	ID               string    // Unique session identifier
	AgentType        string    // InfiniSST or other model types
	LanguagePair     string    // e.g., "English -> Chinese"
	LatencyMultiplier int      // 1-4, affects translation latency/accuracy tradeoff
	Process          *exec.Cmd // Python subprocess running the model
	Stdin            io.WriteCloser
	Stdout           io.ReadCloser
	LastActivity     time.Time
	IsReady          bool
}

// SessionManager handles session lifecycle
type SessionManager struct {
	sessions     map[string]*TranslationSession
	sessionMutex sync.RWMutex
}

// NewSessionManager creates a new session manager
func NewSessionManager() *SessionManager {
	return &SessionManager{
		sessions: make(map[string]*TranslationSession),
	}
}

// GetSession retrieves a session by ID
func (sm *SessionManager) GetSession(id string) (*TranslationSession, bool) {
	sm.sessionMutex.RLock()
	defer sm.sessionMutex.RUnlock()
	session, exists := sm.sessions[id]
	return session, exists
}

// AddSession adds a new session
func (sm *SessionManager) AddSession(session *TranslationSession) {
	sm.sessionMutex.Lock()
	defer sm.sessionMutex.Unlock()
	sm.sessions[session.ID] = session
}

// RemoveSession removes a session
func (sm *SessionManager) RemoveSession(id string) {
	sm.sessionMutex.Lock()
	defer sm.sessionMutex.Unlock()
	delete(sm.sessions, id)
}

// InitRequest represents the parameters for starting a translation session
type InitRequest struct {
	AgentType        string `json:"agent_type"`
	LanguagePair     string `json:"language_pair"`
	LatencyMultiplier int    `json:"latency_multiplier"`
	ClientID         string `json:"client_id"`
}

// InitResponse is the response to an init request
type InitResponse struct {
	SessionID     string `json:"session_id"`
	Queued        bool   `json:"queued"`
	QueuePosition int    `json:"queue_position"`
	Initializing  bool   `json:"initializing"`
	Error         string `json:"error,omitempty"`
}

// UpdateLatencyRequest represents the parameters for updating latency
type UpdateLatencyRequest struct {
	SessionID        string `json:"session_id"`
	LatencyMultiplier int    `json:"latency_multiplier"`
}

// GenericResponse is a generic response for various endpoints
type GenericResponse struct {
	Success bool   `json:"success"`
	Message string `json:"message,omitempty"`
	Error   string `json:"error,omitempty"`
} 