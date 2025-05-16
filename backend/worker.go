package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

// CreateWorkerProcess spawns a new Python process for the translation model
func CreateWorkerProcess(agentType, languagePair string, latencyMultiplier int) (*exec.Cmd, io.WriteCloser, io.ReadCloser, error) {
	// Prepare command to run the Python script
	// For the MVP, we'll focus on English -> Chinese
	modelDir := "../infinisst-model"
	scriptPath := filepath.Join(modelDir, "serve", "worker.py")
	
	// Build command arguments based on original api.sh
	args := []string{
		scriptPath,
		"--agent-type", agentType,
		"--language-pair", languagePair,
		"--latency-multiplier", strconv.Itoa(latencyMultiplier),
		"--w2v2-path", "/compute/babel-4-1/siqiouya/wav2_vec_vox_960h_pl.pt",
		"--w2v2-type", "w2v2",
		"--ctc-finetuned", "True",
		"--length-shrink-cfg", "[(1024,2,2)] * 2",
		"--block-size", "48",
		"--max-cache-size", "576",
		"--xpos", "0",
		"--rope", "1",
		"--audio-normalize", "1",
		"--max-llm-cache-size", "1000",
		"--always-cache-system-prompt",
		"--max-len-a", "10",
		"--max-len-b", "20",
		"--max-new-tokens", strconv.Itoa(10 * latencyMultiplier),
		"--beam", "4",
		"--no-repeat-ngram-lookback", "100",
		"--no-repeat-ngram-size", "5",
		"--repetition-penalty", "1.2",
		"--suppress-non-language",
		"--model-name", "/compute/babel-4-1/siqiouya/llama-3.1-8b-instruct-hf",
		"--lora-rank", "32",
	}

	// Create the command
	cmd := exec.Command("python", args...)

	// Set up stdin, stdout, and stderr pipes
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to create stdin pipe: %w", err)
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		stdin.Close()
		return nil, nil, nil, fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	// Capture stderr for logging
	stderr, err := cmd.StderrPipe()
	if err != nil {
		stdin.Close()
		stdout.Close()
		return nil, nil, nil, fmt.Errorf("failed to create stderr pipe: %w", err)
	}

	// Start the command
	if err := cmd.Start(); err != nil {
		stdin.Close()
		stdout.Close()
		return nil, nil, nil, fmt.Errorf("failed to start worker process: %w", err)
	}

	// Handle stderr for logging
	go func() {
		scanner := bufio.NewScanner(stderr)
		for scanner.Scan() {
			log.Printf("Worker stderr: %s", scanner.Text())
		}
		if err := scanner.Err(); err != nil {
			log.Printf("Error reading stderr: %v", err)
		}
	}()

	return cmd, stdin, stdout, nil
}

// SendAudioChunk sends an audio chunk to the worker process
func SendAudioChunk(stdin io.Writer, audioData []byte, isLast bool) error {
	// Send audio chunk to the worker process using a simple protocol:
	// Size of chunk (4 bytes) + Is last chunk (1 byte) + Audio data
	
	headerSize := 5 // 4 bytes for size + 1 byte for isLast flag
	header := make([]byte, headerSize)
	
	// Write the size of the audio data (big-endian)
	dataSize := len(audioData)
	header[0] = byte(dataSize >> 24)
	header[1] = byte(dataSize >> 16)
	header[2] = byte(dataSize >> 8)
	header[3] = byte(dataSize)
	
	// Write the isLast flag (0 = false, 1 = true)
	if isLast {
		header[4] = 1
	} else {
		header[4] = 0
	}
	
	// Write header
	if _, err := stdin.Write(header); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}
	
	// Write audio data
	if _, err := stdin.Write(audioData); err != nil {
		return fmt.Errorf("failed to write audio data: %w", err)
	}
	
	return nil
}

// ReadTranslationOutput reads translation output from the worker process
func ReadTranslationOutput(stdout io.Reader) (chan string, chan error) {
	outputChan := make(chan string)
	errorChan := make(chan error, 1)
	
	go func() {
		defer close(outputChan)
		defer close(errorChan)
		
		scanner := bufio.NewScanner(stdout)
		for scanner.Scan() {
			text := scanner.Text()
			
			// Check if it's an error message
			if strings.HasPrefix(text, "ERROR:") {
				errorChan <- fmt.Errorf(strings.TrimPrefix(text, "ERROR:"))
				continue
			}
			
			outputChan <- text
		}
		
		if err := scanner.Err(); err != nil {
			errorChan <- fmt.Errorf("error reading from worker stdout: %w", err)
		}
	}()
	
	return outputChan, errorChan
}

// ResetTranslation sends a reset command to the worker
func ResetTranslation(stdin io.Writer) error {
	// Send a reset command (special format recognized by the worker)
	cmd := "RESET\n"
	_, err := stdin.Write([]byte(cmd))
	return err
}

// UpdateLatency sends a latency update command to the worker
func UpdateLatency(stdin io.Writer, latencyMultiplier int) error {
	// Send a latency update command
	cmd := fmt.Sprintf("UPDATE_LATENCY:%d\n", latencyMultiplier)
	_, err := stdin.Write([]byte(cmd))
	return err
}

// CleanupWorkerProcess terminates the worker process
func CleanupWorkerProcess(cmd *exec.Cmd, stdin io.WriteCloser) {
	// Send terminate command
	stdin.Write([]byte("TERMINATE\n"))
	stdin.Close()
	
	// Wait for the process to terminate with a timeout
	doneCh := make(chan error, 1)
	go func() {
		doneCh <- cmd.Wait()
	}()
	
	select {
	case <-doneCh:
		// Process exited normally
	case <-time.After(5 * time.Second):
		// Process did not exit within timeout, force terminate
		if err := cmd.Process.Kill(); err != nil {
			log.Printf("Failed to kill process: %v", err)
		}
	}
} 