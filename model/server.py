#!/usr/bin/env python3
import base64
import json
import os
import random
import time
from typing import List, Dict, Any, Optional

import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# This would be replaced with a real model in production
class MockTranslationModel:
    """Mock implementation of a speech translation model."""
    
    def __init__(self):
        self.sample_phrases = [
            "Hello, welcome to the demo",
            "This is an example of a translation",
            "The translation appears in sync with the audio",
            "You can drag this bar to reposition it",
            "Similar to lyrics display in music apps",
            "InfiniSST provides real-time translation",
            "This is a simulated model response",
            "In a real implementation, this would use a neural network",
            "Multiple languages could be supported",
            "Thank you for trying out this demo"
        ]
    
    def process_audio(self, audio_data: bytes, language: str = "en") -> List[Dict[str, Any]]:
        """Process audio data and return translations with timestamps."""
        # Simulate processing delay
        time.sleep(0.5)
        
        # In a real model, we would:
        # 1. Convert audio to the required format
        # 2. Split into segments
        # 3. Transcribe each segment
        # 4. Translate the transcription
        # 5. Return the translations with timestamps
        
        # For demo, we'll create random translations
        num_segments = random.randint(5, 10)
        translations = []
        
        for i in range(num_segments):
            # Create a random timestamp (0-30 seconds range)
            timestamp = i * random.uniform(2.0, 4.0)
            
            # Select a random phrase
            text = random.choice(self.sample_phrases)
            
            translations.append({
                "time": timestamp,
                "text": text
            })
        
        # Sort by timestamp
        translations.sort(key=lambda x: x["time"])
        return translations
    
    def process_chunk(self, audio_chunk: bytes, language: str = "en") -> Dict[str, Any]:
        """Process a single chunk of audio for real-time translation."""
        # Simulate processing delay
        time.sleep(0.1)
        
        # Return a random phrase
        return {
            "time": time.time(),
            "text": random.choice(self.sample_phrases)
        }

# Initialize model
translation_model = MockTranslationModel()

@app.route('/translate', methods=['POST'])
def translate():
    """API endpoint for translating audio files."""
    try:
        # Parse request
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Get audio data (base64 encoded)
        audio_data_base64 = data.get('audio_data_base64')
        if not audio_data_base64:
            return jsonify({"error": "No audio data provided"}), 400
        
        # Get target language (default to English)
        language = data.get('language', 'en')
        
        # Decode base64 to bytes
        try:
            audio_data = base64.b64decode(audio_data_base64)
        except Exception as e:
            return jsonify({"error": f"Invalid base64 encoding: {str(e)}"}), 400
        
        # Process the translation
        translations = translation_model.process_audio(audio_data, language)
        
        # Return the result
        return jsonify({
            "translations": translations
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/translate_chunk', methods=['POST'])
def translate_chunk():
    """API endpoint for real-time translation of audio chunks."""
    try:
        # Parse request
        data = request.json
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Get audio data (base64 encoded)
        audio_chunk_base64 = data.get('audio_chunk_base64')
        if not audio_chunk_base64:
            return jsonify({"error": "No audio chunk provided"}), 400
        
        # Get target language (default to English)
        language = data.get('language', 'en')
        
        # Decode base64 to bytes
        try:
            audio_chunk = base64.b64decode(audio_chunk_base64)
        except Exception as e:
            return jsonify({"error": f"Invalid base64 encoding: {str(e)}"}), 400
        
        # Process the translation
        translation = translation_model.process_chunk(audio_chunk, language)
        
        # Return the result
        return jsonify(translation)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """API endpoint for health checks."""
    return jsonify({"status": "ok"}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) 