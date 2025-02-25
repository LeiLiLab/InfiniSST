from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import soundfile as sf
import numpy as np
import json
import asyncio
from typing import Dict, Optional
from eval.agents.streamllama import StreamLlama
from eval.agents.tt_alignatt_sllama_stream_att_fw import AlignAttStreamAttFW
import io
import uvicorn

# 支持的翻译模型列表
TRANSLATION_AGENTS = {
    "InfiniSST": StreamLlama,
    "StreamAtt": AlignAttStreamAttFW,
}

# 支持的语言方向
LANGUAGE_PAIRS = {
    "English -> Chinese": ("English", "Chinese", "en", "zh"),
    "English -> German": ("English", "German", "en", "de"),
    "English -> Spanish": ("English", "Spanish", "en", "es"),
}

model_path = "/compute/babel-5-23/siqiouya/runs/{}-{}/8B-traj-s2-v3.6/last.ckpt/pytorch_model.bin"

app = FastAPI()

# Store active translation sessions
active_sessions: Dict[str, dict] = {}

class TranslationSession:
    def __init__(self, agent_type: str, language_pair: str, args):
        self.agent_type = agent_type
        self.language_pair = language_pair
        source_lang, target_lang, src_code, tgt_code = LANGUAGE_PAIRS[language_pair]
        
        args.source_lang = source_lang
        args.target_lang = target_lang
        args.state_dict_path = model_path.format(src_code, tgt_code)
        
        self.agent = TRANSLATION_AGENTS[agent_type](args)
        self.states = self.agent.build_states()
        self.states.reset()
        self.args = args  # Store args in the session
        
    async def process_segment(self, segment: np.ndarray, is_last: bool = False) -> str:
        self.states.source.extend(segment)
        print(len(self.states.source) / 16000)
        if is_last:
            self.states.source_finished = True
            
        action = self.agent.policy(self.states)
        if not action.is_read():
            output = action.content
            self.states.target.append(output)
            return ' '.join(self.states.target) if self.args.target_lang != 'Chinese' else ''.join(self.states.target)
        return ""

@app.post("/init")
async def initialize_translation(agent_type: str, language_pair: str):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-segment-size", type=int, default=960)
    StreamLlama.add_args(parser)
    args = parser.parse_args()

    session = TranslationSession(agent_type, language_pair, args)
    session_id = f"{agent_type}_{language_pair}_{len(active_sessions)}"
    active_sessions[session_id] = session
    return {"session_id": session_id}

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    await websocket.accept()
    
    if session_id not in active_sessions:
        await websocket.close(code=4000, reason="Invalid session ID")
        return
        
    session = active_sessions[session_id]
    chunk_count = 0
    
    try:
        while True:
            # Receive audio data
            data = await websocket.receive_bytes()
            
            # Convert bytes to numpy array
            audio_data = np.frombuffer(data, dtype=np.float32)
            chunk_count += 1
            print(f"Received chunk {chunk_count}, size: {len(audio_data)}")
            
            # Process the segment
            translation = await session.process_segment(audio_data)
            
            if translation:
                print(f"Got translation for chunk {chunk_count}: {translation}")
                await websocket.send_text(translation)
                
    except Exception as e:
        print(f"Error in WebSocket connection: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if session_id in active_sessions:
            del active_sessions[session_id]

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    contents = await file.read()
    audio_data, sr = sf.read(io.BytesIO(contents))
    return {"sample_rate": sr, "duration": len(audio_data) / sr}

# Mount static files
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 