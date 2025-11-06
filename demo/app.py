import re
import argparse

import gradio as gr
import numpy as np

import torch
import torchaudio.functional as F

def prepare_speech(new_chunk):
    sr, y = new_chunk
    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)
        
    y = y.astype(np.float32)
    y /= 32768.0

    resampled_y = F.resample(torch.from_numpy(y), sr, 16000)

    return resampled_y.numpy()

def wav_array_to_base64(wav_array, sample_rate):
    """Convert a numpy audio array to base64 encoded WAV."""
    import base64
    import io
    import soundfile as sf
    
    buffer = io.BytesIO()
    sf.write(buffer, wav_array, sample_rate, format='WAV')
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode('utf-8')

def prepare_inputs(messages, audio_base64):
    if not messages:  # Check for None or empty list
        messages = [
            {
                "role": "system", 
                "content": "You are a professional simultaneous interpreter. You will be given chunks of English audio and you need to translate the audio into Chinese text."
            },
        ]
    messages.append(
        {
            "role": "user",
            "content": [{"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{audio_base64}"}}]
        }
    )
    return messages

def translate(messages, new_chunk, chunk_buffer, chunk_size_seconds, last_chunk_time):
    """
    Translate audio chunks with buffering.
    
    Args:
        messages: Conversation history
        new_chunk: New audio chunk from microphone
        chunk_buffer: List of buffered audio arrays
        chunk_size_seconds: Target chunk size in seconds
        last_chunk_time: Timestamp of last received chunk (to detect pauses)
    
    Returns:
        messages, full_translation, updated_chunk_buffer, current_time
    """
    from openai import OpenAI
    import time
    
    current_time = time.time()
    
    if new_chunk is None:
        current_translation = ''.join([message["content"] for message in messages if message["role"] == "assistant"]) if messages else ""
        return messages, current_translation, chunk_buffer, last_chunk_time
    
    # Initialize messages if None
    if messages is None:
        messages = []
    
    # Initialize chunk_buffer if None
    if chunk_buffer is None:
        chunk_buffer = []
    
    # Check if there was a significant gap (> 2 seconds) - indicates pause/resume
    # Clear partial buffer to avoid concatenating audio from different time periods
    if last_chunk_time is not None and (current_time - last_chunk_time) > 2.0:
        if chunk_buffer:
            print(f"⚠️ Detected pause (gap: {current_time - last_chunk_time:.1f}s). Clearing {len(chunk_buffer)} partial chunks.")
        chunk_buffer = []
    
    # Prepare and buffer the new chunk
    y = prepare_speech(new_chunk)
    chunk_buffer.append(y)
    
    # Calculate how many 0.96s chunks we need to reach target size
    chunks_needed = int(chunk_size_seconds / 0.96)
    
    # If we haven't accumulated enough chunks yet, return without processing
    if len(chunk_buffer) < chunks_needed:
        # Return current state without translation
        current_translation = ''.join([message["content"] for message in messages if message["role"] == "assistant"])
        return messages, current_translation, chunk_buffer, current_time
    
    # We have enough chunks - concatenate and process
    concatenated_audio = np.concatenate(chunk_buffer[:chunks_needed])
    chunk_buffer = chunk_buffer[chunks_needed:]  # Keep any extra chunks for next iteration
    
    # Convert to base64
    audio_base64 = wav_array_to_base64(concatenated_audio, 16000)
    
    # Prepare messages
    messages = prepare_inputs(messages, audio_base64)
    
    # Calculate context window size based on chunk size
    # Larger chunks = longer audio = can keep fewer messages in context
    # Base: 30 messages for 1.92s chunks, scale proportionally
    context_window = max(10, int(30 * (1.92 / chunk_size_seconds)))
    
    # Call OpenAI API
    client = OpenAI(
        base_url="https://jaida-avian-irmgard.ngrok-free.dev/v1",
        api_key="",
    )
    
    model_path = "/data/user_data/siqiouya/ckpts/test_swift/Qwen3-Omni-30B-A3B-Instruct-lora/v1-20251104-033331-hf"
    
    completion = client.chat.completions.create(
        model=model_path,
        messages=[messages[0]] + messages[-context_window:],
        top_p=0.95,
        temperature=0.6,
        extra_body={"top_k": 20}
    )
    print(f"completion: {completion}")
    translation = completion.choices[0].message.content
    messages.append(
        {
            "role": "assistant",
            "content": translation
        }
    )
    
    # Get all translations
    full_translation = ''.join([message["content"] for message in messages if message["role"] == "assistant"])
    
    # Keep only the last 5 lines for display
    translation_lines = full_translation.split('\n') if full_translation else ['']
    # Filter out empty lines for counting, but preserve them in output
    non_empty_lines = [line for line in translation_lines if line.strip()]
    
    if len(non_empty_lines) > 5:
        # Find the last 5 non-empty lines and include any surrounding context
        # Count backwards to find where the 5th-to-last non-empty line is
        count = 0
        for i in range(len(translation_lines) - 1, -1, -1):
            if translation_lines[i].strip():
                count += 1
                if count == 5:
                    display_translation = '\n'.join(translation_lines[i:])
                    break
        else:
            display_translation = full_translation
    else:
        display_translation = full_translation
    
    return messages, display_translation, chunk_buffer, current_time


with gr.Blocks(css="""
    .large-font textarea {
        font-size: 20px !important;
        font-weight: 500;
        overflow-y: auto !important;
    }
    .large-font label {
        font-size: 20px !important;
        font-weight: bold;
    }
""") as demo:
    gr.Markdown("# Simultaneous Speech Translation Demo")
    gr.Markdown("**Instructions:** Select chunk size, then click the microphone to start recording. Refresh page to reset the history.")
    
    # State components
    messages_state = gr.State(value=[])
    chunk_buffer_state = gr.State(value=[])
    last_chunk_time_state = gr.State(value=None)
    
    with gr.Row():
        with gr.Column():
            # Chunk size selector (multiples of 0.96)
            chunk_size_selector = gr.Dropdown(
                choices=[0.96, 1.92, 2.88, 3.84, 4.80, 5.76, 6.72, 7.68, 8.64, 9.60],
                value=1.92,
                label="Chunk Size (seconds)",
                info="Larger chunks = more context but slower response. Must be multiple of 0.96s."
            )
            audio_input = gr.Audio(sources=["microphone"], streaming=True, label="Audio Input")
    
    with gr.Row():
        with gr.Column():
            translation_output = gr.Textbox(
                label="Translation", 
                lines=3,
                max_lines=5,
                interactive=False,
                elem_classes=["large-font"],
                autoscroll=True,
                show_copy_button=True
            )
    
    # Streaming translation
    audio_input.stream(
        translate,
        inputs=[messages_state, audio_input, chunk_buffer_state, chunk_size_selector, last_chunk_time_state],
        outputs=[messages_state, translation_output, chunk_buffer_state, last_chunk_time_state],
        show_progress=False,
        stream_every=0.96  # Base unit - buffering happens inside translate()
    )

demo.launch(share=True)