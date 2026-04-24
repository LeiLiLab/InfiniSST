import os
import torch
import soundfile as sf

from transformers import Qwen3OmniMoeThinkerForConditionalGeneration, Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor, GenerationConfig
from qwen_omni_utils import process_mm_info

MODEL_PATH = "/data/user_data/siqiouya/ckpts/pretrained/llm/Qwen3-Omni-30B-A3B-Instruct"
model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    dtype="auto",
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_PATH)
if hasattr(model, 'disable_talker'):
    model.disable_talker()
import soundfile as sf 
wav, sr = sf.read('/data/user_data/siqiouya/datasets/acl_6060/dev/full_wavs/2022.acl-long.110.wav')
wav1 = wav[:16000 * 20]
wav2 = wav[:16000 * 10]
conversations = [
    [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": wav1},
                {"type": "text", "text": "Transcribe the audio into text."},
            ],
        },
    ],
    [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": wav2},
                {"type": "text", "text": "Transcribe the audio into text."},
            ],
        },
    ],
]
text = processor.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
audios, images, videos = process_mm_info(conversations, use_audio_in_video=False)

inputs = processor(text=text, 
                   audio=audios, 
                   images=images, 
                   videos=videos, 
                   return_tensors="pt", 
                   padding=True, 
                   use_audio_in_video=False)

inputs['input_features'] = inputs['input_features'].to(model.dtype)

generation_config = GenerationConfig(
    do_sample=True,
    temperature=0.6,
    top_p=0.95,
    top_k=1,
    max_new_tokens=2048,
)

text_ids, audio = model.generate(
    **inputs,
    return_audio=False,
    thinker_return_dict_in_generate=True,
    use_audio_in_video=False
)

text = processor.batch_decode(
    text_ids.sequences[:, inputs["input_ids"].shape[1] :],
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False
)

print(text)