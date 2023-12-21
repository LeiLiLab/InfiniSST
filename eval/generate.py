import argparse, sys
sys.path.append('/home/xixu/sllama')
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch, transformers
import os
from eval.utils import disable_torch_init
from model.utils import KeywordsStoppingCriteria
from train.dataset import PromptSpeechToTextDatasetCreator, SpeechToTextDatasetItem

from model.model import SpeechLlamaForCausalLM, SpeechLlamaConfig
from fairseq.data.audio.speech_to_text_dataset import _collate_frames
import conversation as conversation_lib
from conversation import SeparatorStyle

import os
import requests

import torch.nn.functional as F

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_SPEECH_TOKEN = "<speech>"
DEFAULT_SPEECH_PATCH_TOKEN = "<sp_patch>"
DEFAULT_SPEECH_START_TOKEN = "<sp_start>"
DEFAULT_SPEECH_END_TOKEN = "<sp_end>"

prompt_list = ['<speech_here> Describe the speech concisely.',]

def load_speech(image_file):
    import soundfile as sf
    source, sample_rate = sf.read(image_file, dtype="float32",)
    source = torch.from_numpy(source).float()

    return source

def eval_model(args):
    load_type = torch.float16 #torch.float16
    # Model
    disable_torch_init()
    model_name = os.path.expanduser(args.model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="right",
        use_fast=False,
    )
    # process data
    source = load_speech(args.speech_file)
    speech_batch = _collate_frames([source], is_audio_input=True)
    # load model
    model = SpeechLlamaForCausalLM.from_pretrained(args.model_name, torch_dtype=load_type, load_in_8bit=False, use_cache=True).cuda().eval()
    length_after_ssl, length_after_adp = model.model.length_after_ssl, model.model.length_after_adp   
    n_frames = torch.tensor([source.size(0)], dtype=torch.long)
    speech_lens = length_after_adp(length_after_ssl(n_frames))
    to_adds = [int(speech_len)*DEFAULT_SPEECH_PATCH_TOKEN for speech_len in speech_lens]
    to_adds = [DEFAULT_SPEECH_START_TOKEN + to_add + DEFAULT_SPEECH_END_TOKEN for to_add in to_adds]
    
    qs = prompt_list[0]
    before, after = qs.split('<speech_here>')
    mm_prompts = [before + to_add + after for to_add in to_adds]
    
    conv = conversation_lib.default_conversation.copy()       
    conv.append_message(conv.roles[0], mm_prompts[0])
    conv.append_message(conv.roles[1], None)
    prompt_inputs = conv.get_prompt()
    inputs = tokenizer([prompt_inputs])
    print(prompt_inputs)
    input_ids = torch.as_tensor(inputs.input_ids).cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)   
    with torch.inference_mode():
        output_ids = model.generate(
            attention_mask=input_ids.ne(tokenizer.pad_token_id),
            input_ids=input_ids,
            speech_batch=speech_batch.cuda().to(dtype=load_type),
            src_lengths=n_frames.cuda(),
            after_lens=speech_lens.cuda(),
            #do_sample=True,
            num_beams=1,
            max_new_tokens=500,
            stopping_criteria=[stopping_criteria])    

    input_token_len = input_ids.cuda().shape[1]
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()
    print(outputs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--speech-file", type=str, required=True)
    args = parser.parse_args()

    eval_model(args)


