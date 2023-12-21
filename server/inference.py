"""Inference for FastChat models."""
import abc, os
from instruct_final.eval.utils import disable_torch_init
import gc
import math
from typing import Iterable, Optional
import sys
import warnings

import psutil
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    LlamaTokenizer,
    LlamaForCausalLM,
    AutoModel,
    AutoModelForSeq2SeqLM,
    T5Tokenizer,
    AutoConfig,
)
from transformers.generation.logits_process import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

from instruct_speech_new.conversation import SeparatorStyle
from instruct_speech_new import conversation as conversation_lib
from fastchat.model.model_adapter import load_model, get_conversation_template
from fastchat.model.chatglm_model import chatglm_generate_stream

from instruct_speech_new.model.model import SpeechLlamaForCausalLM, SpeechLlamaConfig
from fairseq.data.audio.speech_to_text_dataset import _collate_frames
prefix = '<speech_here>'

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_SPEECH_TOKEN = "<speech>"
DEFAULT_SPEECH_PATCH_TOKEN = "<sp_patch>"
DEFAULT_SPEECH_START_TOKEN = "<sp_start>"
DEFAULT_SPEECH_END_TOKEN = "<sp_end>"

def load_speech_audio_model(model_name):
    disable_torch_init()
    model_name = os.path.expanduser(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name,
        padding_side="right",
        use_fast=False,
    )    
    model = SpeechLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, load_in_8bit=False, use_cache=True).cuda().eval()
    length_after_ssl, length_after_adp = model.model.length_after_ssl, model.model.length_after_adp
    return model, tokenizer, length_after_ssl, length_after_adp 

def load_speech(speech_file):
    import soundfile as sf
    source, sample_rate = sf.read(speech_file, dtype="float32",)
    source = torch.from_numpy(source).float()

    return source

def prepare_logits_processor(
    temperature: float, repetition_penalty: float, top_p: float, top_k: int
) -> LogitsProcessorList:
    processor_list = LogitsProcessorList()
    # TemperatureLogitsWarper doesn't accept 0.0, 1.0 makes it a no-op so we skip two cases.
    if temperature >= 1e-5 and temperature != 1.0:
        processor_list.append(TemperatureLogitsWarper(temperature))
    if repetition_penalty > 1.0:
        processor_list.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    if 1e-8 <= top_p < 1.0:
        processor_list.append(TopPLogitsWarper(top_p))
    if top_k > 0:
        processor_list.append(TopKLogitsWarper(top_k))
    return processor_list


def partial_stop(output, stop_str):
    for i in range(0, min(len(output), len(stop_str))):
        if stop_str.startswith(output[-i:]):
            return True
    return False


@torch.inference_mode()
def generate_stream(
    model, tokenizer, params, device, context_len=2048, stream_interval=2
):
    prompt = params["prompt"]
    speech_batch = params["speech_batch"]
    src_lengths = params["src_lengths"]
    after_lens = params["after_lens"]
    
    len_prompt = len(prompt)
    temperature = float(params.get("temperature", 0.0))
    repetition_penalty = float(params.get("repetition_penalty", 0.0))
    top_p = float(params.get("top_p", 0.0))
    top_k = int(params.get("top_k", -1))  # -1 means disable
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    echo = bool(params.get("echo", True))
    stop_token_ids = params.get("stop_token_ids", None) or []
    stop_token_ids.append(tokenizer.eos_token_id)

    logits_processor = prepare_logits_processor(
        temperature, repetition_penalty, top_p, top_k
    )

    input_ids = tokenizer(prompt).input_ids
    input_echo_len = len(input_ids)
    output_ids = list(input_ids)

    if model.config.is_encoder_decoder:
        max_src_len = context_len
    else:
        max_src_len = context_len - max_new_tokens - 8

    input_ids = input_ids[-max_src_len:]
    
    past_key_values = out = None
    for i in range(max_new_tokens):
        if i == 0:
            out = model(torch.as_tensor([input_ids], device=device), 
                        speech_batch=speech_batch,
                        src_lengths=src_lengths,
                        after_lens=after_lens,
                        use_cache=True
            )
            logits = out.logits
            past_key_values = out.past_key_values
        else:
            out = model(
                input_ids=torch.as_tensor([[token]], device=device),
                use_cache=True,
                past_key_values=past_key_values,
                speech_batch=speech_batch,
                src_lengths=src_lengths,
                after_lens=after_lens,               
            )
            logits = out.logits
            past_key_values = out.past_key_values

        if logits_processor:
            if repetition_penalty > 1.0:
                tmp_output_ids = torch.as_tensor([output_ids], device=logits.device)
            else:
                tmp_output_ids = None
            last_token_logits = logits_processor(tmp_output_ids, logits[:, -1, :])[0]
        else:
            last_token_logits = logits[0, -1, :]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-5 or top_p < 1e-8:  # greedy
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            if echo:
                tmp_output_ids = output_ids
                rfind_start = len_prompt
            else:
                tmp_output_ids = output_ids[input_echo_len:]
                rfind_start = 0

            output = tokenizer.decode(
                tmp_output_ids,
                skip_special_tokens=True,
                spaces_between_special_tokens=False,
            )

            partially_stopped = False
            if stop_str:
                if isinstance(stop_str, str):
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                    else:
                        partially_stopped = partial_stop(output, stop_str)
                elif isinstance(stop_str, Iterable):
                    for each_stop in stop_str:
                        pos = output.rfind(each_stop, rfind_start)
                        if pos != -1:
                            output = output[:pos]
                            stopped = True
                            break
                        else:
                            partially_stopped = partial_stop(output, each_stop)
                            if partially_stopped:
                                break
                else:
                    raise ValueError("Invalid stop field type.")
            
            # prevent yielding partial stop sequence
            if not partially_stopped:
                yield {
                    "text": output,
                    "usage": {
                        "prompt_tokens": input_echo_len,
                        "completion_tokens": i,
                        "total_tokens": input_echo_len + i,
                    },
                    "finish_reason": None,
                }

        if stopped:
            break

    # finish stream event, which contains finish reason
    if i == max_new_tokens - 1:
        finish_reason = "length"
    elif stopped:
        finish_reason = "stop"
    else:
        finish_reason = None

    yield {
        "text": output,
        "usage": {
            "prompt_tokens": input_echo_len,
            "completion_tokens": i,
            "total_tokens": input_echo_len + i,
        },
        "finish_reason": finish_reason,
    }

    # clean
    del past_key_values, out
    gc.collect()
    torch.cuda.empty_cache()


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str):
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream):
        """Stream output."""


def chat_loop(
    model_path: str,
    device: str,
    num_gpus: int,
    max_gpu_memory: str,
    load_8bit: bool,
    cpu_offloading: bool,
    conv_template: Optional[str],
    temperature: float,
    repetition_penalty: float,
    max_new_tokens: int,
    speech_file: str,
    chatio: ChatIO,
    debug: bool,
):
    # Model
    model, tokenizer, length_after_ssl, length_after_adp = load_speech_audio_model(model_path)
    source = load_speech(speech_file)
    speech_batch = _collate_frames([source], is_audio_input=True)
    n_frames = torch.tensor([source.size(0)], dtype=torch.long)
    speech_lens = length_after_adp(length_after_ssl(n_frames))
    to_adds = [int(speech_len)*DEFAULT_SPEECH_PATCH_TOKEN for speech_len in speech_lens]
    to_adds = [DEFAULT_SPEECH_START_TOKEN + to_add + DEFAULT_SPEECH_END_TOKEN for to_add in to_adds]
    # Chat
    conv = conversation_lib.default_conversation.copy()
    #conv.append_message(conv.roles[0], to_adds[0])
    i = 0    
    while True:
        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break
        if i == 0 :
            inp = to_adds[0] + inp
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        generate_stream_func = generate_stream
        prompt = conv.get_prompt()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        #print(prompt)
        
        gen_params = {
            "model": model_path,
            "prompt": prompt,
            "speech_batch": speech_batch.cuda().to(dtype=torch.float16),
            "src_lengths": n_frames.cuda(),
            "after_lens": speech_lens.cuda(),  
            "temperature": temperature,
            "repetition_penalty": repetition_penalty,
            "max_new_tokens": max_new_tokens,
            "stop": stop_str,
            "stop_token_ids": conv.stop_token_ids,
            "echo": False,
        }

        chatio.prompt_for_output(conv.roles[1])
        device = "cuda"
        output_stream = generate_stream_func(model, tokenizer, gen_params, device)
        outputs = chatio.stream_output(output_stream)
        conv.messages[-1][-1] = outputs.strip()
        i = i + 1
        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")
