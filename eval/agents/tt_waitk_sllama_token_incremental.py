import argparse, os, sys, time, json

from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction
from simuleval.agents.states import AgentStates
from dataclasses import dataclass

import numpy
import torch
import torch.nn.functional as F
import transformers

import conversation as conversation_lib
from conversation import SeparatorStyle
from eval.utils import disable_torch_init
from model.model import SpeechLlamaForCausalLM
from model.utils import SpaceStoppingCriteria
from train.uni_wav2vec_monkey_patch import replace_uni_train
from fairseq.data.audio.speech_to_text_dataset import _collate_frames

from eval.agents.tt_waitk_sllama_token import S2TAgentStates, WaitkTokenSpeechLlama
from train.uni_wav2vec_monkey_patch import replace_uni_decode

#TODO: this one is still bugged

@dataclass
class IncrementalS2TAgentStates(S2TAgentStates):
    w2v2_past_key_values: list
    w2v2_past_features: torch.Tensor
    past_key_values: list
    speech_prefix_length: int
    speech_past_length: int

    def reset(self):
        super().reset()
        self.w2v2_past_key_values = []
        self.w2v2_past_features = None
        self.past_key_values = None
        self.speech_prefix_length = -1
        self.speech_past_length = 0

@entrypoint
class IncrementalWaitkTokenSpeechLlama(WaitkTokenSpeechLlama):
    """
    The agent generate the number of seconds from an input audio.
    """

    def __init__(self, args):
        replace_uni_decode(args.blocksize)
        super().__init__(args)
    
    def build_states(self):
        return IncrementalS2TAgentStates([], [], [], None, None, -1, 0)
    
    @staticmethod
    def add_args(parser):
        WaitkTokenSpeechLlama.add_args(parser)
        parser.add_argument("--blocksize", default=1, type=int)

    def policy(self, states: Optional[IncrementalS2TAgentStates] = None):
        if states is None:
            states = self.states

        if states.source_sample_rate == 0:
            # empty source, source_sample_rate not set yet
            length_in_seconds = 0
        else:
            length_in_seconds = float(len(states.source)) / states.source_sample_rate

        if not states.source_finished:
            if (
                length_in_seconds * 1000 / self.source_segment_size
            ) < self.waitk_lagging:
                return ReadAction()
            
        if states.source_finished and length_in_seconds < 0.32:
            return WriteAction(content="", finished=True)

        if len(states.w2v2_past_key_values) == 0:
            states.w2v2_past_key_values = [
                {} for _ in range(self.model.model.speech_tower.cfg.encoder_layers)
            ]
        
        source = torch.tensor(states.source).to(
            device=self.model.device, dtype=self.model.dtype
        )
        # source = F.layer_norm(source, source.size())
        speech_batch = _collate_frames([source], is_audio_input=True)
        n_frames = torch.tensor([source.size(0)], dtype=torch.long)
        speech_lens = self.length_after_adp(self.length_after_ssl(n_frames))

        to_adds = [0*self.DEFAULT_SPEECH_PATCH_TOKEN for speech_len in speech_lens]
        to_adds = [self.DEFAULT_SPEECH_START_TOKEN + to_add + self.DEFAULT_SPEECH_END_TOKEN for to_add in to_adds]

        # qs = self.prompt
        # before, after = qs.split('<speech_here>')
        # mm_prompts = [before + to_add + after for to_add in to_adds]

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], to_adds[0])
        conv.append_message(conv.roles[1], None)
        prompt_inputs = conv.get_prompt()

        max_number_of_tokens = length_in_seconds * self.max_len_a + self.max_len_b

        prediction_ids = []
        self.model.model.speech_features_extracted = False
        while len(states.target_ids) + len(prediction_ids) <= max_number_of_tokens:
            
            inputs = self.tokenizer([prompt_inputs])
            input_ids = inputs.input_ids[0] + states.target_ids + prediction_ids
            input_ids_tensor = torch.tensor([input_ids], dtype=torch.long).cuda()

            with torch.inference_mode():
                output_id = self.model(
                    input_ids=input_ids_tensor,
                    attention_mask=input_ids_tensor.ne(self.tokenizer.pad_token_id),
                    speech_batch=speech_batch,
                    src_lengths=n_frames.to(device=self.model.device),
                    after_lens=speech_lens.to(device=self.model.device),
                    use_cache=True,
                    states=states,
                )[0][0, -1:].argmax()

            prediction_id = [output_id.item()]

            if prediction_id[-1] == self.tokenizer.eos_token_id and not states.source_finished:
                prediction_id = prediction_id[:-1]
                if len(prediction_id) == 0:
                    break
            prediction_ids.extend(prediction_id)
            
            if prediction_ids[-1] == self.tokenizer.eos_token_id:
                break

            if not states.source_finished:
                break
        
        states.num_frames_read = len(states.source)

        states.target_ids.extend(prediction_ids)
        if states.source_finished:
            return WriteAction(
                content=self.tokenizer.decode(states.buffer_ids + prediction_ids, skip_special_tokens=True),
                finished=True,
            )
        else:
            prev_buffer_word = self.tokenizer.decode(states.buffer_ids)
            after_buffer_word = self.tokenizer.decode(states.buffer_ids + prediction_ids)
            if len(prev_buffer_word.split(' ')) < len(after_buffer_word.split(' ')):
                states.buffer_ids = prediction_ids
                return WriteAction(
                    content=prev_buffer_word,
                    finished=False,
                )
            else:
                states.buffer_ids.extend(prediction_ids)
                return ReadAction()