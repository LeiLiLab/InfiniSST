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
from train.uni_wav2vec_monkey_patch import replace_forward
from fairseq.data.audio.speech_to_text_dataset import _collate_frames

from eval.agents.tt_waitk_sllama import S2TAgentStates, WaitkSpeechLlama
from train.uni_wav2vec_monkey_patch import uni_forward

@dataclass
class IncrementalS2TAgentStates(S2TAgentStates):
    w2v2_past_key_values: list
    past_key_values: list
    speech_prefix_length: int
    num_frames_read: int

    def reset(self):
        super().reset()
        self.w2v2_past_key_values = []
        self.past_key_values = None
        self.speech_prefix_length = -1
        self.num_frames_read = 0

@entrypoint
class IncrementalWaitkSpeechLlama(WaitkSpeechLlama):
    """
    The agent generate the number of seconds from an input audio.
    """

    def __init__(self, args):
        super().__init__(args)
        uni_forward()
    
    def build_states(self):
        return IncrementalS2TAgentStates([], [], None, -1, 0)

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
        speech_batch = _collate_frames([source[states.num_frames_read:]], is_audio_input=True)
        n_frames = torch.tensor([source.size(0)], dtype=torch.long)
        speech_lens = self.length_after_adp(self.length_after_ssl(n_frames)) - \
            self.length_after_adp(self.length_after_ssl(torch.tensor(states.num_frames_read)))
        n_frames -= states.num_frames_read

        to_adds = [100*self.DEFAULT_SPEECH_PATCH_TOKEN for speech_len in speech_lens]
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
        while len(states.target_ids) + len(prediction_ids) <= max_number_of_tokens:
            
            inputs = self.tokenizer([prompt_inputs])
            input_ids = inputs.input_ids[0] + states.target_ids + prediction_ids
            input_ids_tensor = torch.as_tensor([input_ids]).cuda()

            self.model.model.speech_features_extracted = False
            stopping_criteria = SpaceStoppingCriteria(self.tokenizer)
            with torch.inference_mode():
                # output = self.model(
                #     input_ids=input_ids_tensor,
                #     speech_batch=speech_batch,
                #     src_lengths=n_frames.to(device=self.model.device),
                #     after_lens=speech_lens.to(device=self.model.device),
                # )[0][0, -1]

                output_ids = self.model.generate(
                    attention_mask=input_ids_tensor.ne(self.tokenizer.pad_token_id),
                    input_ids=input_ids_tensor,
                    speech_batch=speech_batch,
                    src_lengths=n_frames.to(device=self.model.device),
                    after_lens=speech_lens.to(device=self.model.device),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=500,
                    repetition_penalty=self.repeat_penalty,
                    stopping_criteria=[stopping_criteria],
                    states=states,
                    use_cache=True
                )
            if stopping_criteria(output_ids, None):
                output_ids = output_ids[:, :-1]

            input_token_len = input_ids_tensor.shape[1]
            prediction_id = output_ids[0, input_token_len:].tolist()

            # prediction_id = output.argmax().item()
            # if prediction_id in input_ids[-2:]:
            #     break

            # n_space_after = self.tokenizer.decode(input_ids + [prediction_id], skip_special_tokens=True).count(' ')
            # if n_space == -1:
            #     n_space = n_space_after
            # elif n_space_after > n_space and not states.source_finished:
            #     break
                
            # prediction_ids.append(prediction_id)

            if prediction_id[-1] == self.tokenizer.eos_token_id and not states.source_finished:
                prediction_id = prediction_id[:-1]
                if len(prediction_id) == 0:
                    break
            prediction_ids.extend(prediction_id)

            # print(self.tokenizer.decode(input_ids + [prediction_id], skip_special_tokens=True))
            # print(self.tokenizer.decode(input_ids + prediction_id, skip_special_tokens=True))
            
            if prediction_ids[-1] == self.tokenizer.eos_token_id:
                break

            if not states.source_finished:
                break
        
        states.num_frames_read = len(states.source)
        states.target_ids.extend(prediction_ids)
        possible_full_word = self.tokenizer.decode(prediction_ids, skip_special_tokens=True)

        if possible_full_word != '' or states.source_finished:
            return WriteAction(
                content=possible_full_word,
                finished=states.source_finished,
            )
        else:
            return ReadAction()