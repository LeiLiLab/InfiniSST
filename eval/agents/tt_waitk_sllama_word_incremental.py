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

from eval.agents.tt_waitk_sllama_word import S2TAgentStates, WaitkSpeechLlama
from train.uni_wav2vec_monkey_patch import replace_uni_decode

@dataclass
class IncrementalS2TAgentStates(S2TAgentStates):
    w2v2_past_features: torch.Tensor
    w2v2_start_pos: int
    past_key_values: list
    speech_prefix_length: int
    speech_past_length: int
    attention_mask: torch.Tensor
    future_text_mask: list

    def reset(self):
        super().reset()
        self.w2v2_past_features = None
        self.w2v2_start_pos = 0
        self.past_key_values = None
        self.speech_prefix_length = -1
        self.speech_past_length = 0
        self.future_text_indices = []
        self.attention_mask = None
        self.future_text_mask = None


@entrypoint
class IncrementalWaitkSpeechLlama(WaitkSpeechLlama):
    """
    The agent generate the number of seconds from an input audio.
    """

    def __init__(self, args):
        replace_uni_decode(args.blocksize)
        super().__init__(args)
    
    def build_states(self):
        return IncrementalS2TAgentStates([], None, None, None, 0, None, -1, 0, None, None)
    
    @staticmethod
    def add_args(parser):
        WaitkSpeechLlama.add_args(parser)
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
            
        if states.ref_target_ids is None and getattr(self, "tgt_id_segs", None) is not None:
            states.ref_target_ids = self.tgt_id_segs[self.test_instance_id]
            
        if states.source_finished and length_in_seconds < 0.32:
            self.test_instance_id += 1
            states.ref_target_ids = None
            return WriteAction(content="", finished=True)
       
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

        self.model.model.speech_features_extracted = False
            
        inputs = self.tokenizer([prompt_inputs])
        

        stopping_criteria = SpaceStoppingCriteria(self.tokenizer, 1)
        with torch.inference_mode():
            # output = self.model(
            #     input_ids=input_ids_tensor,
            #     speech_batch=speech_batch,
            #     src_lengths=n_frames.to(device=self.model.device),
            #     after_lens=speech_lens.to(device=self.model.device),
            # )[0][0, -1]

            # output_ids = self.model.generate(
            #     attention_mask=input_ids_tensor.ne(self.tokenizer.pad_token_id).repeat(self.batch_size, 1),
            #     input_ids=input_ids_tensor.repeat(self.batch_size, 1),
            #     speech_batch=speech_batch.repeat(self.batch_size, 1),
            #     src_lengths=n_frames.to(device=self.model.device).repeat(self.batch_size),
            #     after_lens=speech_lens.to(device=self.model.device).repeat(self.batch_size),
            #     do_sample=False,
            #     num_beams=1,
            #     max_new_tokens=max_number_of_tokens - len(states.target_ids) - len(prediction_ids),
            #     repetition_penalty=self.repeat_penalty,
            #     stopping_criteria=[stopping_criteria] if not states.source_finished else None,
            #     states=states,
            #     use_cache=True
            # )

            prediction_ids = []
            pop_flag = True
            while True:
                input_ids = inputs.input_ids[0] + states.target_ids + prediction_ids
                input_ids_tensor = torch.as_tensor([input_ids]).cuda()

                output = self.model(
                    input_ids=input_ids_tensor.repeat(self.batch_size, 1),
                    attention_mask=input_ids_tensor.ne(self.tokenizer.pad_token_id).repeat(self.batch_size, 1),
                    use_cache=True,
                    speech_batch=speech_batch.repeat(self.batch_size, 1),
                    src_lengths=n_frames.to(device=self.model.device).repeat(self.batch_size),
                    after_lens=speech_lens.to(device=self.model.device).repeat(self.batch_size),
                    past_key_values=states.past_key_values,
                    states=states
                )
                logits = output.logits[0, -1]
                token_id = logits.argmax().item()

                if not states.source_finished:
                    if self.tokenizer.convert_ids_to_tokens(token_id).startswith('▁'):
                        if len(prediction_ids) > 0:
                            pop_flag = True
                            break
                    elif token_id == self.tokenizer.eos_token_id:
                        pop_flag = True
                        break
                else:
                    if token_id == self.tokenizer.eos_token_id:
                        break
                
                prediction_ids.append(token_id)

                if len(states.target_ids + prediction_ids) >= max_number_of_tokens:
                    break

        # if getattr(self, "prof", None) is not None:
        #     self.prof.step()

        if pop_flag:
            states.future_text_mask.pop()
            states.position_ids = states.position_ids[:, :-1]

            states.past_key_values = list(states.past_key_values)
            for i in range(len(states.past_key_values)):
                states.past_key_values[i] = (
                    states.past_key_values[i][0][:, :, :-1],
                    states.past_key_values[i][1][:, :, :-1]
                )

        states.num_frames_read = len(states.source)
        states.target_ids.extend(prediction_ids)
        possible_full_word = self.tokenizer.decode(prediction_ids, skip_special_tokens=True)

        if states.source_finished:
            self.test_instance_id += 1
            states.ref_target_ids = None

            # if getattr(self, "prof", None):
            #     self.prof.stop()

            # self.prof = torch.profiler.profile(
            #     schedule=torch.profiler.schedule(wait=0, warmup=1, active=100, repeat=1),
            #     on_trace_ready=torch.profiler.tensorboard_trace_handler("profile/w2v2_llama2/uni_llama2_v2/#{}".format(self.test_instance_id)),
            # )
            # self.prof.start()

        if possible_full_word != '' or states.source_finished:
            return WriteAction(
                content=possible_full_word,
                finished=states.source_finished,
            )
        else:
            return ReadAction()
