import argparse, os, sys, time, json
from collections import Counter

from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction
from simuleval.agents.states import AgentStates
from dataclasses import dataclass

import copy
import numpy
import torch
import torch.nn.functional as F
import transformers

import conversation as conversation_lib
from conversation import SeparatorStyle
from eval.utils import disable_torch_init
from model.model import SpeechLlamaForCausalLM
from model.utils import SpaceStoppingCriteria, KeywordsStoppingCriteria
# from train.uni_wav2vec_monkey_patch import replace_uni_train
from fairseq.data.audio.speech_to_text_dataset import _collate_frames

from eval.agents.tt_alignatt_sllama3 import AlignAttSpeechLlama3 as AlignAtt, AlignAttStates as S2TAgentStates

import logging
logger = logging.getLogger(__name__)

@entrypoint
class AlignAttStreamAttFW(AlignAtt):
    def __init__(self, args):
        super().__init__(args)
        self.preserve_t = args.text_preserve_num
        self.min_speech_duration = args.min_speech_duration
        self.max_speech_duration = args.max_speech_duration
    @staticmethod
    def add_args(parser):
        AlignAtt.add_args(parser)
        parser.add_argument("--text-preserve-num", type=int, default=40)
        parser.add_argument("--min-speech-duration", type=float, default=10)
        parser.add_argument("--max-speech-duration", type=float, default=28.8)

    @torch.inference_mode()
    def policy(self, states: Optional[S2TAgentStates] = None):

        action = super().policy(states)
        print(' '.join(states.target) + ' ' + ('' if action.is_read() else action.content))

        if states is not None and not states.source_finished:

            if self.preserve_t != -1:
                n_words_to_preserve = self.preserve_t
                preserved_target_ids = []
                for idx in states.target_ids[::-1]:
                    preserved_target_ids.append(idx)
                    if (self.target_lang != 'Chinese' and self.tokenizer.decode(idx).startswith(' ')) or self.target_lang == 'Chinese':
                        n_words_to_preserve -= 1
                        if n_words_to_preserve == 0:
                            break
                preserved_target_ids = preserved_target_ids[::-1]
                while '�' in self.tokenizer.decode(preserved_target_ids):
                    preserved_target_ids.pop(0)
                states.target_ids = preserved_target_ids

                target = self.tokenizer.decode(states.target_ids, skip_special_tokens=True).strip()
                n_word = len(target.split(' ')) if self.target_lang != 'Chinese' else len(target)

                if len(states.target_ids) > 0:
                    src_idx = states.most_attended_indices[-len(states.target_ids):].min()
                    src_idx = min(src_idx, max(0, len(states.source) - int(self.min_speech_duration * 16000)))
                    states.source = states.source[src_idx:]

            states.source = states.source[-int(self.max_speech_duration * 16000):]
            
            print('-' * 100)
            print(f"speech_len: {len(states.source) / 16000}, text_len: {len(states.target_ids)}, preserved text: {self.tokenizer.decode(states.target_ids)}")
            print('-' * 100)

            # if target_len > self.preserve_t or len(states.source) > self.preserve_s:
            #     target = ' '.join(target.split(' ')[-self.preserve_t:]) if self.target_lang != 'Chinese' else target[-self.preserve_t:]
            #     target = target.strip()
            #     states.target_ids = self.tokenizer.encode(target, add_special_tokens=False)

            #     # orig_most_attended_indices = copy.deepcopy(states.most_attended_indices)
            #     states.most_attended_indices = states.most_attended_indices[-len(states.target_ids):]
            #     # bug for n pop
            #     n_pop = 0
            #     # print("most attended indices:", len(states.most_attended_indices))

            #     for i, idx in enumerate(states.most_attended_indices):
            #         if len(states.source) - idx >= self.preserve_s:
            #             n_pop = i + 1
            #     if self.target_lang == 'Chinese':
            #         while n_pop < len(states.target_ids) and '�' in self.tokenizer.decode(states.target_ids[n_pop:]):
            #             n_pop += 1
            #     else:
            #         while n_pop < len(states.target_ids) and not self.tokenizer.decode(states.target_ids[n_pop:]).startswith(' '):
            #             n_pop += 1
            #     states.most_attended_indices = states.most_attended_indices[n_pop:]
            #     states.target_ids = states.target_ids[n_pop:]

            #     target = self.tokenizer.decode(states.target_ids, skip_special_tokens=True).strip()
            #     states.target_ids = self.tokenizer.encode(target, add_special_tokens=False)
                
            #     if len(states.most_attended_indices) > 0:
            #         index = states.most_attended_indices.min() # earliest index; discard eos
            #         states.source = states.source[index:]

        return action