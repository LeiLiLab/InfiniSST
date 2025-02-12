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

@entrypoint
class AlignAttStreamAttFW(AlignAtt):
    def __init__(self, args):
        super().__init__(args)
        self.preserve_s = int(args.speech_preserve_duration * 16000)
        self.preserve_t = args.text_preserve_num

    @staticmethod
    def add_args(parser):
        AlignAtt.add_args(parser)
        parser.add_argument("--speech-preserve-duration", type=float, default=30)
        parser.add_argument("--text-preserve-num", type=int, default=30)
    
    @torch.inference_mode()
    def policy(self, states: Optional[S2TAgentStates] = None):
        print(len(states.target_ids), len(states.source) / 16000)

        action = super().policy(states)
        print(' '.join(states.target) + ' ' + ('' if action.is_read() else action.content))

        if states is not None and not states.source_finished:
            target = self.tokenizer.decode(states.target_ids, skip_special_tokens=True).strip()
            target_len = len(target.split(' ')) if self.target_lang != 'zh' else len(target)
            if target_len > self.preserve_t or len(states.source) > self.preserve_s:
                target = ' '.join(target.split(' ')[-self.preserve_t:]) if self.target_lang != 'zh' else target[-self.preserve_t:]
                states.target_ids = self.tokenizer.encode(target, add_special_tokens=False)

                states.most_attended_indices = states.most_attended_indices[-len(states.target_ids):]
                # bug for n pop
                # n_pop = 0
                # print("most attended indices:", len(states.most_attended_indices))
                # for i, idx in enumerate(states.most_attended_indices):
                #     if len(states.source) - idx >= self.preserve_s:
                #         n_pop = i + 1
                # states.most_attended_indices = states.most_attended_indices[n_pop:]
                # states.target_ids = states.target_ids[n_pop:]
                # print("n_pop:", n_pop)
                # print("most attended history indices:", len(states.most_attended_indices))
                if len(states.most_attended_indices) > 0:
                    index = states.most_attended_indices.min() # earliest index; discard eos
                    states.source = states.source[index:]
                    # states.most_attended_indices -= index
                # else:
                #     states.source = states.source[-self.preserve_s:]

        return action