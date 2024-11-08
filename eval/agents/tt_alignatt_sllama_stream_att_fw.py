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

from eval.agents.tt_alignatt_sllama import AlignAtt, S2TAgentStates

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
        if states is not None and not states.source_finished:
            if len(states.target_ids) > self.preserve_t or len(states.source) > self.preserve_s:
                states.target_ids = states.target_ids[-self.preserve_t:]
                states.most_attended_indices = states.most_attended_indices[-self.preserve_t:]

                n_pop = 0
                for i, idx in enumerate(states.most_attended_indices):
                    if len(states.source) - idx >= self.preserve_s:
                        n_pop = i + 1
                states.most_attended_indices = states.most_attended_indices[n_pop:]
                states.target_ids = states.target_ids[n_pop:]

                if len(states.most_attended_indices) > 0:
                    index = states.most_attended_indices.min()
                    states.source = states.source[index:]
                    states.most_attended_indices -= index
                else:
                    states.source = states.source[-self.preserve_s:]

        print(len(states.target_ids), len(states.source) / 16000)

        action = super().policy(states)
        # print(' '.join(states.target) + ' ' + ('' if action.is_read() else action.content))

        return action