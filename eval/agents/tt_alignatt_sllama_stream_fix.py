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
class AlignAttStreamFix(AlignAtt):
    def __init__(self, args):
        super().__init__(args)
        self.preserve_s = int(args.speech_preserve_duration * args.speech_sample_rate)
        self.preserve_t = args.text_preserve_num

    @staticmethod
    def add_args(parser):
        AlignAtt.add_args(parser)
        parser.add_argument("--speech-preserve-duration", type=float, default=10.0)
        parser.add_argument("--speech-sample-rate", type=int, default=16000)
        parser.add_argument("--text-preserve-num", type=int, default=30)
    
    def policy(self, states: Optional[S2TAgentStates] = None):
        if states is not None:
            states.source = states.source[-self.preserve_s:]
            states.target_ids = states.target_ids[-self.preserve_t:]
        return super().policy(states)