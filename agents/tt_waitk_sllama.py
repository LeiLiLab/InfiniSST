import argparse, os, sys, time, json
sys.path.append('/mnt/taurus/home/siqiouyang/work/projects/sllama-dev/')

from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction

import numpy
import torch
import torch.nn.functional as F
import transformers

import conversation as conversation_lib
from conversation import SeparatorStyle
from eval.utils import disable_torch_init
from model.model import SpeechLlamaForCausalLM
from model.utils import KeywordsStoppingCriteria
from fairseq.data.audio.speech_to_text_dataset import _collate_frames

@entrypoint
class WaitkSpeechLlama(SpeechToTextAgent):
    """
    The agent generate the number of seconds from an input audio.
    """

    IGNORE_INDEX = -100
    DEFAULT_PAD_TOKEN = "[PAD]"
    DEFAULT_EOS_TOKEN = "</s>"
    DEFAULT_BOS_TOKEN = "<s>"
    DEFAULT_UNK_TOKEN = "<unk>"
    DEFAULT_SPEECH_TOKEN = "<speech>"
    DEFAULT_SPEECH_PATCH_TOKEN = "<sp_patch>"
    DEFAULT_SPEECH_START_TOKEN = "<sp_start>"
    DEFAULT_SPEECH_END_TOKEN = "<sp_end>"

    def __init__(self, args):
        super().__init__(args)
        self.waitk_lagging = args.waitk_lagging
        self.source_segment_size = args.source_segment_size
        # self.continuous_write = args.continuous_write
        self.prompt = args.prompt
        self.load_model(args.model_dir)

    def load_model(self, model_dir):
        load_type = torch.float16 # torch.float32
        disable_torch_init()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir,
            padding_side="right",
            use_fast=False,
        )
        self.model = SpeechLlamaForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=load_type,
            low_cpu_mem_usage=True,
            device_map='auto',
            config=os.path.join(model_dir, 'config_large.json'),
        ).eval()

        if 'model.embed_tokens' in self.model.hf_device_map.keys():
            device_input = 'cuda:' + str(self.model.hf_device_map['model.embed_tokens'])    
        else:
            device_input = 'cuda'  

        self.length_after_ssl, self.length_after_adp = self.model.model.initialize_speech_modules(
            speech_tower_path='/mnt/taurus/data/xixu/models/wav2_vec_vox_960h_pl.pt',
            speech_tower_type=None,
            len_adapter_channels=self.model.config.len_adapter_channels,
            len_adapter_kernel_sizes=self.model.config.len_adapter_kernel_sizes,
            ssl_fintuned=self.model.config.ssl_fintuned,
        )

        self.model.model.speech_tower.to(dtype=load_type, device=device_input)
        length_adapter_weights = torch.load(os.path.join(model_dir, 'length_adapter.bin'), map_location='cpu')
        mlp_adapter_weights = torch.load(os.path.join(model_dir, 'mlp_adapter.bin'), map_location='cpu')
        self.model.model.mm_length_adapter.load_state_dict(length_adapter_weights)
        self.model.model.mm_mlp_adapter.load_state_dict(mlp_adapter_weights)
        self.model.model.mm_length_adapter.to(dtype=load_type, device=device_input)
        self.model.model.mm_mlp_adapter.to(dtype=load_type, device=device_input)        


    @staticmethod
    def add_args(parser):
        parser.add_argument("--waitk-lagging", default=1, type=int)
        # parser.add_argument(
        #     "--continuous-write",
        #     default=1,
        #     type=int,
        #     help="Max number of words to write at each step",
        # )
        parser.add_argument("--model-dir", required=True, type=str)
        parser.add_argument(
            "--prompt", 
            default="<speech_here> Start by converting the English audio into Spanish written form.", 
            type=str
        )

    def policy(self, states: Optional[AgentStates] = None):
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

        source = torch.tensor(states.source).to(
            device=self.model.device, dtype=self.model.dtype
        )
        source = F.layer_norm(source, source.size())
        speech_batch = _collate_frames([source], is_audio_input=True)
        n_frames = torch.tensor([source.size(0)], dtype=torch.long)
        speech_lens = self.length_after_adp(self.length_after_ssl(n_frames))

        to_adds = [int(speech_len)*self.DEFAULT_SPEECH_PATCH_TOKEN for speech_len in speech_lens]
        to_adds = [self.DEFAULT_SPEECH_START_TOKEN + to_add + self.DEFAULT_SPEECH_END_TOKEN for to_add in to_adds]

        qs = self.prompt
        before, after = qs.split('<speech_here>')
        mm_prompts = [before + to_add + after for to_add in to_adds]

        conv = conversation_lib.default_conversation.copy()
        conv.messages = []
        conv.append_message(conv.roles[0], mm_prompts[0])
        conv.append_message(conv.roles[1], None)
        prompt_inputs = conv.get_prompt()

        prediction_ids = []
        prediction = []
        while len(prediction_ids) == 0 or states.source_finished:
            target_ids = self.tokenizer.encode(" ".join(states.target),add_special_tokens=False)
            inputs = self.tokenizer([prompt_inputs])
            input_ids = inputs.input_ids[0] + target_ids + prediction_ids
            input_ids = torch.as_tensor([input_ids]).cuda()

            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            
            output = self.model(
                input_ids=input_ids,
                speech_batch=speech_batch,
                src_lengths=n_frames.to(device=self.model.device),
                after_lens=speech_lens.to(device=self.model.device),
            )[0][0, -1]

            prediction_id = output.argmax().item()
            prediction_ids.append(prediction_id)
            prediction.extend(self.tokenizer.convert_ids_to_tokens([prediction_id]))
            
            if prediction[-1] == stop_str:
                break

        return WriteAction(
            content=self.tokenizer.decode(prediction_ids, skip_special_tokens=True),
            finished=states.source_finished and prediction[-1] == stop_str,
        )