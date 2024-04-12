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

@dataclass
class S2TAgentStates(AgentStates):
    target_ids: list

    def reset(self):
        super().reset()
        self.target_ids = []


@entrypoint
class RALCP(SpeechToTextAgent):
    """
    The agent generate the number of seconds from an input audio.
    https://arxiv.org/pdf/2309.06706.pdf
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
        transformers.set_seed(998244353)
        self.agreement_threshold = args.agreement_threshold
        self.beam = args.beam
        self.min_start_sec = args.min_start_sec
        self.source_segment_size = args.source_segment_size
        # self.continuous_write = args.continuous_write
        self.prompt = args.prompt
        self.max_len_a = args.max_len_a
        self.max_len_b = args.max_len_b
        self.repeat_penalty = args.repeat_penalty
        self.speech_tower_path = args.speech_tower_path
        self.speech_tower_type = args.speech_tower_type
        self.load_model(args.model_dir)
 
    
    def build_states(self):
        return S2TAgentStates([])

    def load_model(self, model_dir):
        load_type = torch.float16
        # load_type = torch.float32
        disable_torch_init()
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir,
            padding_side="right",
            use_fast=False,
        )

        if not os.path.exists(os.path.join(model_dir, 'config_large.json')):
            config = json.load(open(os.path.join(model_dir, 'config.json')))
            config['large_model'] = True
            update_config = os.path.join(model_dir, 'config_large.json')
            json.dump(config, open(update_config, 'w'), indent=2)

        self.model = SpeechLlamaForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=load_type,
            low_cpu_mem_usage=True,
            device_map='auto',
            config=os.path.join(model_dir, 'config_large.json'),).eval()

        if torch.cuda.is_available():
            if 'model.embed_tokens' in self.model.hf_device_map.keys():
                self.device = 'cuda:' + str(self.model.hf_device_map['model.embed_tokens'])    
            else:
                self.device = 'cuda'  
        else:
            self.device = 'cpu'  

        # self.length_after_ssl, self.length_after_adp = self.model.model.initialize_speech_modules(
        #     speech_tower_path='/mnt/taurus/data/xixu/models/wav2_vec_vox_960h_pl.pt',
        #     speech_tower_type=None,
        #     len_adapter_channels=self.model.config.len_adapter_channels,
        #     len_adapter_kernel_sizes=self.model.config.len_adapter_kernel_sizes,
        #     ssl_fintuned=self.model.config.ssl_fintuned,
        # )
            
        self.model.model.length_after_ssl, self.model.model.length_after_adp = self.model.model.initialize_speech_modules(
            speech_tower_path=self.speech_tower_path,
            speech_tower_type=self.speech_tower_type,
            len_adapter_channels=self.model.config.len_adapter_channels,
            len_adapter_kernel_sizes=self.model.config.len_adapter_kernel_sizes,
            ssl_fintuned=self.model.config.ssl_fintuned,
            stage1_complete=self.model.config.stage1_complete,
        )

        length_adapter_weights = torch.load(os.path.join(model_dir, 'length_adapter.bin'), map_location='cpu')
        mlp_adapter_weights = torch.load(os.path.join(model_dir, 'mlp_adapter.bin'), map_location='cpu')
        speech_tower_weights = torch.load(os.path.join(model_dir, 'speech_tower.bin'), map_location='cpu')

        self.model.model.mm_length_adapter.load_state_dict(length_adapter_weights)
        self.model.model.mm_mlp_adapter.load_state_dict(mlp_adapter_weights)
        self.model.model.speech_tower.load_state_dict(speech_tower_weights)

        self.model.model.mm_length_adapter.to(dtype=load_type, device=self.device)
        self.model.model.mm_mlp_adapter.to(dtype=load_type, device=self.device)     
        self.model.model.speech_tower.to(dtype=load_type, device=self.device)

        self.model.model.config.inference = True


    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--agreement-threshold", 
            default=1.0, 
            type=float,
        )
        parser.add_argument(
            "--beam",
            default=1,
            type=int,
        )
        parser.add_argument(
            "--min-start-sec",
            default=0.32,
            type=float,
        )
        parser.add_argument(
            "--model-dir", 
            required=True, 
            type=str
        )
        parser.add_argument(
            "--prompt", 
            default="<speech_here> Start by converting the English audio into Spanish written form.", 
            type=str
        )
        parser.add_argument(
            "--max-len-a",
            type=int,
            default=5,
            help="Max number of tokens generated per second"
        )
        parser.add_argument(
            "--max-len-b",
            type=int,
            default=20,
            help="Max number of tokens generated additionally"
        )
        parser.add_argument(
            "--repeat-penalty",
            type=float,
            default=1.0,
            help="Repetition penalty for generation"
        )
        parser.add_argument(
            "--speech-tower-path",
            type=str,
            default=None,
            help="Path to the speech tower"
        )
        parser.add_argument(
            "--speech-tower-type",
            type=str,
            default=None,
            help="Type of the speech tower"
        )

    def policy(self, states: Optional[S2TAgentStates] = None):
        if states is None:
            states = self.states

        if states.source_sample_rate == 0:
            # empty source, source_sample_rate not set yet
            length_in_seconds = 0
        else:
            length_in_seconds = float(len(states.source)) / states.source_sample_rate

        if not states.source_finished and length_in_seconds < self.min_start_sec:
            return ReadAction()
            
        # if states.source_finished and length_in_seconds < 0.32:
        #     return WriteAction(content="", finished=True)

        source = torch.tensor(states.source).to(
            device=self.model.device, dtype=self.model.dtype
        )
        source = F.layer_norm(source, source.size())
        speech_batch = _collate_frames([source], is_audio_input=True)
        n_frames = torch.tensor([source.size(0)], dtype=torch.long)
        speech_lens = self.model.model.length_after_adp(self.model.model.length_after_ssl(n_frames))

        to_adds = [int(speech_len)*self.DEFAULT_SPEECH_PATCH_TOKEN for speech_len in speech_lens]
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
        # self.model.model.speech_features_extracted = False # appears bug when move it here
        inputs = self.tokenizer([prompt_inputs])
        input_ids = inputs.input_ids[0] + states.target_ids
        input_ids_tensor = torch.as_tensor([input_ids]).to(self.device)

        self.model.model.speech_features_extracted = False
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids_tensor)
        self.model.eval()
        with torch.inference_mode():
            output_ids = self.model.generate(
                attention_mask=input_ids_tensor.ne(self.tokenizer.pad_token_id),
                input_ids=input_ids_tensor,
                speech_batch=speech_batch.to(device=self.device),
                src_lengths=n_frames.to(device=self.device),
                after_lens=speech_lens.to(device=self.device),
                do_sample=False,
                num_beams=self.beam,
                num_return_sequences=self.beam,
                max_new_tokens=30,
                # repetition_penalty=self.repeat_penalty,
                stopping_criteria=[stopping_criteria]
            )

        input_token_len = input_ids_tensor.shape[1]
        if states.source_finished:
            prediction_ids.extend(output_ids[0, input_token_len:].tolist())
        else:
            prediction_id_tensor = output_ids[:, input_token_len:]

            finished = [False] * self.beam
            prediction_id = []
            for i in range(prediction_id_tensor.size(1)):
                cnt = Counter()
                max_cnt = max_id = -1
                for j, id in enumerate(prediction_id_tensor[:, i]):
                    id = id.item()
                    if not finished[j]:
                        cnt[id] += 1
                        if cnt[id] > max_cnt:
                            max_cnt = cnt[id]
                            max_id = id
                    if id == self.tokenizer.eos_token_id:
                        finished[j] = True
                if max_cnt / self.beam < self.agreement_threshold:
                    break
                for j, id in enumerate(prediction_id_tensor[:, i]):
                    if id != max_id:
                        finished[j] = True
                prediction_id.append(max_id)
                if max_id == self.tokenizer.eos_token_id:
                    break

            if len(prediction_id) > 0 and prediction_id[-1] == self.tokenizer.eos_token_id and not states.source_finished:
                prediction_id = prediction_id[:-1]

            prediction_ids.extend(prediction_id)
        
        states.target_ids.extend(prediction_ids)
        possible_full_word = self.tokenizer.decode(prediction_ids, skip_special_tokens=True)
        possible_full_word = possible_full_word.strip()

        if possible_full_word != '' or states.source_finished:
            return WriteAction(
                content=possible_full_word,
                finished=states.source_finished,
            )
        else:
            return ReadAction()
