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
    ref_target_ids: list
    position_ids: torch.Tensor
    most_attended_indices: torch.Tensor

    def reset(self):
        super().reset()
        self.target_ids = []
        self.ref_target_ids = None
        self.position_ids = None
        self.most_attend_pos = torch.LongTensor([])


@entrypoint
class AlignAtt(SpeechToTextAgent):
    """
    The agent generate the number of seconds from an input audio.
    https://www.isca-archive.org/interspeech_2023/papi23_interspeech.pdf
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
        self.frame_num = args.frame_num
        self.attn_layer = args.attn_layer
        self.min_start_sec = args.min_start_sec
        self.source_segment_size = args.source_segment_size
        # self.continuous_write = args.continuous_write
        self.prompt = args.prompt
        self.max_len_a = args.max_len_a
        self.max_len_b = args.max_len_b
        self.load_model(args.model_dir)
        if getattr(args, "force_target", False):
            self.load_benchmark_data(args.target)
        self.batch_size = args.batch_size
        self.no_repeat_ngram_size = args.no_repeat_ngram_size
        self.test_instance_id = 0

    def build_states(self):
        return S2TAgentStates([], None, None, torch.LongTensor([]))

    def load_model(self, model_dir):
        load_type = torch.float16
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
            config=os.path.join(model_dir, 'config_large.json'),
        )

        if 'model.embed_tokens' in self.model.hf_device_map.keys():
            self.device = 'cuda:' + str(self.model.hf_device_map['model.embed_tokens'])    
        else:
            self.device = 'cuda'  

        self.length_after_ssl, self.length_after_adp = self.model.model.initialize_speech_modules(
            speech_tower_path='/data/user_data/siqiouya/runs/pretrained/wav2_vec_vox_960h_pl.pt',
            speech_tower_type=None,
            len_adapter_channels=self.model.config.len_adapter_channels,
            len_adapter_kernel_sizes=self.model.config.len_adapter_kernel_sizes,
            ssl_fintuned=self.model.config.ssl_fintuned,
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

        self.model.eval()
        self.model.model.config.inference = True


    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--source-lang", 
            type=str,
            default='English',
        )
        parser.add_argument(
            "--target-lang", 
            type=str,
            default='German',
        )
        parser.add_argument(
            "--frame-num",
            default=1, 
            type=int,
        )
        parser.add_argument(
            "--attn-layer",
            type=int,
            default=-1
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
            "--force-target",
            action="store_true"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1
        )
        parser.add_argument(
            "--no-repeat-ngram-size",
            type=int,
            default=3,
        )

    @torch.inference_mode()
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

        speech_tokens = [int(speech_len)*self.DEFAULT_SPEECH_PATCH_TOKEN for speech_len in speech_lens]
        speech_tokens = [self.DEFAULT_SPEECH_START_TOKEN + tokens + self.DEFAULT_SPEECH_END_TOKEN for tokens in speech_tokens]

        instruction = f"Translate the following speech from {self.args.source_lang} to {self.args.target_lang}:"
        prompts = [f"{instruction} {tokens}" for tokens in speech_tokens]

        max_number_of_tokens = int(length_in_seconds * self.max_len_a + self.max_len_b)

        self.model.model.speech_features_extracted = False
        inputs = self.tokenizer(prompts)

        stop_str = "<|end_of_text|>"
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, torch.tensor(inputs.input_ids)
        )
                
        input_ids = inputs.input_ids[0] + states.target_ids
        input_ids_tensor = torch.as_tensor([input_ids]).cuda()


        output = self.model.generate(
            attention_mask=input_ids_tensor.ne(self.tokenizer.pad_token_id).repeat(self.batch_size, 1),
            input_ids=input_ids_tensor.repeat(self.batch_size, 1),
            speech_batch=speech_batch.repeat(self.batch_size, 1),
            src_lengths=n_frames.to(device=self.model.device).repeat(self.batch_size),
            after_lens=speech_lens.to(device=self.model.device).repeat(self.batch_size),
            do_sample=False,
            num_beams=1,
            max_new_tokens=max(1, max_number_of_tokens - len(states.target_ids)),
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            stopping_criteria=[stopping_criteria],
            pad_token_id=self.tokenizer.eos_token_id,
            output_attentions=True,
            return_dict_in_generate=True,
            states=states,
        )
        
        output_ids = output.sequences[0, input_ids_tensor.size(1):]

        if not states.source_finished:
            attentions = output.attentions

            speech_start_pos = torch.where(input_ids_tensor[0] == self.model.config.sp_start_token_id)[0] + 1
            speech_end_pos = torch.where(input_ids_tensor[0] == self.model.config.sp_end_token_id)[0]

            states.most_attended_indices = []
            for i in range(len(output_ids)):
                if self.attn_layer == -1:
                    sum_att = attentions[i][0][0].mean(dim=0)[-1, speech_start_pos : speech_end_pos]
                    for j in range(1, len(attentions[i])):
                        sum_att += attentions[i][j][0].mean(dim=0)[-1, speech_start_pos : speech_end_pos]
                else:
                    sum_att = attentions[i][self.attn_layer][0].mean(dim=0)[-1, speech_start_pos : speech_end_pos]
                most_attended_idx = sum_att.argmax()
                if speech_start_pos + most_attended_idx >= speech_end_pos - self.frame_num:
                    break
                states.most_attended_indices.append(most_attended_idx * 1280)
            states.most_attended_indices = torch.LongTensor(states.most_attended_indices)

            prediction_ids = output_ids[:states.most_attended_indices.size(0)]
        else:
            prediction_ids = output_ids

        if not states.source_finished:
            if len(prediction_ids) > 0:
                if prediction_ids[-1] == self.tokenizer.eos_token_id:
                    prediction_ids = prediction_ids[:-1]
                else:
                    for i in range(len(prediction_ids)):
                        if self.tokenizer.convert_ids_to_tokens([prediction_ids[len(prediction_ids) - i - 1]])[0].startswith('▁'):
                            prediction_ids = prediction_ids[:len(prediction_ids) - i - 1]
                            break
        
        states.target_ids.extend(prediction_ids)
        possible_full_word = self.tokenizer.decode(prediction_ids, skip_special_tokens=True).strip()

        # print(self.tokenizer.decode(states.target_ids))

        if states.source_finished:
            self.test_instance_id += 1
            states.ref_target_ids = None

        if possible_full_word != '' or states.source_finished:
            return WriteAction(
                content=possible_full_word,
                finished=states.source_finished,
            )
        else:
            return ReadAction()
