import os, json

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
from fairseq.examples.speech_to_text.data_utils import load_df_from_tsv

@dataclass
class S2TAgentStates(AgentStates):
    target_ids: list
    ref_target_ids: list
    position_ids: torch.Tensor

    def reset(self):
        super().reset()
        self.target_ids = []
        self.ref_target_ids = None
        self.position_ids = None


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
        transformers.set_seed(998244353)
        self.waitk_lagging = args.waitk_lagging
        self.n_word_per_input = args.n_word_per_input
        self.source_segment_size = args.source_segment_size
        # self.continuous_write = args.continuous_write
        self.prompt = args.prompt
        self.max_len_a = args.max_len_a
        self.max_len_b = args.max_len_b
        self.repeat_penalty = args.repeat_penalty
        self.uni = getattr(args, "uni", False)
        self.load_model(args.model_dir)
        if getattr(args, "force_target", False):
            self.load_benchmark_data(args.target)
        self.batch_size = args.batch_size
        self.test_instance_id = 0
        self.warmup = args.warmup
    
    def build_states(self):
        return S2TAgentStates([], None, None)

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

        if self.uni:
            replace_uni_train()

        self.model = SpeechLlamaForCausalLM.from_pretrained(
            model_dir,
            torch_dtype=load_type,
            low_cpu_mem_usage=True,
            device_map='auto',
            config=os.path.join(model_dir, 'config_large.json'),
        )

        if 'model.embed_tokens' in self.model.hf_device_map.keys():
            device_input = 'cuda:' + str(self.model.hf_device_map['model.embed_tokens'])    
        else:
            device_input = 'cuda'  

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

        self.model.model.mm_length_adapter.to(dtype=load_type, device=device_input)
        self.model.model.mm_mlp_adapter.to(dtype=load_type, device=device_input)     
        self.model.model.speech_tower.to(dtype=load_type, device=device_input)

        self.model.eval()
        self.model.model.config.inference = True

    def load_benchmark_data(self, target_file):
        with open(target_file, 'r') as r:
            target_texts = [line.strip() for line in r.readlines() if line.strip() != '']
        self.tgt_id_segs = []
        for t in target_texts:
            tgt_id = self.tokenizer(
                [t],
                truncation=False,
            )['input_ids'][0][1:]

            tgt_id_seg = []
            i = 0
            while i < len(tgt_id):
                j = i + 1
                while j < len(tgt_id) and \
                    not self.tokenizer.convert_ids_to_tokens([tgt_id[j]])[0].startswith('▁'):
                    j += 1
                tgt_id_seg.append(tgt_id[i : j])
                i = j
            self.tgt_id_segs.append(tgt_id_seg)

    @staticmethod
    def add_args(parser):
        parser.add_argument("--waitk-lagging", default=1, type=int)
        parser.add_argument("--n-word-per-input", default=1, type=int)
        parser.add_argument(
            "--model-dir", 
            required=True, 
            type=str
        )
        parser.add_argument(
            "--uni", 
            action="store_true"
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
            "--force-target",
            action="store_true"
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1
        )
        parser.add_argument(
            "--warmup",
            type=int,
            default=0,
        )

    def policy(self, states: Optional[S2TAgentStates] = None):
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
            ) < self.waitk_lagging + self.warmup:
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
        self.model.model.speech_features_extracted = False
            
        inputs = self.tokenizer([prompt_inputs])
        input_ids = inputs.input_ids[0] + states.target_ids + prediction_ids
        input_ids_tensor = torch.as_tensor([input_ids]).cuda()
        n_word = self.n_word_per_input
        if length_in_seconds == self.waitk_lagging + self.warmup:
            n_word += self.warmup * self.n_word_per_input

        stopping_criteria = SpaceStoppingCriteria(self.tokenizer, n_word)
        with torch.inference_mode():
            # output = self.model(
            #     input_ids=input_ids_tensor,
            #     speech_batch=speech_batch,
            #     src_lengths=n_frames.to(device=self.model.device),
            #     after_lens=speech_lens.to(device=self.model.device),
            # )[0][0, -1]

            output_ids = self.model.generate(
                attention_mask=input_ids_tensor.ne(self.tokenizer.pad_token_id).repeat(self.batch_size, 1),
                input_ids=input_ids_tensor.repeat(self.batch_size, 1),
                speech_batch=speech_batch.repeat(self.batch_size, 1),
                src_lengths=n_frames.to(device=self.model.device).repeat(self.batch_size),
                after_lens=speech_lens.to(device=self.model.device).repeat(self.batch_size),
                do_sample=False,
                num_beams=1,
                max_new_tokens=max_number_of_tokens - len(states.target_ids) - len(prediction_ids),
                repetition_penalty=self.repeat_penalty,
                stopping_criteria=[stopping_criteria] if not states.source_finished else None,
                states=states,
            )
        if not states.source_finished and \
            (stopping_criteria(output_ids, None) or output_ids[0, -1] == self.tokenizer.eos_token_id):
            output_ids = output_ids[:, :-1]
            states.position_ids = states.position_ids[:, :-1]

        # if getattr(self, "prof", None) is not None:
        #     self.prof.step()

        input_token_len = input_ids_tensor.shape[1]
        prediction_id = output_ids[0, input_token_len:].tolist()
        prediction_ids.extend(prediction_id)
        
        states.target_ids.extend(prediction_ids)
        possible_full_word = self.tokenizer.decode(prediction_ids, skip_special_tokens=True)

        if states.source_finished:
            self.test_instance_id += 1
            states.ref_target_ids = None

            # if getattr(self, "prof", None):
            #     self.prof.stop()

            # self.prof = torch.profiler.profile(
            #     schedule=torch.profiler.schedule(wait=0, warmup=1, active=100, repeat=1),
            #     on_trace_ready=torch.profiler.tensorboard_trace_handler("profile/w2v2_llama2/recomp_llama2/#{}".format(self.test_instance_id)),
            # )
            # self.prof.start()

        if possible_full_word != '' or states.source_finished:
            return WriteAction(
                content=possible_full_word,
                finished=states.source_finished,
            )
        else:
            return ReadAction()
