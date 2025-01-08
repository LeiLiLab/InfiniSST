from typing import Optional
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.agents.actions import WriteAction, ReadAction
from dataclasses import dataclass
from model.utils import KeywordsStoppingCriteria, SpaceStoppingCriteria
import torch
import transformers
from train.dataset import (
    DEFAULT_SPEECH_PATCH_TOKEN,
    DEFAULT_SPEECH_START_TOKEN,
    DEFAULT_SPEECH_END_TOKEN
)
from eval.agents.streamllama import StreamLlama, S2TAgentStates

@entrypoint
class WaitkSpeechLlama3(StreamLlama):
    def __init__(self, args):
        super().__init__(args)
        self.waitk_lagging = args.waitk_lagging
        self.n_word_per_input = args.n_word_per_input
        self.warmup = args.warmup
        self.test_instance_id = 0
        if getattr(args, "force_target", False):
            self.load_benchmark_data(args.target)

    @staticmethod
    def add_args(parser):
        StreamLlama.add_args(parser)
        parser.add_argument("--waitk-lagging", default=1, type=int)
        parser.add_argument("--n-word-per-input", default=1, type=int)
        parser.add_argument("--warmup", type=int, default=0)
        parser.add_argument("--force-target", action="store_true")

    def _prepare_inputs_offline(self, states, speech_lens):
        messages = []
        if states.speech_cache is None:
            messages.append(
                {
                    "role": "system",
                    "content": f"Translate the following speech from {self.source_lang} to {self.target_lang}."
                }
            )
        messages.append(
            {
                "role": "user",
                "content": speech_lens[0] * DEFAULT_SPEECH_PATCH_TOKEN
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": self.tokenizer.decode(states.target_ids, skip_special_tokens=True).strip(),
            }
        )
        input_ids = self.tokenizer.apply_chat_template(
            [messages],
            return_tensors='pt',
            padding=True, 
            truncation=False, 
            add_special_tokens=False
        )[:, :-1]
        input_ids = input_ids.cuda()
        return input_ids
    
    def policy(self, states: Optional[S2TAgentStates] = None):
        if states is None:
            states = self.states

        if states.source_sample_rate == 0:
            length_in_seconds = 0
        else:
            length_in_seconds = float(len(states.source)) / states.source_sample_rate

        if not states.source_finished:
            if (length_in_seconds * 1000 / self.source_segment_size) < self.waitk_lagging + self.warmup:
                return ReadAction()

        if states.source_finished and length_in_seconds < 0.32:
            self.test_instance_id += 1
            return WriteAction(content="", finished=True)
        prediction_ids = []

        # Reuse speech processing from StreamLlama
        speech_batch, n_frames, speech_lens = self._prepare_speech(states)
        input_ids = self._prepare_inputs_offline(states, speech_lens)
        # print(input_ids)
        # target_ids = torch.tensor(states.target_ids).unsqueeze(0)                  
        # prediction_ids = torch.tensor(prediction_ids).unsqueeze(0)                   
        # input_ids_tensor = torch.cat([input_ids, target_ids, prediction_ids], dim=1).cuda()
        # print(input_ids_tensor)
        max_number_of_tokens = int(length_in_seconds * self.max_len_a + self.max_len_b)

        stop_str = "<|eot_id|>"
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        n_word = self.n_word_per_input
        if length_in_seconds == self.waitk_lagging + self.warmup:
            n_word += self.warmup * self.n_word_per_input

        stopping_criteria = SpaceStoppingCriteria(self.tokenizer, n_word)
        
        self.model.model.speech_features_extracted = False
        outputs = self.model.generate(
            attention_mask=None,
            input_ids=input_ids,
            speech_batch=speech_batch,
            src_lengths=n_frames,
            after_lens=speech_lens,
            do_sample=False,
            num_beams=self.beam,
            max_new_tokens=max(1, max_number_of_tokens - len(states.target_ids) - len(prediction_ids)),
            repetition_penalty=self.repetition_penalty,
            stopping_criteria=[stopping_criteria] if not states.source_finished else None,
            pad_token_id=self.tokenizer.pad_token_id,
            states=states,
        )

        if not states.source_finished and \
            (stopping_criteria(outputs, None) or outputs[0, -1] == self.tokenizer.eos_token_id):
            outputs = outputs[:, :-1]

        # states.past_key_values = outputs.past_key_values
        input_token_len = input_ids.shape[1]
        prediction_id = outputs[0, input_token_len:].tolist()
        prediction_ids.extend(prediction_id)

        states.target_ids.extend(prediction_ids)
        translation = self.tokenizer.decode(prediction_ids, skip_special_tokens=True).strip()
        full_translation = self.tokenizer.decode(states.target_ids, skip_special_tokens=True).strip()
        print("prediction", translation)
        print("full_translation", full_translation)


        if translation != '' or states.source_finished:
            return WriteAction(
                content=translation,
                finished=states.source_finished,
            )
        else:
            return ReadAction() 