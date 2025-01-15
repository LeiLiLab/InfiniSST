from typing import Optional
from simuleval.utils import entrypoint
from simuleval.agents.actions import WriteAction, ReadAction
from dataclasses import dataclass
import torch
import transformers
from fairseq.data.audio.speech_to_text_dataset import _collate_frames
from model.utils import KeywordsStoppingCriteria
from eval.agents.streamllama import StreamLlama, S2TAgentStates
from train.dataset import (
    DEFAULT_SPEECH_PATCH_TOKEN,
    DEFAULT_SPEECH_START_TOKEN,
    DEFAULT_SPEECH_END_TOKEN
)

@dataclass
class AlignAttStates(S2TAgentStates):
    most_attended_indices: torch.Tensor = torch.LongTensor([])

    def reset(self):
        super().reset()
        self.most_attended_indices = torch.LongTensor([])

@entrypoint
class AlignAttSpeechLlama3(StreamLlama):
    def __init__(self, args):
        super().__init__(args)
        self.frame_num = args.frame_num
        self.attn_layer = args.attn_layer
        self.min_start_sec = args.min_start_sec
        self.test_instance_id = 0
        if getattr(args, "force_target", False):
            self.load_benchmark_data(args.target)

    def build_states(self):
        states = super().build_states()
        # Create a new state with all the fields from the parent state
        new_states = AlignAttStates(*vars(states).values())
        return new_states

    @staticmethod
    def add_args(parser):
        StreamLlama.add_args(parser)
        parser.add_argument("--frame-num", default=1, type=int)
        parser.add_argument("--attn-layer", type=int, default=-1)
        parser.add_argument("--min-start-sec", default=0.32, type=float)
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

    def policy(self, states: Optional[AlignAttStates] = None):
        if states is None:
            states = self.states

        if states.source_sample_rate == 0:
            length_in_seconds = 0
        else:
            length_in_seconds = float(len(states.source)) / states.source_sample_rate

        if not states.source_finished and length_in_seconds < self.min_start_sec:
            return ReadAction()

        if states.source_finished and length_in_seconds < 0.32:
            self.test_instance_id += 1
            return WriteAction(content="", finished=True)

        source = torch.tensor(states.source).to(
            device=self.model.device, dtype=self.model.dtype
        )
        speech_batch = _collate_frames([source], is_audio_input=True)
        n_frames = torch.tensor([source.size(0)], dtype=torch.long)
        speech_lens = self.length_shrink_func(n_frames)
        input_ids = self._prepare_inputs_offline(states, speech_lens)
        max_number_of_tokens = int(length_in_seconds * self.max_len_a + self.max_len_b)

        self.model.model.speech_features_extracted = False
        outputs = self.model.generate(
            attention_mask=None,
            input_ids=input_ids,
            speech_batch=speech_batch,
            src_lengths=n_frames,
            after_lens=speech_lens,
            do_sample=False,
            num_beams=self.beam,
            max_new_tokens=max(1, max_number_of_tokens - len(states.target_ids)),
            no_repeat_ngram_size=self.no_repeat_ngram_size,
            repetition_penalty=self.repetition_penalty,
            pad_token_id=self.tokenizer.pad_token_id,
            output_attentions=True,
            return_dict_in_generate=True,
        )

        input_token_len = input_ids.shape[1]
        output_ids = outputs.sequences[0, input_token_len:]
        
        if not states.source_finished:
            attentions = outputs.attentions
            speech_start_pos = torch.where(input_ids[0] == 128256)[0][0].item() + 1
            speech_end_pos = torch.where(input_ids[0] == 128256)[0][-1].item()

            states.most_attended_indices = []
            for i in range(len(output_ids)):
                if self.attn_layer == -1:
                    sum_att = attentions[i][0][0].mean(dim=0)[-1, speech_start_pos:speech_end_pos]
                    for j in range(1, len(attentions[i])):
                        sum_att += attentions[i][j][0].mean(dim=0)[-1, speech_start_pos:speech_end_pos]
                else:
                    sum_att = attentions[i][self.attn_layer][0].mean(dim=0)[-1, speech_start_pos:speech_end_pos]
                most_attended_idx = sum_att.argmax()
                if speech_start_pos + most_attended_idx >= speech_end_pos - self.frame_num:
                    break
                states.most_attended_indices.append(most_attended_idx * 1280)
            states.most_attended_indices = torch.LongTensor(states.most_attended_indices)

            prediction_ids = output_ids[:states.most_attended_indices.size(0)]
        else:
            prediction_ids = output_ids

        states.target_ids.extend(prediction_ids)
        translation = self.tokenizer.decode(prediction_ids, skip_special_tokens=True).strip()

        if translation != '' or states.source_finished:
            return WriteAction(
                content=translation,
                finished=states.source_finished,
            )
        else:
            return ReadAction() 