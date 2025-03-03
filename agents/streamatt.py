from typing import Optional
from simuleval.utils import entrypoint

import torch

from agents.alignatt import AlignAtt, AlignAttStates as S2TAgentStates

import logging
logger = logging.getLogger(__name__)

@entrypoint
class StreamAtt(AlignAtt):
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

        return action