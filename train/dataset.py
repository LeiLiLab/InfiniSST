# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import csv
import io
import logging
import re
import time
import random
import copy
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from tqdm import tqdm

import jieba
import transformers
import numpy as np
import torch
from torch.utils.data import DistributedSampler
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    FairseqDataset,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from fairseq.data.audio.audio_utils import (
    get_fbank,
    get_waveform,
    read_from_stored_zip,
    is_npy_data,
    is_sf_audio_data,
    parse_path,
    FEATURE_OR_SF_AUDIO_FILE_EXTENSIONS,
)
from fairseq.data.audio.feature_transforms import CompositeAudioFeatureTransform
from fairseq.data.audio.data_cfg import S2TDataConfig
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset, _collate_frames

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
DEFAULT_SPEECH_TOKEN = "<speech>"
DEFAULT_SPEECH_PATCH_TOKEN = "<sp_patch>"
DEFAULT_SPEECH_START_TOKEN = "<sp_start>"
DEFAULT_SPEECH_END_TOKEN = "<sp_end>"
DEFAULT_TEXT_END_TOKEN = "<text_end>"
DEFAULT_LATENCY_TOKEN = "<latency_{}>"

logger = logging.getLogger(__name__)

def get_features_or_waveform(
        path: str,
):
    import soundfile as sf
    _path, slice_ptr = parse_path(path)
    if len(slice_ptr) == 0:
        waveform, sample_rate = sf.read(_path, dtype="float32",)
    elif len(slice_ptr) == 2:
        waveform, sample_rate = sf.read(_path, dtype="float32",
                                start=int(slice_ptr[0]), frames=int(slice_ptr[1]))
    else:
        raise ValueError(f"Invalid path: {_path}")
    return waveform, sample_rate

@dataclass
class SpeechToTextDatasetItem(object):
    id: None
    index: int
    source: torch.Tensor
    task: None
    src_text: None
    target: Optional[torch.Tensor] = None
    speech_word: Optional[List] = None
    text_word: Optional[List] = None
    trajectory: Optional[List] = None
    sampled_trajectory: Optional[List] = None
    
class PromptSpeechToTextDataset(SpeechToTextDataset):

    def __init__(
        self,
        audio_paths: List[str],
        n_frames: Optional[List[int]] = None,
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        speech_words: Optional[List] = None,
        text_words: Optional[List] = None,
        trajectories: Optional[List[List[str]]] = None,
        sampled_trajectories: Optional[List[List[str]]] = None,
    ):
        self.audio_paths = audio_paths
        self.n_frames = n_frames
        self.tgt_texts = tgt_texts
        self.src_texts = src_texts
        self.ids = ids
        self.tasks = tasks
        self.speech_words = speech_words
        self.text_words = text_words
        self.trajectories = trajectories
        self.sampled_trajectories = sampled_trajectories       
    
    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        while True:
            try:
                source, sr = get_features_or_waveform(
                    self.audio_paths[index],
                )
                break  # Exit the loop if successful
            except Exception as e:
                time.sleep(random.uniform(0, 1))  # Sleep for a random time <= 1 second

        source = torch.from_numpy(source).float()
        # with torch.no_grad():
        #     source = F.layer_norm(source, source.shape)
        text = self.tgt_texts[index]
        id = self.ids[index]
        task = self.tasks[index]
        src_text = self.src_texts[index]
        speech_word = self.speech_words[index] if self.speech_words is not None else None
        text_word = self.text_words[index] if self.text_words is not None else None
        trajectory = self.trajectories[index] if self.trajectories is not None else None
        sampled_trajectory = self.sampled_trajectories[index] if self.sampled_trajectories is not None else None
        return SpeechToTextDatasetItem(
            index=index, source=source, target=text, src_text=src_text, id=id, task=task,
            speech_word=speech_word, text_word=text_word, 
            trajectory=trajectory, sampled_trajectory=sampled_trajectory
        )
    
    def __len__(self):
        return len(self.audio_paths)
        
class PromptSpeechToTextDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_AUDIO, KEY_N_FRAMES = "id", "audio", "n_frames"
    KEY_TGT_TEXT = "tgt_text"
    # optional columns
    KEY_SPEAKER, KEY_SRC_TEXT = "speaker", "src_text"
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    KEY_TRAJECTORY = "trajectory"
    # default values
    DEFAULT_SPEAKER = DEFAULT_SRC_TEXT = DEFAULT_TGT_TEXT = DEFAULT_LANG = DEFAULT_LANG_N_FRAMES = DEFAULT_TASK = ""
    TASK = "task"

    @classmethod
    def _load_samples_from_tsv(cls, root: str, split: str):
        tsv_path = Path(root) / f"{split}.tsv"
        if not tsv_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {tsv_path}")
        with open(tsv_path) as f:
            reader = csv.DictReader(
                f,
                delimiter="\t",
                quotechar=None,
                doublequote=False,
                lineterminator="\n",
                quoting=csv.QUOTE_NONE,
            )
            samples = [dict(e) for e in reader]
        if len(samples) == 0:
            raise ValueError(f"Empty manifest: {tsv_path}")
        return samples

    @classmethod
    def from_tsv(
        cls,
        root: str,
        split: str,
    ) -> PromptSpeechToTextDataset:
        samples = cls._load_samples_from_tsv(root, split)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [s[cls.KEY_AUDIO] for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s.get(cls.KEY_TGT_TEXT, cls.DEFAULT_TGT_TEXT) for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        tasks = [s.get(cls.TASK, cls.DEFAULT_TASK) for s in samples] 

        speech_words_str = [s.get('speech_word', '') for s in samples]
        text_words_str = [s.get('text_word', '') for s in samples]
        speech_words = [eval(s) if s != '' else None for s in speech_words_str]
        text_words = [eval(s) if s != '' else None for s in text_words_str]          

        trajectories_str = [s.get(cls.KEY_TRAJECTORY, '') for s in samples]
        trajectories = [eval(s) if s != '' else None for s in trajectories_str]

        sampled_trajectories_str = [s.get('sampling', '') for s in samples]
        sampled_trajectories = [eval(s) if s != '' else None for s in sampled_trajectories_str]

        return PromptSpeechToTextDataset(
            audio_paths,
            n_frames=n_frames,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            ids=ids,
            tasks=tasks,
            speech_words=speech_words,
            text_words=text_words,
            trajectories=trajectories,
            sampled_trajectories=sampled_trajectories
        )


class SpeechSampler(DistributedSampler):
    def __init__(self, dataset, shuffle, batch_size, batch_size_sent=30, min_ms=0, multiplier=1, filter=True, tokenizer=None, model_type="w2v2_llama31"):
        super().__init__(dataset=dataset, shuffle=shuffle)
        self.batch_size = batch_size
        self.batch_size_sent = batch_size_sent
        self.model_type = model_type
        self._obtain_batches(min_ms, multiplier, filter, tokenizer)

    def get_eff_size(self, idx, tokenizer):
        if self.model_type == "w2v2_llama31":
            sp_seg_frame = int(12 * 0.08 * 16000)
            n_seg = (self.dataset.n_frames[idx] + sp_seg_frame - 1) // sp_seg_frame
            eff_size = n_seg * 5 * 2 # headers
            eff_size += n_seg * 12 # speech features
            eff_size += len(tokenizer(self.dataset.tgt_texts[idx], add_special_tokens=False).input_ids) # text tokens
            eff_size += 39 # beginning prompt
        elif self.model_type == "qwen2ac":
            sp_seg_frame = 16000
            n_seg = (self.dataset.n_frames[idx] + sp_seg_frame - 1) // sp_seg_frame
            eff_size = n_seg * 5 * 2 # headers
            eff_size += n_seg * 25 # speech features
            eff_size += len(tokenizer(self.dataset.tgt_texts[idx], add_special_tokens=False).input_ids) # text tokens
            eff_size += 14 # beginning prompt
        return eff_size

    def _obtain_batches(self, min_ms, multiplier, filter, tokenizer):
        eff_sizes = []
        for idx in range(len(self.dataset)):
            eff_size = self.get_eff_size(idx, tokenizer)
            eff_sizes.append((eff_size, idx))

        sorted_eff_sizes = sorted(eff_sizes)

        batch_indices = []
        indices = []
        n_skipped = 0
        for eff_size, idx in sorted_eff_sizes:
            if not filter or self.dataset.n_frames[idx] >= min_ms * 16:
                if eff_size * (len(indices) + 1) <= self.batch_size and len(indices) < self.batch_size_sent:
                    indices.append(idx)
                else:
                    batch_indices.append(indices)
                    indices = [idx]
            else:
                n_skipped += 1
        print('{} out of {} samples skipped'.format(n_skipped, len(sorted_eff_sizes)))
        assert len(indices) > 0
        batch_indices.append(indices)
        
        n_batches = len(batch_indices)
        n_batches = n_batches // multiplier * multiplier

        self.batch_indices = batch_indices[:n_batches][::-1]
    
    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices_batch_ind = torch.randperm(len(self.batch_indices), generator=g).tolist()
        else:
            indices_batch_ind = list(range(len(self.batch_indices)))

        indices_batch_ind = indices_batch_ind[self.rank:len(self):self.num_replicas]

        for i in indices_batch_ind:
            yield self.batch_indices[i]
        
    def __len__(self):
        return len(self.batch_indices)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer, length_shrink_func, source_lang, target_lang, **kwargs):
        self.tokenizer = tokenizer
        self.length_shrink_func = length_shrink_func
        self.source_lang = source_lang
        self.target_lang = target_lang
     
    def __call__(self, samples: List[SpeechToTextDatasetItem]) -> Dict[str, torch.Tensor]:
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        speech_batch = _collate_frames([x.source for x in samples], is_audio_input=True)
        n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)
        speech_lens = self.length_shrink_func(n_frames)

        texts = [x.target for x in samples]
     
        # Create speech tokens based on length
        speech_tokens = [speech_lens.max() * DEFAULT_SPEECH_PATCH_TOKEN for _ in speech_lens]
        speech_tokens = [DEFAULT_SPEECH_START_TOKEN + tokens + DEFAULT_SPEECH_END_TOKEN for tokens in speech_tokens]

        text_tokens = [DEFAULT_SPEECH_START_TOKEN + x.src_text + DEFAULT_SPEECH_END_TOKEN for x in samples]

        # Create prompts
        instruction = f"Translate the following speech from {self.source_lang} to {self.target_lang}:"
        prompts = [f"{instruction} {speech_token} {text}<|end_of_text|>" for speech_token, text in zip(speech_tokens, texts)]
        text_prompts = [f"{instruction} {text_token} {text}<|end_of_text|>" for text_token, text in zip(text_tokens, texts)]
        
        # Get instruction length for masking
        instruction_ids = self.tokenizer(instruction + " ", add_special_tokens=False).input_ids
        instruction_len = len(instruction_ids)

        # Tokenize with explicit padding settings
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        # Create targets and handle padding properly
        targets = input_ids.clone()
        for i in range(len(samples)):
            # 1. Mask instruction tokens
            targets[i, :instruction_len] = IGNORE_INDEX
            
            # 2. Mask speech tokens
            start_pos = (input_ids[i] == self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_START_TOKEN)).nonzero()[0][0]
            end_pos = (input_ids[i] == self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_END_TOKEN)).nonzero()[0][0]
            targets[i, start_pos : end_pos + 1] = IGNORE_INDEX
            
            # 3. Mask padding tokens
            targets[i, attention_mask[i] == 0] = IGNORE_INDEX

        # Tokenize with explicit padding settings
        text_tokenized = self.tokenizer(
            text_prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        text_input_ids = text_tokenized.input_ids
        text_attention_mask = text_tokenized.attention_mask

        # Create targets and handle padding properly
        text_targets = text_input_ids.clone()
        for i in range(len(samples)):
            # 1. Mask instruction tokens
            text_targets[i, :instruction_len] = IGNORE_INDEX
            # 2. Mask speech tokens
            start_pos = (text_input_ids[i] == self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_START_TOKEN)).nonzero()[0][0]
            end_pos = (text_input_ids[i] == self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_END_TOKEN)).nonzero()[0][0]
            text_targets[i, start_pos : end_pos + 1] = IGNORE_INDEX
            # 3. Mask padding tokens
            text_targets[i, text_attention_mask[i] == 0] = IGNORE_INDEX
                
        batch = dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=attention_mask,
            speech_batch=speech_batch,
            src_lengths=n_frames,
            after_lens=speech_lens,
            target_text=texts,
            ids=indices,

            text_input_ids=text_input_ids,
            text_labels=text_targets,
            text_attention_mask=text_attention_mask,
        )

        return batch

class DataCollatorForSupervisedInstructDataset(DataCollatorForSupervisedDataset):
    def __call__(self, samples: List[SpeechToTextDatasetItem]) -> Dict[str, torch.Tensor]:
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        speech_batch = _collate_frames([x.source for x in samples], is_audio_input=True)
        n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)
        speech_lens = self.length_shrink_func(n_frames)

        texts = [x.target for x in samples]

        prompts = []
        instruction = f"Translate the following speech from {self.source_lang} to {self.target_lang}."
        for i, x in enumerate(samples):
            messages = [{
                "role": "system",
                "content": instruction
            }]
            messages.append(
                {
                    "role": "user",
                    "content": speech_lens.max() * DEFAULT_SPEECH_PATCH_TOKEN
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": x.target
                }
            )
            prompts.append(messages)
     
        # Tokenize with explicit padding settings
        tokenized = self.tokenizer.apply_chat_template(
            prompts,
            return_tensors='pt',
            padding=True, 
            truncation=False, 
            add_special_tokens=False
        )
        input_ids = tokenized
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Create targets and handle padding properly
        targets = input_ids.clone()
        targets[attention_mask == 0] = IGNORE_INDEX
        user_id = self.tokenizer.convert_tokens_to_ids('user')
        assist_id = self.tokenizer.convert_tokens_to_ids('assistant')
        start_header_id = self.tokenizer.convert_tokens_to_ids('<|start_header_id|>')
        label_mask = torch.zeros_like(targets, dtype=torch.bool)
        for i in range(len(samples)):
            user_pos = (targets[i] == user_id).nonzero()
            assist_pos = (targets[i] == assist_id).nonzero()

            user_pos = [
                pos for pos in user_pos if targets[i, pos[0] - 1] == start_header_id
            ]
            assist_pos = [
                pos for pos in assist_pos if targets[i, pos[0] - 1] == start_header_id
            ]

            assert len(user_pos) == 1 and len(assist_pos) == 1

            label_mask[i, assist_pos[0][0] + 2:] = True
        targets[~label_mask] = IGNORE_INDEX
                
        batch = dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=attention_mask,
            speech_batch=speech_batch,
            src_lengths=n_frames,
            after_lens=speech_lens,
            target_text=texts,
            ids=indices,
        )

        return batch
    

class DataCollatorForOfflineQwen2ACDataset:
    def __init__(self, processor, source_lang, target_lang, **kwargs):
        self.processor = processor
        self.source_lang = source_lang
        self.target_lang = target_lang

    def __call__(self, samples: List[SpeechToTextDatasetItem]) -> Dict[str, torch.Tensor]:
        audios = [x.source.numpy() for x in samples]
        n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)

        prompts = []
        instruction = f"Translate the following speech from {self.source_lang} to {self.target_lang}."
        for i, x in enumerate(samples):
            messages = [{
                "role": "system",
                "content": instruction
            }]
            messages.append(
                {
                    "role": "user",
                    "content": "<|audio_bos|><|AUDIO|><|audio_eos|>"
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": x.target
                }
            )
            prompts.append(messages)
     
        # Tokenize with explicit padding settings
        texts = self.processor.apply_chat_template(
            prompts,
            return_tensors='pt',
            padding=True, 
            truncation=False, 
            add_special_tokens=False
        )

        inputs = self.processor(
            text=texts, 
            audios=audios, 
            sampling_rate=self.processor.feature_extractor.sampling_rate, 
            return_tensors="pt", 
            padding="longest",
            max_length=n_frames.max(),
        )

        input_ids = inputs["input_ids"]
        attention_mask = (input_ids != self.processor.tokenizer.pad_token_id).long()

        # Create targets and handle padding properly
        targets = input_ids.clone()
        targets[attention_mask == 0] = IGNORE_INDEX
        user_id = self.processor.tokenizer.convert_tokens_to_ids('user')
        assist_id = self.processor.tokenizer.convert_tokens_to_ids('assistant')
        start_header_id = self.processor.tokenizer.convert_tokens_to_ids('<|im_start|>')
        label_mask = torch.zeros_like(targets, dtype=torch.bool)
        for i in range(len(samples)):
            user_pos = (targets[i] == user_id).nonzero()
            assist_pos = (targets[i] == assist_id).nonzero()

            user_pos = [
                pos for pos in user_pos if targets[i, pos[0] - 1] == start_header_id
            ]
            assist_pos = [
                pos for pos in assist_pos if targets[i, pos[0] - 1] == start_header_id
            ]

            assert len(user_pos) == 1 and len(assist_pos) == 1

            label_mask[i, assist_pos[0][0] + 2:] = True
        targets[~label_mask] = IGNORE_INDEX        
                
        inputs["labels"] = targets
        inputs["src_lengths"] = n_frames

        return inputs


@dataclass
class DataCollatorForTrajectoryDataset(object):
    def __init__(self, 
            tokenizer, length_shrink_func, source_lang, target_lang, 
            block_size=48, **kwargs
        ):
        self.tokenizer = tokenizer
        self.length_shrink_func = length_shrink_func
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.speech_segment_size = block_size // 4    

    def validate(self, dataset):
        if dataset.trajectories is not None:
            sp_seg_frame = int(12 * 0.08 * 16000)
            for i in range(len(dataset.audio_paths)):
                if dataset.trajectories[i] is not None:
                    n_frame = dataset.n_frames[i]
                    if n_frame % sp_seg_frame != 0:
                        n_pad = sp_seg_frame - n_frame % sp_seg_frame
                        n_frame += n_pad
                    n_frame += 79 + 320
                    speech_len = self.length_shrink_func(torch.tensor(n_frame))
                    trajectory_len = len(dataset.trajectories[i])

                    assert trajectory_len == speech_len // self.speech_segment_size
     
    def __call__(self, samples: List[SpeechToTextDatasetItem]) -> Dict[str, torch.Tensor]:
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)

        # pad to multiple
        sp_seg_frame = int(self.speech_segment_size * 0.08 * 16000)
        for x in samples:
            if x.source.shape[0] % sp_seg_frame != 0:
                n_pad = sp_seg_frame - x.source.shape[0] % sp_seg_frame
                x.source = torch.cat([x.source, torch.zeros(n_pad).to(x.source)], dim=0)

        speech_batch = _collate_frames([x.source for x in samples], is_audio_input=True)
        offset = torch.zeros(len(samples), 79 + 320).to(speech_batch)
        speech_batch = torch.cat([offset, speech_batch], dim=1)

        n_frames = torch.tensor([x.source.size(0) + 79 + 320 for x in samples], dtype=torch.long)        
        speech_lens = self.length_shrink_func(n_frames)

        trajectory_lens = [len(x.trajectory) for x in samples]
        assert all([t_l == s_l // self.speech_segment_size for t_l, s_l in zip(trajectory_lens, speech_lens)])

        prompts = []
        instruction = f"Translate the following speech from {self.source_lang} to {self.target_lang}: "
        for i, x in enumerate(samples):
            prompt = instruction
            for j, text in enumerate(x.trajectory):
                n_sp_token = min(
                    self.speech_segment_size, 
                    speech_lens[i] - j * self.speech_segment_size
                )
                assert n_sp_token > 0

                sp_tokens = DEFAULT_SPEECH_START_TOKEN + \
                    n_sp_token * DEFAULT_SPEECH_PATCH_TOKEN + \
                    DEFAULT_SPEECH_END_TOKEN

                prompt += sp_tokens + text + "<|end_of_text|>"
            prompts.append(prompt)
     
        # Get instruction length for masking
        instruction_ids = self.tokenizer(instruction, add_special_tokens=False).input_ids
        instruction_len = len(instruction_ids)

        # Tokenize with explicit padding settings
        tokenized = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = tokenized.input_ids
        attention_mask = tokenized.attention_mask

        # Create targets and handle padding properly
        targets = input_ids.clone()
        sp_start_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_START_TOKEN)
        sp_end_id = self.tokenizer.convert_tokens_to_ids(DEFAULT_SPEECH_END_TOKEN)
        for i in range(len(samples)):
            # 1. Mask instruction tokens
            targets[i, :instruction_len] = IGNORE_INDEX
            
            # 2. Mask speech tokens
            start_positions = (input_ids[i] == sp_start_id).nonzero()
            end_positions = (input_ids[i] == sp_end_id).nonzero()
            for start, end in zip(start_positions, end_positions):
                targets[i, start[0] : end[0] + 1] = IGNORE_INDEX
            
            # 3. Mask padding tokens
            targets[i, attention_mask[i] == 0] = IGNORE_INDEX

        batch = dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=attention_mask,
            speech_batch=speech_batch,
            src_lengths=n_frames,
            after_lens=speech_lens,
            ids=indices,
        )

        return batch

class DataCollatorForTrajectoryInstructDataset(DataCollatorForTrajectoryDataset):
    def __init__(self, 
            tokenizer, length_shrink_func, source_lang, target_lang, 
            block_size=48, perturb=(0.3, 0.3, 0.4), **kwargs
        ):
        super().__init__(tokenizer, length_shrink_func, source_lang, target_lang, block_size, **kwargs)
        assert sum(perturb) == 1
        self.perturb = perturb

    def validate(self, dataset):
        if dataset.trajectories is not None:
            sp_seg_frame = int(12 * 0.08 * 16000)
            instruction = f"Translate the following speech from {self.source_lang} to {self.target_lang}."
            for i in tqdm(range(len(dataset.audio_paths))):
                if dataset.trajectories[i] is not None:
                    n_frame = dataset.n_frames[i]
                    if n_frame % sp_seg_frame != 0:
                        n_pad = sp_seg_frame - n_frame % sp_seg_frame
                        n_frame += n_pad
                    n_frame += 79 + 320
                    speech_len = self.length_shrink_func(torch.tensor(n_frame))
                    trajectory_len = len(dataset.trajectories[i])

                    assert trajectory_len == speech_len // self.speech_segment_size

                    messages = [{
                        "role": "system",
                        "content": instruction
                    }]
                    for j, text in enumerate(dataset.trajectories[i]):
                        n_sp_token = min(
                            self.speech_segment_size, 
                            speech_len - j * self.speech_segment_size
                        )
                        assert n_sp_token > 0

                        messages.append(
                            {
                                "role": "user",
                                "content": n_sp_token * DEFAULT_SPEECH_PATCH_TOKEN
                            }
                        )
                        messages.append(
                            {
                                "role": "assistant",
                                "content": text
                            }
                        )

                    tokenized = self.tokenizer.apply_chat_template(
                        [messages],
                        return_tensors='pt',
                        padding=True, 
                        truncation=False, 
                        add_special_tokens=False
                    )
                    attention_mask = (tokenized != self.tokenizer.pad_token_id).long()
                    targets = tokenized.clone()
                    targets[attention_mask == 0] = IGNORE_INDEX

                    user_id = self.tokenizer.convert_tokens_to_ids('user')
                    assist_id = self.tokenizer.convert_tokens_to_ids('assistant')
                    start_header_id = self.tokenizer.convert_tokens_to_ids('<|start_header_id|>')

                    user_pos = (targets[0] == user_id).nonzero()
                    assist_pos = (targets[0] == assist_id).nonzero()

                    # print(user_pos, targets)

                    user_pos = [
                        pos for pos in user_pos if targets[0, pos[0] - 1] == start_header_id
                    ]
                    assist_pos = [
                        pos for pos in assist_pos if targets[0, pos[0] - 1] == start_header_id
                    ]

                    assert len(user_pos) == len(assist_pos), (user_id, assist_id, targets)

    def __call__(self, samples: List[SpeechToTextDatasetItem]) -> Dict[str, torch.Tensor]:
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)

        # pad to multiple
        sp_seg_frame = int(self.speech_segment_size * 0.08 * 16000)
        for x in samples:
            if x.source.shape[0] % sp_seg_frame != 0:
                n_pad = sp_seg_frame - x.source.shape[0] % sp_seg_frame
                x.source = torch.cat([x.source, torch.zeros(n_pad).to(x.source)], dim=0)

        speech_batch = _collate_frames([x.source for x in samples], is_audio_input=True)
        offset = torch.zeros(len(samples), 79 + 320).to(speech_batch)
        speech_batch = torch.cat([offset, speech_batch], dim=1)

        n_frames = torch.tensor([x.source.size(0) + 79 + 320 for x in samples], dtype=torch.long)        
        speech_lens = self.length_shrink_func(n_frames)

        for x in samples:
            try:
                if type(x.trajectory[0]) == str:
                    x.trajectory = [[seg, True] for seg in x.trajectory]
            except:
                print(x)
                raise KeyError

        mode = np.random.choice(['opt', 'aug', 'off'], p=self.perturb)
        for x in samples:
            if mode == 'opt':
                # with prob self.perturb[0], use the optimal trajectory
                continue
            elif mode == 'aug':
                # with prob self.perturb[1], use the delayed trajectory
                traj = x.trajectory

                # shift
                shift_traj = []
                for i in range(len(traj)):
                    seg = traj[len(traj) - i - 1][0]
                    if seg == "" or np.random.rand() < 0.5 or i == 0:
                        shift_traj.append([seg, True])
                        continue
                    words = list(jieba.cut(seg))
                    shift_idx = np.random.randint(len(words))
                    shift_traj[-1][0] = ''.join(words[shift_idx:]) + shift_traj[-1][0]
                    shift_traj.append([''.join(words[:shift_idx]), False])

                shift_traj = shift_traj[::-1]

                # merge
                merge_traj = copy.deepcopy(shift_traj)
                for i in range(len(merge_traj) - 1):
                    seg, _ = merge_traj[i]
                    if seg == "" or np.random.rand() < 0.5:
                        continue
                    
                    merge_traj[i] = ["", False]
                    merge_traj[i + 1][0] = seg + merge_traj[i + 1][0]
                
                x.trajectory = merge_traj
            else:
                # with prob self.perturb[2], use the offline trajectory
                x.trajectory = [['', False]] * len(x.trajectory)
                x.trajectory[-1] = [x.target, True]


        trajectory_lens = [len(x.trajectory) for x in samples]
        assert all([t_l == s_l // self.speech_segment_size for t_l, s_l in zip(trajectory_lens, speech_lens)])

        prompts = []
        instruction = f"Translate the following speech from {self.source_lang} to {self.target_lang}."
        for i, x in enumerate(samples):
            messages = [{
                "role": "system",
                "content": instruction
            }]
            for j, (text, _) in enumerate(x.trajectory):
                n_sp_token = min(
                    self.speech_segment_size, 
                    speech_lens[i] - j * self.speech_segment_size
                )
                assert n_sp_token > 0

                messages.append(
                    {
                        "role": "user",
                        "content": n_sp_token * DEFAULT_SPEECH_PATCH_TOKEN
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": text
                    }
                )
            prompts.append(messages)
     
        # Tokenize with explicit padding settings
        tokenized = self.tokenizer.apply_chat_template(
            prompts,
            return_tensors='pt',
            padding=True, 
            truncation=False, 
            add_special_tokens=False
        )
        input_ids = tokenized
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Create targets and handle padding properly
        targets = input_ids.clone()
        targets[attention_mask == 0] = IGNORE_INDEX
        user_id = self.tokenizer.convert_tokens_to_ids('user')
        assist_id = self.tokenizer.convert_tokens_to_ids('assistant')
        start_header_id = self.tokenizer.convert_tokens_to_ids('<|start_header_id|>')
        label_mask = torch.zeros_like(targets, dtype=torch.bool)
        for i in range(len(samples)):
            user_pos = (targets[i] == user_id).nonzero()
            assist_pos = (targets[i] == assist_id).nonzero()

            user_pos = [
                pos for pos in user_pos if targets[i, pos[0] - 1] == start_header_id
            ]
            assist_pos = [
                pos for pos in assist_pos if targets[i, pos[0] - 1] == start_header_id
            ]

            assert len(user_pos) == len(assist_pos)

            for j in range(len(user_pos) - 1):
                if samples[i].trajectory[j][1]:
                    label_mask[i, assist_pos[j][0] + 2 : user_pos[j + 1][0] - 1] = True
            label_mask[i, assist_pos[-1][0] + 2:] = True
        targets[~label_mask] = IGNORE_INDEX

        batch = dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=attention_mask,
            speech_batch=speech_batch,
            src_lengths=n_frames,
            after_lens=speech_lens,
            ids=indices,
            mode=mode,
        )

        return batch

class DataCollatorForTrajectoryInstructMultiLatencyDataset(DataCollatorForTrajectoryDataset):
    def __init__(self, 
            tokenizer, length_shrink_func, source_lang, target_lang, 
            block_size=48, max_multiplier=1, prob_aug=0., trainer=None, **kwargs
        ):
        super().__init__(tokenizer, length_shrink_func, source_lang, target_lang, block_size, **kwargs)
        assert max_multiplier >= 1 and prob_aug >= 0 and prob_aug <= 1
        self.max_multiplier = max_multiplier
        self.prob_aug = prob_aug
        self.trainer = trainer

    def __call__(self, samples: List[SpeechToTextDatasetItem]) -> Dict[str, torch.Tensor]:
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)

        multiplier = np.random.randint(1, self.max_multiplier + 1)
        latency_token = DEFAULT_LATENCY_TOKEN.format(multiplier)

        # pad to multiple
        sp_seg_frame = int(self.speech_segment_size * 0.08 * 16000) * multiplier
        for x in samples:
            if x.source.shape[0] % sp_seg_frame != 0:
                n_pad = sp_seg_frame - x.source.shape[0] % sp_seg_frame
                x.source = torch.cat([x.source, torch.zeros(n_pad).to(x.source)], dim=0)

        speech_batch = _collate_frames([x.source for x in samples], is_audio_input=True)
        offset = torch.zeros(len(samples), 79 + 320).to(speech_batch)
        speech_batch = torch.cat([offset, speech_batch], dim=1)

        n_frames = torch.tensor([x.source.size(0) + 79 + 320 for x in samples], dtype=torch.long)        
        speech_lens = self.length_shrink_func(n_frames)

        # trajectory_lens = [len(x.trajectory) for x in samples]
        # assert all([t_l == s_l // self.speech_segment_size for t_l, s_l in zip(trajectory_lens, speech_lens)])

        for x in samples:
            if type(x.trajectory[0]) == str:
                x.trajectory = [[seg, True] for seg in x.trajectory]
        
        for x in samples:
            traj = x.trajectory
            new_traj = []
            for i in range(0, len(traj), multiplier):
                partial_translation = ''.join(
                    traj[j][0] for j in range(i, min(i + multiplier, len(traj)))
                )
                new_traj.append([partial_translation, True])
            x.trajectory = new_traj

        if np.random.rand() < self.prob_aug: # only zh
            for x in samples:
                traj = x.trajectory

                # shift
                shift_traj = []
                for i in range(len(traj)):
                    seg = traj[len(traj) - i - 1][0]
                    if seg == "" or np.random.rand() < 0.5 or i == 0:
                        shift_traj.append([seg, True])
                        continue
                    words = list(jieba.cut(seg))
                    shift_idx = np.random.randint(len(words))
                    shift_traj[-1][0] = ''.join(words[shift_idx:]) + shift_traj[-1][0]
                    shift_traj.append([''.join(words[:shift_idx]), False])

                shift_traj = shift_traj[::-1]

                # merge
                merge_traj = copy.deepcopy(shift_traj)
                for i in range(len(merge_traj) - 1):
                    seg, _ = merge_traj[i]
                    if seg == "" or np.random.rand() < 0.5:
                        continue
                    
                    merge_traj[i] = ["", False]
                    merge_traj[i + 1][0] = seg + merge_traj[i + 1][0]
                
                x.trajectory = merge_traj

        prompts = []
        instruction = f"Translate the following speech from {self.source_lang} to {self.target_lang} with latency {latency_token}."
        for i, x in enumerate(samples):
            messages = [{
                "role": "system",
                "content": instruction
            }]
            for j, (text, _) in enumerate(x.trajectory):
                n_sp_token = min(
                    self.speech_segment_size * multiplier, 
                    speech_lens[i] - j * self.speech_segment_size * multiplier
                )
                assert n_sp_token > 0

                messages.append(
                    {
                        "role": "user",
                        "content": n_sp_token * DEFAULT_SPEECH_PATCH_TOKEN
                    }
                )
                messages.append(
                    {
                        "role": "assistant",
                        "content": text
                    }
                )
            prompts.append(messages)
     
        # Tokenize with explicit padding settings
        tokenized = self.tokenizer.apply_chat_template(
            prompts,
            return_tensors='pt',
            padding=True, 
            truncation=False, 
            add_special_tokens=False
        )
        input_ids = tokenized
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        # Create targets and handle padding properly
        targets = input_ids.clone()
        targets[attention_mask == 0] = IGNORE_INDEX
        user_id = self.tokenizer.convert_tokens_to_ids('user')
        assist_id = self.tokenizer.convert_tokens_to_ids('assistant')
        start_header_id = self.tokenizer.convert_tokens_to_ids('<|start_header_id|>')
        label_mask = torch.zeros_like(targets, dtype=torch.bool)
        for i in range(len(samples)):
            user_pos = (targets[i] == user_id).nonzero()
            assist_pos = (targets[i] == assist_id).nonzero()

            user_pos = [
                pos for pos in user_pos if targets[i, pos[0] - 1] == start_header_id
            ]
            assist_pos = [
                pos for pos in assist_pos if targets[i, pos[0] - 1] == start_header_id
            ]

            assert len(user_pos) == len(assist_pos)

            for j in range(len(user_pos) - 1):
                if samples[i].trajectory[j][1]:
                    label_mask[i, assist_pos[j][0] + 2 : user_pos[j + 1][0] - 1] = True
            label_mask[i, assist_pos[-1][0] + 2:] = True
        targets[~label_mask] = IGNORE_INDEX

        batch = dict(
            input_ids=input_ids,
            labels=targets,
            attention_mask=attention_mask,
            speech_batch=speech_batch,
            src_lengths=n_frames,
            after_lens=speech_lens,
            ids=indices,
            multiplier=multiplier
        )

        return batch
        