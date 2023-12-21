# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import csv
import io
import logging
import re
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import torch
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
from fairseq.data.audio.speech_to_text_dataset import SpeechToTextDataset


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
    
class PromptSpeechToTextDataset(SpeechToTextDataset):

    def __init__(
        self,
        audio_paths: List[str],
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
    ):
        self.audio_paths = audio_paths
        self.tgt_texts = tgt_texts
        self.src_texts = src_texts
        self.ids = ids
        self.tasks = tasks

    def __getitem__(
        self, index: int
    ) -> Tuple[int, torch.Tensor, Optional[torch.Tensor]]:
        source, sr = get_features_or_waveform(
            self.audio_paths[index],
        )
        source = torch.from_numpy(source).float()
        with torch.no_grad():
            source = F.layer_norm(source, source.shape)       
        text = self.tgt_texts[index]
        id = self.ids[index]
        task = self.tasks[index]
        src_text = self.src_texts[index]
        
        return SpeechToTextDatasetItem(
            index=index, source=source, target=text, src_text=src_text, id=id, task=task
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
    # default values
    DEFAULT_SPEAKER = DEFAULT_SRC_TEXT = DEFAULT_LANG = DEFAULT_LANG_N_FRAMES = DEFAULT_TASK = ""
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
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        tasks = [s.get(cls.TASK, cls.DEFAULT_TASK) for s in samples] 

        return PromptSpeechToTextDataset(
            audio_paths,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            ids=ids,
            tasks=tasks,
        )
