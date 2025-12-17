import os
import re
import json
import pickle
import contextlib
from time import perf_counter

from typing import Optional, List, Dict
from simuleval.agents.states import AgentStates
from simuleval.utils import entrypoint
from simuleval.data.segments import SpeechSegment
from simuleval.agents import SpeechToTextAgent
from simuleval.agents.actions import WriteAction, ReadAction
from simuleval.agents.states import AgentStates
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoProcessor

from peft import LoraConfig, get_peft_model

from tqdm import tqdm
from model.llama31 import SpeechLlamaForCausalLM
from model.qwen25 import SpeechQwenForCausalLM
from model.patches.patch_w2v2 import patch_w2v2
from model.patches.patch_llama31 import patch_llama31
from model.patches.patch_qwen25 import patch_qwen25
from model.patches.patch_hf import patch_hf

from agents.options import (
    add_speech_encoder_args,
    add_simuleval_args,
    add_gen_args
)
from model.w2v2 import SpeechEncoderW2V2RoPE
from model.seamlessm4t_v2_encoder import (
    SeamlessM4Tv2Config,
    SeamlessM4Tv2SpeechEncoder
)
from train.dataset import (
    DEFAULT_SPEECH_PATCH_TOKEN,
    DEFAULT_LATENCY_TOKEN,
    normalize
)

import logging

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None
logger = logging.getLogger(__name__)

def synchronized_timer(description: str):
    @contextlib.contextmanager
    def timer_with_sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_time = perf_counter() - start
        print(f"{description}: {elapsed_time:.4f} seconds")
    return timer_with_sync()

@dataclass
class S2TAgentStates(AgentStates):
    src_len: int
    speech_cache: None
    past_key_values: None
    target_ids: list
    segment_idx: int
    translations_list: list
    references: List[Dict[str, str]]
    MAX_SRC_LEN = 16000 * 30

    def reset(self):
        super().reset()
        self.src_len = 0
        self.speech_cache = None
        self.past_key_values = None
        self.target_ids = []
        self.segment_idx = 0
        self.translations_list = []
        self.references = []


class ProcessorAudioAlias:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, *args, **kwargs):
        if "audio" in kwargs and "audios" not in kwargs:
            kwargs["audios"] = kwargs.pop("audio")
        return self.processor(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.processor, item)


class TokenizerKwCleaner:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        kwargs.pop("audio", None)
        kwargs.pop("audios", None)
        return self.tokenizer(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self.tokenizer, item)


class TermRAGRetriever:
    def __init__(
        self,
        index_path: Optional[str],
        model_path: Optional[str],
        base_model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
        device: str = "cuda:0",
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        top_k: int = 5,
        target_lang: str = "zh",
    ):
        self.enabled = False
        self.index = None
        self.term_list: List[Dict[str, object]] = []
        self.embedding_dim = 512
        self.device_str = device
        if device and device.startswith("cuda") and torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")
            if device and device.startswith("cuda"):
                logger.warning("CUDA unavailable, falling back to CPU for RAG retriever")
        self.top_k = top_k
        self.target_lang = target_lang.lower() if target_lang else "zh"
        self.model = None
        self.speech_encoder = None

        if faiss is None:
            logger.warning("FAISS is not available; disabling RAG retriever")
            return
        if not index_path or not os.path.exists(index_path):
            logger.warning("RAG index path is missing; disabling RAG retriever")
            return
        try:
            self._load_index(index_path)
        except Exception as exc:
            logger.exception("Failed to load RAG index from %s: %s", index_path, exc)
            return

        if not model_path or not os.path.exists(model_path):
            logger.warning("RAG model checkpoint not found at %s; disabling RAG retriever", model_path)
            return

        try:
            self._load_model(
                model_path=model_path,
                base_model_name=base_model_name,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
        except Exception as exc:
            logger.exception("Failed to load RAG model: %s", exc)
            self.index = None
            self.term_list = []
            return

        self.enabled = self.index is not None and self.model is not None
        if self.enabled:
            logger.info(
                "TermRAGRetriever initialized with %d terms (embedding_dim=%d, top_k=%d, target_lang=%s)",
                len(self.term_list),
                self.embedding_dim,
                self.top_k,
                self.target_lang,
            )

    def _load_index(self, index_path: str):
        with open(index_path, "rb") as f:
            data = pickle.load(f)
        serialized_index = data.get("faiss_index")
        if isinstance(serialized_index, bytes):
            serialized_index = np.frombuffer(serialized_index, dtype=np.uint8)
        elif isinstance(serialized_index, bytearray):
            serialized_index = np.frombuffer(bytes(serialized_index), dtype=np.uint8)
        elif isinstance(serialized_index, np.ndarray):
            if serialized_index.dtype != np.uint8:
                serialized_index = serialized_index.astype(np.uint8)
        else:
            raise ValueError(f"Unsupported faiss_index type: {type(serialized_index)}")
        self.index = faiss.deserialize_index(serialized_index)
        self.term_list = data.get("term_list", [])
        self.embedding_dim = int(data.get("embedding_dim", 512))

    def _load_model(
        self,
        model_path: str,
        base_model_name: str,
        lora_r: int,
        lora_alpha: int,
        lora_dropout: float,
    ):
        from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
        from peft import LoraConfig, get_peft_model, TaskType
        from retriever.gigaspeech.modal.Qwen2_Audio_train import Qwen2AudioSpeechEncoder

        processor = AutoProcessor.from_pretrained(base_model_name)
        processor.tokenizer = TokenizerKwCleaner(processor.tokenizer)
        processor = ProcessorAudioAlias(processor)
        base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=torch.float16,
        ).to(self.device)
        base_model.eval()

        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        base_model = get_peft_model(base_model, lora_config)
        base_model.eval()

        speech_encoder = Qwen2AudioSpeechEncoder.__new__(Qwen2AudioSpeechEncoder)
        speech_encoder.device = self.device
        speech_encoder.model_name = base_model_name
        speech_encoder.processor = processor
        speech_encoder.model = base_model
        speech_encoder._analyze_model_structure()

        speech_hidden = speech_encoder.get_hidden_size()

        class SimpleContrastiveModel(nn.Module):
            def __init__(self, speech_encoder, speech_hidden_dim, proj_dim, device):
                super().__init__()
                self.speech_encoder = speech_encoder
                self.proj_speech = nn.Linear(speech_hidden_dim, proj_dim).to(device)

            def encode_audio(self, audio_inputs):
                with torch.no_grad():
                    emb = self.speech_encoder.predict(audio_inputs)
                if not isinstance(emb, torch.Tensor):
                    emb = torch.as_tensor(emb)
                emb = emb.float().to(self.proj_speech.weight.device)
                if emb.dim() == 3:
                    emb = emb.mean(dim=1)
                return F.normalize(self.proj_speech(emb), dim=-1)

        model = SimpleContrastiveModel(
            speech_encoder,
            speech_hidden_dim=speech_hidden,
            proj_dim=self.embedding_dim,
            device=self.device,
        )

        checkpoint = torch.load(model_path, map_location=self.device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
        if state_dict:
            first_key = next(iter(state_dict))
            if first_key.startswith("module."):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

        proj_state = {}
        lora_state = {}
        for key, value in state_dict.items():
            if key.startswith("proj_speech"):
                proj_state[key] = value
            elif key.startswith("proj_text"):
                continue
            elif "lora_" in key or "base_model" in key:
                if key.startswith("speech_qwen2_model.") or key.startswith("text_qwen2_model."):
                    new_key = key.split(".", 1)[1] if "." in key else key
                    lora_state[new_key] = value
                else:
                    lora_state[key] = value

        if proj_state:
            filtered_proj = {k: v for k, v in proj_state.items() if k.startswith("proj_speech")}
            missing, unexpected = model.load_state_dict(filtered_proj, strict=False)
            if missing:
                logger.debug("Missing projection keys for RAG model: %s", missing)
            if unexpected:
                logger.debug("Unexpected projection keys for RAG model: %s", unexpected)
        if lora_state:
            missing_keys, unexpected_keys = base_model.load_state_dict(lora_state, strict=False)
            if missing_keys:
                logger.debug("Missing LoRA keys during RAG load: %s", missing_keys[:10])
            if unexpected_keys:
                logger.debug("Unexpected LoRA keys during RAG load: %s", unexpected_keys[:10])

        self.model = model.eval()
        self.speech_encoder = speech_encoder

    def retrieve(
        self,
        audio_tensor: torch.Tensor,
        top_k: Optional[int] = None,
        target_lang: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        if not self.enabled or self.index is None or self.model is None:
            return []
        if not isinstance(audio_tensor, torch.Tensor):
            audio_tensor = torch.tensor(audio_tensor)
        audio_tensor = audio_tensor.detach().cpu().float()
        if audio_tensor.numel() == 0:
            return []
        if audio_tensor.abs().sum().item() == 0.0:
            return []

        audio_np = audio_tensor.numpy()
        audio_inputs = [audio_np]
        with torch.no_grad():
            embedding = self.model.encode_audio(audio_inputs)
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().float().numpy()

        max_k = top_k or self.top_k
        max_k = max(1, max_k)
        D, I = self.index.search(embedding, max_k)

        target_lang = (target_lang or self.target_lang or "zh").lower()
        results: List[Dict[str, str]] = []
        seen_terms = set()

        for idx in I[0]:
            if idx < 0 or idx >= len(self.term_list):
                continue
            term_entry = self.term_list[idx]
            if not isinstance(term_entry, dict):
                continue
            term = term_entry.get("term", "")
            if not term or term in seen_terms:
                continue
            seen_terms.add(term)
            translation = ""
            translations = term_entry.get("target_translations") or {}
            if isinstance(translations, dict):
                translation = translations.get(target_lang) or translations.get(target_lang.upper()) or ""
            results.append({"term": term, "translation": translation})
            if len(results) >= max_k:
                break
        return results

@entrypoint
class InfiniSST(SpeechToTextAgent):

    def __init__(self, args):
        super().__init__(args)
        transformers.set_seed(998244353)

        # simuleval
        self.min_start_sec = args.min_start_sec
        self.latency_multiplier = args.latency_multiplier
        self.source_segment_size = getattr(args, 'source_segment_size', 960 * args.latency_multiplier)
        self.max_latency_multiplier = args.max_latency_multiplier
        self.source_lang = args.source_lang
        self.target_lang = args.target_lang
        
        # gen
        self.beam = args.beam
        # assert self.beam > 1
        self.no_repeat_ngram_lookback = args.no_repeat_ngram_lookback
        self.no_repeat_ngram_size = args.no_repeat_ngram_size
        self.repetition_penalty = args.repetition_penalty
        self.suppress_non_language = args.suppress_non_language
        self.max_len_a = args.max_len_a
        self.max_len_b = args.max_len_b
        self.max_new_tokens = args.max_new_tokens
        self.do_sample = args.do_sample
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.epsilon_cutoff = args.epsilon_cutoff
        self.temperature = args.temperature
        self.pseudo_batch_size = args.pseudo_batch_size

        logger.info(f"max_new_tokens: {self.max_new_tokens}")

        # cache
        self.max_llm_cache_size = args.max_llm_cache_size
        self.always_cache_system_prompt = args.always_cache_system_prompt
        self.cache_checkpoints = []
        
        # Add DPO sampling flag
        self.dpo_sampling = args.dpo_sampling
        self.output_file = args.output_file if hasattr(args, 'output_file') else 'translations.json'

        self.audio_normalize = args.audio_normalize
        
        # model
        self.load_model(args)
        self.rag_retriever = None
        self.rag_top_k = getattr(args, "rag_top_k", 10)
        self.rag_target_lang = getattr(args, "rag_target_lang", "zh")
        if getattr(args, "rag_enabled", False):
            self.rag_retriever = TermRAGRetriever(
                index_path=getattr(args, "rag_index_path", None),
                model_path=getattr(args, "rag_model_path", None),
                base_model_name=getattr(args, "rag_base_model", "Qwen/Qwen2-Audio-7B-Instruct"),
                device=getattr(args, "rag_device", "cuda:0"),
                lora_r=getattr(args, "rag_lora_r", 16),
                lora_alpha=getattr(args, "rag_lora_alpha", 32),
                lora_dropout=getattr(args, "rag_lora_dropout", 0.0),
                top_k=self.rag_top_k,
                target_lang=self.rag_target_lang,
            )
            if not self.rag_retriever or not self.rag_retriever.enabled:
                logger.warning("RAG retriever not operational; continuing without references")
                self.rag_retriever = None
        else:
            logger.info("RAG retrieval disabled for InfiniSST agent")
    
    @staticmethod
    def add_args(parser):
        add_simuleval_args(parser)
        add_speech_encoder_args(parser)
        add_gen_args(parser)
        parser.add_argument("--model-type", type=str, default="w2v2_llama31")
        parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
        parser.add_argument("--state-dict-path", type=str, default=None)
        parser.add_argument("--lora-path", type=str, default=None)
        parser.add_argument("--lora-rank", type=int, default=8)
        parser.add_argument("--latency-multiplier", type=int, default=4)
        parser.add_argument("--max-latency-multiplier", type=int, default=4)
        parser.add_argument("--max-llm-cache-size", type=int, default=10000)
        parser.add_argument("--always-cache-system-prompt", action='store_true') # LLM-Inf
        parser.add_argument("--dpo-sampling", action='store_true', help="Enable storing sampling for DPO")
        parser.add_argument("--output-file", type=str, default="translations.json", help="Output file for sampling")
        parser.add_argument("--pseudo-batch-size", type=int, default=1)
        parser.add_argument("--audio-normalize", type=int, default=0)
        parser.add_argument("--rag-enabled", action='store_true', help="Enable glossary RAG retrieval for prompt augmentation")
        parser.add_argument("--rag-index-path", type=str, default=None, help="Path to prebuilt RAG FAISS index (.pkl)")
        parser.add_argument("--rag-model-path", type=str, default=None, help="Path to trained RAG contrastive checkpoint")
        parser.add_argument("--rag-base-model", type=str, default="Qwen/Qwen2-Audio-7B-Instruct", help="Base model name for RAG encoder")
        parser.add_argument("--rag-device", type=str, default="cuda:0", help="Device identifier for RAG encoder")
        parser.add_argument("--rag-top-k", type=int, default=10, help="Number of glossary terms to retrieve per chunk")
        parser.add_argument("--rag-target-lang", type=str, default="zh", help="Target language key for term translations")
        parser.add_argument("--rag-lora-r", type=int, default=16, help="LoRA rank for RAG model loading")
        parser.add_argument("--rag-lora-alpha", type=int, default=32, help="LoRA alpha for RAG model loading")
        parser.add_argument("--rag-lora-dropout", type=float, default=0.0, help="LoRA dropout for RAG model loading")

    def build_states(self):
        return S2TAgentStates(
            src_len=0,
            speech_cache=None,
            past_key_values=None,
            target_ids=[],
            segment_idx=0,
            translations_list=[],
            references=[]
        )
    
    def update_multiplier(self, multiplier):
        self.latency_multiplier = multiplier
        # self.source_segment_size = 960 * multiplier
        self.max_new_tokens = 10 * multiplier

    def load_seamless_llama31(self, args):
        patch_llama31()
        patch_hf()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = "<|finetune_right_pad_id|>"

        self.bad_words_ids = []
        if self.suppress_non_language:
            bad_words = ['(', '（']
            for idx in tqdm(range(len(self.tokenizer)), desc="Obtaining bad words ids"):
                decoded_token = self.tokenizer.decode(idx, skip_special_tokens=True)
                if any(bad_word in decoded_token for bad_word in bad_words):
                    self.bad_words_ids.append(idx)

        model = SpeechLlamaForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map='cuda',
        ).eval()

        self.processor = AutoProcessor.from_pretrained(args.seamless_path)

        config = SeamlessM4Tv2Config.from_pretrained(args.seamless_path)
        config.llm_embedding_dim = model.model.embed_tokens.embedding_dim
        config.position_embeddings_type = 'rope'
        config.speech_encoder_chunk_size = args.block_size
        config.speech_encoder_left_chunk_num = args.max_cache_size // args.block_size
        speech_encoder = SeamlessM4Tv2SpeechEncoder(config).eval()
        speech_encoder.to(dtype=model.dtype, device=model.device)
        model.model.speech_encoder = speech_encoder

        model.preprocess(tokenizer=self.tokenizer, max_multiplier=self.max_latency_multiplier, resize=False)

        logger.info("Loading SLLM weights from {}".format(args.state_dict_path))
        state_dict = torch.load(args.state_dict_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict, strict=True)
    
        self.model = model
        self.model.model.inference = True
        self.llama31 = '3.1' in args.model_name

    def load_w2v2_llama31(self, args):
        patch_w2v2(args.rope)
        patch_llama31()
        patch_hf()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = "<|finetune_right_pad_id|>"

        self.bad_words_ids = []
        if self.suppress_non_language:
            bad_words = ['(', '（', '"', '“']
            for idx in tqdm(range(len(self.tokenizer)), desc="Obtaining bad words ids"):
                decoded_token = self.tokenizer.decode(idx, skip_special_tokens=True)
                if any(bad_word in decoded_token for bad_word in bad_words):
                    self.bad_words_ids.append(idx)

        self.model = SpeechLlamaForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map='cuda',
        ).eval()

        speech_encoder_args = [
            args.w2v2_path,
            args.ctc_finetuned,
            args.length_shrink_cfg,
            
            args.block_size,
            args.max_cache_size,
            self.model.model.embed_tokens.embedding_dim,
            None,
            bool(args.rope)
        ]
        if args.w2v2_type == 'w2v2':
            speech_encoder = SpeechEncoderW2V2RoPE(*speech_encoder_args)
        else:
            raise ValueError(f"Unsupported type: {args.w2v2_type}")
        speech_encoder.eval()
        speech_encoder.to(dtype=self.model.dtype, device=self.model.device)
        self.length_shrink_func = speech_encoder._get_feat_extract_output_lengths
        
        self.model.model.speech_encoder = speech_encoder
        self.model.preprocess(tokenizer=self.tokenizer, max_multiplier=self.max_latency_multiplier, resize=False)

        logger.info("Loading SLLM weights from {}".format(args.state_dict_path))
        state_dict = torch.load(args.state_dict_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)

        if args.lora_path:
            logger.info(f"Loading LORA weights from {args.lora_path}")
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=True,
                r=args.lora_rank,
                target_modules='all-linear',
                lora_alpha=16,
                lora_dropout=0.1,
            )
            self.model = get_peft_model(self.model, lora_config, adapter_name='lora_adapter')
            
            lora_state_dict = torch.load(args.lora_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(lora_state_dict, strict=False)
            self.model = self.model.merge_and_unload()

        self.model.model.inference = True
        self.llama31 = '3.1' in args.model_name
    
    def load_w2v2_qwen25(self, args):
        patch_w2v2(args.rope)
        patch_qwen25()
        patch_hf()

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.model_name,
            padding_side="right",
            use_fast=False,
        )
        self.tokenizer.pad_token = "<|finetune_right_pad_id|>"

        self.bad_words_ids = []
        if self.suppress_non_language:
            bad_words = ['(', '（']
            for idx in tqdm(range(len(self.tokenizer)), desc="Obtaining bad words ids"):
                decoded_token = self.tokenizer.decode(idx, skip_special_tokens=True)
                if any(bad_word in decoded_token for bad_word in bad_words):
                    self.bad_words_ids.append(idx)

        self.model = SpeechQwenForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map='cuda',
        ).eval()

        speech_encoder_args = [
            args.w2v2_path,
            args.ctc_finetuned,
            args.length_shrink_cfg,
            
            args.block_size,
            args.max_cache_size,
            self.model.model.embed_tokens.embedding_dim,
            None,
            bool(args.rope),
            False,
        ]
        if args.w2v2_type == 'w2v2':
            speech_encoder = SpeechEncoderW2V2RoPE(*speech_encoder_args)
        else:
            raise ValueError(f"Unsupported type: {args.w2v2_type}")
        speech_encoder.eval()
        speech_encoder.to(dtype=self.model.dtype, device=self.model.device)
        self.length_shrink_func = speech_encoder._get_feat_extract_output_lengths
        
        self.model.model.speech_encoder = speech_encoder
        self.model.preprocess(tokenizer=self.tokenizer, resize=False)

        logger.info("Loading SLLM weights from {}".format(args.state_dict_path))
        state_dict = torch.load(args.state_dict_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)

        if args.lora_path:
            logger.info(f"Loading LORA weights from {args.lora_path}")
            lora_config = LoraConfig(
                task_type="CAUSAL_LM",
                inference_mode=True,
                r=args.lora_rank,
                target_modules='all-linear',
                lora_alpha=16,
                lora_dropout=0.1,
            )
            self.model = get_peft_model(self.model, lora_config, adapter_name='lora_adapter')
            
            lora_state_dict = torch.load(args.lora_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(lora_state_dict, strict=False)
            self.model = self.model.merge_and_unload()

        self.model.model.inference = True

    def load_model(self, args):
        if args.model_type == "w2v2_llama31":
            self.load_w2v2_llama31(args)
        elif args.model_type == "seamless_llama31":
            self.load_seamless_llama31(args)
        elif args.model_type == 'w2v2_qwen25':
            self.load_w2v2_qwen25(args)
        elif args.model_type == "qwen3_omni":
            self.load_qwen3_omni(args)
        else:
            raise ValueError(f"Unsupported model type: {args.model_type}")

    def _prepare_speech(self, states):
        sp_seg_frame = int(self.args.block_size * 0.02 * 16000) * self.latency_multiplier
        
        # Only tensorize the new part
        if len(states.source) > states.MAX_SRC_LEN:
            states.src_len -= len(states.source) - states.MAX_SRC_LEN
            states.source = states.source[-states.MAX_SRC_LEN:]

        source = states.source
        if self.audio_normalize:
            source = normalize(torch.tensor(states.source).unsqueeze(0))[0].tolist()
        
        new_segment = source[states.src_len:]
        if len(new_segment) > 0:
            rag_audio_tensor = torch.tensor(new_segment, dtype=torch.float32)
        else:
            rag_audio_tensor = torch.tensor([], dtype=torch.float32)

        source = torch.tensor(new_segment)
        
        # Pad if needed
        if source.size(0) % sp_seg_frame != 0:
            n_pad = sp_seg_frame - source.size(0) % sp_seg_frame
            source = torch.cat([source, torch.zeros(n_pad).to(source)], dim=0)
            
        # Add offset only for first chunk
        if states.src_len == 0:
            offset = torch.zeros(79 + 320).to(source)
            source = torch.cat([offset, source], dim=0)
        
        if self.args.model_type == "seamless_llama31":
            if states.src_len > 0:
                if states.src_len >= sp_seg_frame + 320 + 79:
                    source_left_pad = states.source[states.src_len - sp_seg_frame - 320 - 79 : states.src_len]
                elif states.src_len == sp_seg_frame:
                    offset = [0.] * (79 + 320)
                    source_left_pad = offset + states.source[: states.src_len]
                else:
                    raise ValueError(f"Invalid source length: {len(states.source)}")
                source_left_pad = torch.tensor(source_left_pad)
                source = torch.cat([source_left_pad, source], dim=0)
            
            source = self.processor(
                audios=source.numpy(), 
                sampling_rate=16000,
                do_normalize_per_mel_bins=False, 
                return_tensors="pt",
            )['input_features'][0, -self.args.block_size * self.latency_multiplier:]
        
        states.src_len = len(states.source)

        speech_batch = source.unsqueeze(0).to(device=self.model.device, dtype=self.model.dtype)
        return speech_batch, rag_audio_tensor

    def _prepare_inputs(self, states):
        messages = []
        sp_seg_token = self.args.block_size // 4 if 'w2v2' in self.args.model_type else self.args.block_size // 8
        sp_seg_token *= self.latency_multiplier
        if states.speech_cache is None:
            latency_token = DEFAULT_LATENCY_TOKEN.format(self.latency_multiplier)
            messages.append(
                {
                    "role": "system",
                    "content": f"Translate the following speeches from {self.source_lang} to {self.target_lang} as a simultaneous interpreter."
                }
            )
            self.system_prompt_size = self.tokenizer.apply_chat_template(
                [messages],
                return_tensors='pt',
                padding=True, 
                truncation=False, 
                add_special_tokens=False
            ).size(1)
        reference_block = ""
        if states.references:
            reference_payload = {"reference": states.references}
            reference_block = json.dumps(reference_payload, ensure_ascii=False) + "\n"
        user_content = reference_block + (sp_seg_token * DEFAULT_SPEECH_PATCH_TOKEN)
        messages.append(
            {
                "role": "user",
                "content": user_content
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": "",
            }
        )
        input_ids = self.tokenizer.apply_chat_template(
            [messages],
            return_tensors='pt',
            padding=True, 
            truncation=False, 
            add_special_tokens=False
        )        
        assert self.args.model_type in ["w2v2_llama31", "w2v2_qwen25"]
        if self.args.model_type == "w2v2_llama31":
            input_ids = input_ids[:, :-1]
        elif self.args.model_type == "w2v2_qwen25":
            input_ids = input_ids[:, :-2]
        # to remove system prompt and preserve last EOT
        if states.speech_cache is not None:
            if self.args.model_type == "w2v2_llama31":
                if self.llama31:
                    input_ids = input_ids[:, 25:] 
                else:
                    input_ids[:, 0] = self.tokenizer.eos_token_id # llama-3-8B-instruct
            elif self.args.model_type == "w2v2_qwen25":
                input_ids = input_ids[:, 19:]
        input_ids = input_ids.to(device=self.model.device)
        return input_ids

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
        
        if states.source_finished and length_in_seconds < 0.32:
            return WriteAction(content="", finished=True)
        
        with synchronized_timer('generate'):
            speech_batch, rag_audio_tensor = self._prepare_speech(states)
            if self.rag_retriever:
                if rag_audio_tensor.numel() > 0:
                    references = self.rag_retriever.retrieve(
                        rag_audio_tensor,
                        top_k=self.rag_top_k,
                        target_lang=self.rag_target_lang,
                    )
                    states.references = references
                    if references:
                        print(f"[RAG] {json.dumps({'reference': references}, ensure_ascii=False)}")
                    else:
                        states.references = []
                else:
                    states.references = []
            else:
                states.references = []
            input_ids = self._prepare_inputs(states)

            if self.args.model_type == "seamless_llama31":
                speech_batch = speech_batch.repeat(self.pseudo_batch_size, 1, 1)
            else:
                speech_batch = speech_batch.repeat(self.pseudo_batch_size, 1)
            input_ids = input_ids.repeat(self.pseudo_batch_size, 1)
            if states.speech_cache is not None:
                for i, (k, v) in enumerate(states.past_key_values):
                    states.past_key_values.key_cache[i] = k.repeat(self.pseudo_batch_size, 1, 1, 1)
                    states.past_key_values.value_cache[i] = v.repeat(self.pseudo_batch_size, 1, 1, 1)
            
            encoder_input_ids = torch.tensor(
                states.target_ids[-self.no_repeat_ngram_lookback:]
            ).unsqueeze(0).to(self.model.device)
            encoder_input_ids = encoder_input_ids.repeat(self.pseudo_batch_size, 1)

            if states.source_finished:
                states.segment_idx = -1            

            self.model.model.speech_features_extracted = False
            outputs = self.model.generate(
                attention_mask=None,
                input_ids=input_ids,
                speech_batch=speech_batch,
                do_sample=self.do_sample,
                top_p=self.top_p,
                top_k=self.top_k,
                epsilon_cutoff=self.epsilon_cutoff,
                temperature=self.temperature,
                num_beams=self.beam,
                max_new_tokens=self.max_new_tokens,
                num_return_sequences=1,
                encoder_input_ids=encoder_input_ids,
                encoder_no_repeat_ngram_size=self.no_repeat_ngram_size,
                no_repeat_ngram_size=self.no_repeat_ngram_size,
                # encoder_repetition_penalty=self.repetition_penalty,
                repetition_penalty=self.repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                return_legacy_cache=False,
                use_cache=True,
                past_key_values=states.past_key_values,
                suppress_tokens=self.bad_words_ids,
                states=states,
                multiplier=self.latency_multiplier,
            )

            states.past_key_values = outputs.past_key_values
            if self.beam > 1:
                states.past_key_values = states.past_key_values[0]
            cur_llm_cache_size = states.past_key_values[0][0].size(2)
            self.cache_checkpoints.append(cur_llm_cache_size)

            if cur_llm_cache_size > self.max_llm_cache_size:
                new_llm_cache_size = 0
                for i, ckpt in enumerate(self.cache_checkpoints):
                    new_llm_cache_size = cur_llm_cache_size - ckpt
                    if new_llm_cache_size <= self.max_llm_cache_size:
                        self.cache_checkpoints = self.cache_checkpoints[i + 1:]
                        n_cache_trimmed = ckpt
                        if self.always_cache_system_prompt:
                            n_cache_trimmed -= self.system_prompt_size
                        self.cache_checkpoints = [
                            ckpt - n_cache_trimmed for ckpt in self.cache_checkpoints
                        ]
                        break

                for i, (k, v) in enumerate(states.past_key_values):
                    k_cache = k[:, :, -new_llm_cache_size:]
                    v_cache = v[:, :, -new_llm_cache_size:]
                    if self.always_cache_system_prompt:
                        k_cache = torch.cat([k[:, :, :self.system_prompt_size], k_cache], dim=2)
                        v_cache = torch.cat([v[:, :, :self.system_prompt_size], v_cache], dim=2)
                    states.past_key_values.key_cache[i] = k_cache
                    states.past_key_values.value_cache[i] = v_cache

        output_ids = outputs.sequences[0, input_ids.size(1):-1].tolist()
        
        states.target_ids.extend(output_ids)
        translation = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        translation = re.sub(r'[（）()"“”�]', '', translation)

        if self.dpo_sampling:
            # Format translation with single quotes and proper UTF-8
            formatted_translation = f"'{translation}'" if translation else "''"
            states.translations_list.append(formatted_translation)
            
            if states.source_finished:
                try:
                    with open(self.output_file, 'a', encoding='utf-8') as f:
                        formatted_list = f"[{', '.join(states.translations_list)}]"
                        f.write(formatted_list + '\n')
                    states.translations_list = []
                except Exception as e:
                    print(f"Error writing translations to file: {e}")
        # print(f"{length_in_seconds / 60:.2f}", ':', self.tokenizer.decode(states.target_ids))
        # print(f"Speech length in minutes: {length_in_seconds / 60:.2f}")
        print(states.past_key_values[0][0].size(2), ' '.join(states.target))

        # print(states.segment_idx, ":", translation)
        states.segment_idx += 1

        if translation != '' or states.source_finished:
            return WriteAction(
                content=translation,
                finished=states.source_finished,
            )
        else:
            return ReadAction()