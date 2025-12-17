import os

# ⚠️ CRITICAL: Set VLLM_USE_V1=0 BEFORE importing vllm
# This forces vLLM to use the stable v0 engine instead of the experimental v1 engine
# Must be set before any vllm imports
os.environ['VLLM_USE_V1'] = '0'

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
from transformers import (
    AutoProcessor, 
    Qwen3OmniMoeThinkerForConditionalGeneration, 
    Qwen3OmniMoeForConditionalGeneration, 
    Qwen3OmniMoeProcessor, 
    GenerationConfig, 
    Qwen3OmniMoeConfig
)
from qwen_omni_utils import process_mm_info

from vllm import LLM, SamplingParams

from tqdm import tqdm

from agents.options import (
    add_simuleval_args,
    add_gen_args,
)

import logging
logger = logging.getLogger(__name__)

try:
    import faiss  # type: ignore
except ImportError:
    faiss = None

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


class ProcessorAudioAlias:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, *args, **kwargs):
        if "audios" in kwargs and "audio" not in kwargs:
            kwargs["audio"] = kwargs.pop("audios")
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
        score_threshold: float = 0.5,
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
        self.score_threshold = float(max(0.0, min(1.0, score_threshold)))
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
                (
                    "TermRAGRetriever initialized with %d terms "
                    "(embedding_dim=%d, top_k=%d, target_lang=%s, confidence_threshold=%.2f)"
                ),
                len(self.term_list),
                self.embedding_dim,
                self.top_k,
                self.target_lang,
                self.score_threshold,
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

    @staticmethod
    def _distance_to_confidence(distance: float) -> float:
        if not np.isfinite(distance):
            return 0.0
        cosine = 1.0 - 0.5 * distance
        cosine = max(-1.0, min(1.0, cosine))
        confidence = (cosine + 1.0) / 2.0
        return float(max(0.0, min(1.0, confidence)))

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
        candidate_logs: List[Dict[str, object]] = []

        for distance, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.term_list):
                continue
            term_entry = self.term_list[idx]
            if not isinstance(term_entry, dict):
                continue
            term = term_entry.get("term", "")
            if not term or term in seen_terms:
                continue
            seen_terms.add(term)
            confidence = self._distance_to_confidence(float(distance))
            candidate_logs.append(
                {
                    "term": term,
                    "confidence": round(confidence, 4),
                }
            )
            if confidence < self.score_threshold:
                continue
            translation = ""
            translations = term_entry.get("target_translations") or {}
            if isinstance(translations, dict):
                translation = translations.get(target_lang) or translations.get(target_lang.upper()) or ""
            results.append({"term": term, "translation": translation})
            if len(results) >= max_k:
                break
        if candidate_logs:
            logger.info(
                "RAG candidates (threshold=%.2f): %s",
                self.score_threshold,
                json.dumps(candidate_logs, ensure_ascii=False),
            )
        if not results and candidate_logs:
            logger.info(
                "RAG references suppressed because all confidences are below threshold %.2f",
                self.score_threshold,
            )
        return results


@dataclass
class S2TAgentStates(AgentStates):
    src_len: int
    target_ids: list
    segment_idx: int
    messages: list
    references: list
    MAX_SRC_LEN = 16000 * 30

    def reset(self):
        super().reset()
        self.src_len = 0
        self.target_ids = []
        self.segment_idx = 0
        self.messages = []
        self.references = []


@entrypoint
class InfiniSSTOmniVLLMRAG(SpeechToTextAgent):

    def __init__(self, args):
        super().__init__(args)
        transformers.set_seed(998244353)

        # simuleval
        self.min_start_sec = args.min_start_sec
        self.source_lang = args.source_lang
        self.target_lang = args.target_lang
        
        # gen
        self.beam = args.beam
        self.max_new_tokens = args.max_new_tokens
        self.do_sample = args.do_sample
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.temperature = args.temperature

        self.generation_config = GenerationConfig(  
            num_beams=self.beam,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            max_new_tokens=self.max_new_tokens,
        )

        # cache
        self.max_cache_chunks = args.max_cache_chunks
        self.keep_cache_chunks = args.keep_cache_chunks
        
        # RAG retriever
        self.rag_retriever: Optional[TermRAGRetriever] = None
        self.rag_top_k = getattr(args, "rag_top_k", 5)
        self.rag_target_lang = getattr(args, "rag_target_lang", "zh")
        self.rag_conf_threshold = getattr(args, "rag_confidence_threshold", 0.5)
        if getattr(args, "rag_enabled", False):
            logger.info("Initializing RAG retriever...")
            self.rag_retriever = TermRAGRetriever(
                index_path=getattr(args, "rag_index_path", None),
                model_path=getattr(args, "rag_model_path", None),
                base_model_name=getattr(args, "rag_base_model", "Qwen/Qwen2-Audio-7B-Instruct"),
                device=getattr(args, "rag_device", "cuda:1"),
                lora_r=getattr(args, "rag_lora_r", 16),
                lora_alpha=getattr(args, "rag_lora_alpha", 32),
                lora_dropout=getattr(args, "rag_lora_dropout", 0.0),
                top_k=self.rag_top_k,
                target_lang=self.rag_target_lang,
                score_threshold=self.rag_conf_threshold,
            )
            if not self.rag_retriever or not self.rag_retriever.enabled:
                logger.warning("RAG retriever not operational; continuing without references")
                self.rag_retriever = None
        else:
            logger.info("RAG retrieval disabled for InfiniSSTOmniVLLMRAG agent")
        
        # model
        self.use_vllm = args.use_vllm
        self.load_model(args)
    
    @staticmethod
    def add_args(parser):
        add_simuleval_args(parser)
        add_gen_args(parser)
        parser.add_argument("--model-name", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
        parser.add_argument("--use-vllm", type=int, default=0)
        parser.add_argument("--max-cache-chunks", type=int, default=120)
        parser.add_argument("--keep-cache-chunks", type=int, default=60)
        parser.add_argument("--rag-enabled", action="store_true", help="Enable glossary RAG retrieval for prompt augmentation")
        parser.add_argument("--rag-index-path", type=str, default=None, help="Path to prebuilt RAG FAISS index (.pkl)")
        parser.add_argument("--rag-model-path", type=str, default=None, help="Path to trained RAG contrastive checkpoint")
        parser.add_argument("--rag-base-model", type=str, default="Qwen/Qwen2-Audio-7B-Instruct", help="Base model name for RAG encoder")
        parser.add_argument("--rag-device", type=str, default="cuda:1", help="Device identifier for RAG encoder")
        parser.add_argument("--rag-top-k", type=int, default=5, help="Number of glossary terms to retrieve per chunk")
        parser.add_argument("--rag-target-lang", type=str, default="zh", help="Target language key for term translations")
        parser.add_argument("--rag-lora-r", type=int, default=16, help="LoRA rank for RAG model loading")
        parser.add_argument("--rag-lora-alpha", type=int, default=32, help="LoRA alpha for RAG model loading")
        parser.add_argument("--rag-lora-dropout", type=float, default=0.0, help="LoRA dropout for RAG model loading")
        parser.add_argument(
            "--rag-confidence-threshold",
            type=float,
            default=0.5,
            help="Minimum confidence (0-1) required to keep retrieved glossary references",
        )

    def build_states(self):
        return S2TAgentStates(
            src_len=0,
            target_ids=[],
            segment_idx=0,
            messages=[],
            references=[],
        )

    def load_model(self, args):
        if args.use_vllm:
            """
            vllm serve /data/user_data/siqiouya/ckpts/test_swift/Qwen3-Omni-30B-A3B-Instruct-lora/v1-20251104-033331-hf \
                --gpu-memory-utilization 0.9 \
                --tensor-parallel-size 2 \
                --limit-mm-per-prompt '{"audio": 60}' \
                --max-model-len 2048 \
                --enable-prefix-caching  
            """
            # GPU Allocation Strategy:
            # - RAG uses cuda:2 (specified in --rag-device)
            # - vLLM uses TP=2, which will automatically use GPU 0 and 1
            # - Reduced gpu_memory_utilization to 0.9 to leave more headroom for MoE models
            gpu_memory_util = 0.9  # Reduced from 0.95 for stability
            tp_size = 2
            
            logger.info(f"="*80)
            logger.info(f"vLLM Configuration:")
            logger.info(f"  Model: {args.model_name}")
            logger.info(f"  Tensor Parallel Size: {tp_size}")
            logger.info(f"  GPU Memory Utilization: {gpu_memory_util}")
            logger.info(f"  Max Model Len: 2048")
            logger.info(f"  Enable Prefix Caching: True")
            logger.info(f"  Limit MM Per Prompt (audio): {self.max_cache_chunks}")
            
            if self.rag_retriever and self.rag_retriever.enabled:
                logger.info(f"  RAG Status: Enabled on separate GPU")
            else:
                logger.info(f"  RAG Status: Disabled")
            logger.info(f"="*80)
            
            logger.info(f"Initializing vLLM engine... This may take a few minutes.")
            
            self.model = LLM(
                model=args.model_name, 
                trust_remote_code=True, 
                gpu_memory_utilization=gpu_memory_util,
                tensor_parallel_size=tp_size,
                limit_mm_per_prompt={'audio': self.max_cache_chunks},
                max_num_seqs=1,
                max_model_len=10240,
                enable_prefix_caching=True,
                enforce_eager=False,  # Use CUDA graphs for better performance
            )
            
            logger.info(f"✅ vLLM engine initialized successfully!")
            self.sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                max_tokens=self.max_new_tokens,
            )
        else:
            self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
                args.model_name,
                dtype="auto",
                device_map="auto",
                attn_implementation="flash_attention_2",
                enable_audio_output=False,
            )
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(args.model_name)

    def _normalize_references(self, references: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Normalize references to match the training format exactly.
        Only keep 'term' and 'translation' fields, filter out empty terms.
        """
        norm_refs = []
        for r in references:
            term = r.get("term", "")
            if not term:
                continue
            norm_refs.append({
                "term": term,
                "translation": r.get("translation", "")
            })
        return norm_refs

    def _prepare_speech(self, states):        
        # Only tensorize the new part
        if len(states.source) > states.MAX_SRC_LEN:
            states.src_len -= len(states.source) - states.MAX_SRC_LEN
            states.source = states.source[-states.MAX_SRC_LEN:]

        increment = np.array(states.source[states.src_len:])   

        if len(increment) < 15360:
            increment = np.pad(increment, (0, 15360 - len(increment)), mode='constant', constant_values=0)

        states.src_len = len(states.source)
        return increment

    def _prepare_inputs(self, states, increment, references):
        if len(states.messages) == 0:
            states.messages.append(
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": f"You are a professional simultaneous interpreter. You will be given chunks of English audio and you need to translate the audio into Chinese text. Use the wordlings for term reference."}
                    ]
                }
            )
        
        # Build user content with audio and references
        user_content = [{"type": "audio", "audio": increment}]
        
        # ✅ Match training format exactly: "<audio>, references: [{...}, {...}]"
        # Normalize references to only keep 'term' and 'translation' fields
        # norm_refs = self._normalize_references(references)
        # if norm_refs:
        #     reference_text = f", references: {json.dumps(norm_refs, ensure_ascii=False)}"
        #     user_content.append({"type": "text", "text": reference_text})
        reference_text = "\n\nwordlings: {}"
        user_content.append({"type": "text", "text": reference_text})

        states.messages.append(
            {
                "role": "user",
                "content": user_content
            }
        )

        print("len(messages):", len(states.messages))

        text = self.processor.apply_chat_template(
            states.messages, 
            add_generation_prompt=True, 
            tokenize=False
        )
        audios, images, videos = process_mm_info(states.messages, use_audio_in_video=False)
        print("len(audios):", len(audios))

        if self.use_vllm:
            inputs = {
                'prompt': text,
                'multi_modal_data': {
                    'audio': audios,
                },
                "mm_processor_kwargs": {
                    "use_audio_in_video": False,
                },
            }

            input_ids = self.processor(
                text=text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=False
            )['input_ids']
            print("input_ids size:", input_ids.size())
        else:
            inputs = self.processor(
                text=text, 
                audio=audios, 
                images=images, 
                videos=videos, 
                return_tensors="pt", 
                padding=True, 
                use_audio_in_video=False
            )
            inputs['input_features'] = inputs['input_features'].to(self.model.dtype)
        return inputs

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
            increment = self._prepare_speech(states)
            
            # RAG retrieval
            references: List[Dict[str, str]] = []
            if self.rag_retriever:
                rag_audio_tensor = torch.tensor(increment, dtype=torch.float32)
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
            
            inputs = self._prepare_inputs(states, increment, references)
            print(f"inputs:\n{inputs}")

            if self.use_vllm:
                outputs = self.model.generate(
                    [inputs], 
                    sampling_params=self.sampling_params,
                    use_tqdm=False,
                )
                translation = outputs[0].outputs[0].text
            else:
                text_ids, _ = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config,
                    return_audio=False,
                    thinker_return_dict_in_generate=True,
                    use_audio_in_video=False,
                )
                translation = self.processor.batch_decode(
                    text_ids.sequences[:, inputs["input_ids"].shape[1] :],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False
                )[0]

            states.messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": translation}]
                }
            )

            if len(states.messages) >= 2 * self.max_cache_chunks + 1:
                print("before trim:", len(states.messages))
                states.messages = [states.messages[0]] + states.messages[-2 * self.keep_cache_chunks:]
                print("after trim:", len(states.messages))

            if states.source_finished:
                states.segment_idx = -1

        print(''.join(states.target))

        # print(states.segment_idx, ":", translation)
        states.segment_idx += 1

        if translation != '' or states.source_finished:
            return WriteAction(
                content=translation,
                finished=states.source_finished,
            )
        else:
            return ReadAction()

