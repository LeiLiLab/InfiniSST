import queue
from typing import List
from collections import Counter
import numpy as np
import torch
import flashinfer

from model.llama31 import SpeechLlamaModel

PAGE_SIZE = 16

class PageTable:
    def __init__(self, max_batch_size, max_steps, layer, q_heads, kv_heads, kv_dim, device, dtype=torch.bfloat16, wrapper_type='prefill'):
        self.max_steps = max_steps
        max_num_pages = 4 * max_batch_size * (max_steps + PAGE_SIZE - 1) // PAGE_SIZE

        self.paged_kv_cache = torch.zeros(
            layer, max_num_pages, 2, PAGE_SIZE, kv_heads, kv_dim, 
            dtype=dtype, device=device
        ) # NHD
        self.paged_queue = list(range(max_num_pages))
        self.page_cnt = torch.zeros(max_num_pages, dtype=torch.int32)
        self.workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)

        self.wrapper_type = wrapper_type
        if wrapper_type == 'prefill':
            self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD"
            )
            self.wrapper.plan(
                torch.tensor([0, 1], dtype=torch.int32, device=device),
                torch.tensor([0, 1], dtype=torch.int32, device=device),
                torch.tensor([0], dtype=torch.int32, device=device),
                torch.tensor([16], dtype=torch.int32, device=device),
                q_heads,
                kv_heads,
                kv_dim,
                PAGE_SIZE,
                causal=True,
                pos_encoding_mode='ROPE_LLAMA',
                q_data_type=dtype,
                kv_data_type=dtype,
            )
        else:
            self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD", use_tensor_cores=True
            )
            self.wrapper.plan(
                torch.tensor([0, 1], dtype=torch.int32, device=device),
                torch.tensor([0], dtype=torch.int32, device=device),
                torch.tensor([16], dtype=torch.int32, device=device),
                q_heads,
                kv_heads,
                kv_dim,
                PAGE_SIZE,
                pos_encoding_mode='ROPE_LLAMA',
                q_data_type=dtype,
                kv_data_type=dtype,
            )

        

class LLMCache:
    paged_kv_indices: torch.Tensor = None
    paged_kv_last_page_len: int = None

    def __init__(self):
        self.paged_kv_indices = []
        self.paged_kv_last_page_len = PAGE_SIZE

class SpeechCache:
    src: torch.Tensor = None
    src_len: int = 0

    paged_kv_indices: torch.Tensor = None
    paged_kv_last_page_len: int = None

    def __init__(self, src=None, src_len=0):
        self.src = src
        self.src_len = src_len

        self.paged_kv_indices = []
        self.paged_kv_last_page_len = PAGE_SIZE

def get_cache_size(paged_kv_indices, paged_kv_last_page_len):
    return (len(paged_kv_indices) - 1) * PAGE_SIZE + paged_kv_last_page_len

def init_paged_kv_cache(
    max_batch_size, 
    max_speech_steps, speech_layer, speech_kv_heads, speech_kv_dim, 
    max_llm_steps, llm_layer, llm_q_heads, llm_kv_heads, llm_kv_dim, 
    dtype=torch.bfloat16, device_prefill='cuda:0', device_decode='cuda:1'
):
    # speech prefill
    speech_pagetable = PageTable(
        max_batch_size, max_speech_steps, speech_layer, speech_kv_heads, speech_kv_heads, speech_kv_dim, 
        device_prefill, dtype=dtype, wrapper_type='prefill'
    )

    # llm prefill
    llm_prefill_pagetable = PageTable(
        max_batch_size, max_llm_steps, llm_layer, llm_q_heads, llm_kv_heads, llm_kv_dim, 
        device_prefill, dtype=dtype, wrapper_type='prefill'
    )

    # llm decode
    llm_decode_pagetable = PageTable(
        max_batch_size, max_llm_steps, llm_layer, llm_q_heads, llm_kv_heads, llm_kv_dim, 
        device_decode, dtype=dtype, wrapper_type='decode'
    )

    if device_prefill == device_decode:
        llm_decode_pagetable.paged_queue = llm_prefill_pagetable.paged_queue
        llm_decode_pagetable.paged_kv_cache = llm_prefill_pagetable.paged_kv_cache
        llm_decode_pagetable.page_cnt = llm_prefill_pagetable.page_cnt

    return speech_pagetable, llm_prefill_pagetable, llm_decode_pagetable

def allocate_paged_kv_cache(
    pagetable,
    paged_kv_indices,
    paged_kv_last_page_len,
    n,
):
    if paged_kv_last_page_len + n <= PAGE_SIZE:
        paged_kv_last_page_len += n
    else:
        num_new_page = (n - (PAGE_SIZE - paged_kv_last_page_len) + PAGE_SIZE - 1) // PAGE_SIZE
        page_indices = pagetable.paged_queue[:num_new_page]
        paged_kv_indices.extend(page_indices)
        pagetable.page_cnt[page_indices] += 1
        for _ in range(num_new_page):
            pagetable.paged_queue.pop(0)
        paged_kv_last_page_len = (n - (PAGE_SIZE - paged_kv_last_page_len) - 1) % PAGE_SIZE + 1
    return pagetable, paged_kv_indices, paged_kv_last_page_len

def pop_paged_kv_cache(
    pagetable,
    paged_kv_indices,
    paged_kv_last_page_len,
    max_steps, # preserve kv cache for last max_steps of tokens
    max_steps_start=0, # preserve kv cache for first max_steps_start tokens
):
    kv_cache_size = get_cache_size(paged_kv_indices, paged_kv_last_page_len)
    kv_cache_size_start = (max_steps_start + PAGE_SIZE - 1) // PAGE_SIZE * PAGE_SIZE

    if kv_cache_size - kv_cache_size_start > max_steps:
        num_pages_to_pop = (kv_cache_size - kv_cache_size_start - max_steps + PAGE_SIZE - 1) // PAGE_SIZE
        num_pages_start = kv_cache_size_start // PAGE_SIZE

        # Convert page indices to tensor for faster operations
        page_indices_to_pop = torch.tensor(paged_kv_indices[num_pages_start : num_pages_start + num_pages_to_pop])
        
        # Batch update page counts
        pagetable.page_cnt[page_indices_to_pop] -= 1
        
        # Get indices of pages that can be freed
        free_mask = pagetable.page_cnt[page_indices_to_pop] == 0
        free_indices = page_indices_to_pop[free_mask]
        
        # Update paged queue and indices in one operation
        if len(free_indices) > 0:
            # Convert to list only once for queue update
            free_indices_list = free_indices.tolist()
            pagetable.paged_queue.extend(free_indices_list)
            
            # Update paged_kv_indices in one operation
            paged_kv_indices = paged_kv_indices[:num_pages_start] + paged_kv_indices[num_pages_start + num_pages_to_pop:]

    return pagetable, paged_kv_indices, paged_kv_last_page_len


def copy_paged_kv_cache(
    src_paged_kv_indices,
    src_paged_kv_last_page_len,
    src_pagetable,
    tgt_pagetable,
):
    src_device = src_pagetable.paged_kv_cache.device
    tgt_device = tgt_pagetable.paged_kv_cache.device

    src_size = get_cache_size(src_paged_kv_indices, src_paged_kv_last_page_len)

    # layer, max_num_pages, 2, PAGE_SIZE, kv_heads, kv_dim, 
    tgt_paged_kv_indices = []
    tgt_paged_kv_last_page_len = 16
    tgt_pagetable, tgt_paged_kv_indices, tgt_paged_kv_last_page_len = \
        allocate_paged_kv_cache(
            tgt_pagetable,
            tgt_paged_kv_indices,
            tgt_paged_kv_last_page_len,
            src_size
        )

    tgt_pagetable.paged_kv_cache[:, tgt_paged_kv_indices] = \
        src_pagetable.paged_kv_cache[:, src_paged_kv_indices].to(tgt_device)
    
    return tgt_pagetable, tgt_paged_kv_indices, tgt_paged_kv_last_page_len

def duplicate_paged_kv_cache(
    paged_kv_indices,
    paged_kv_last_page_len,
    pagetable,
):
    # layer, max_num_pages, 2, PAGE_SIZE, kv_heads, kv_dim, 
    tgt_paged_kv_indices = paged_kv_indices[:-1]
    tgt_paged_kv_last_page_len = 16

    pagetable.page_cnt[tgt_paged_kv_indices] += 1

    pagetable, tgt_paged_kv_indices, tgt_paged_kv_last_page_len = \
        allocate_paged_kv_cache(
            pagetable,
            tgt_paged_kv_indices,
            tgt_paged_kv_last_page_len,
            paged_kv_last_page_len
        )

    pagetable.paged_kv_cache[:, tgt_paged_kv_indices[-1]] = \
        pagetable.paged_kv_cache[:, paged_kv_indices[-1]]
    
    return pagetable, tgt_paged_kv_indices, tgt_paged_kv_last_page_len


def move_paged_kv_cache(
    src_paged_kv_indices,
    src_paged_kv_last_page_len,
    src_pagetable,
    tgt_pagetable,
):
    if src_pagetable.paged_kv_cache.device == tgt_pagetable.paged_kv_cache.device:
        return src_pagetable, tgt_pagetable, src_paged_kv_indices, src_paged_kv_last_page_len

    tgt_pagetable, tgt_paged_kv_indices, tgt_paged_kv_last_page_len = \
        copy_paged_kv_cache(
            src_paged_kv_indices,
            src_paged_kv_last_page_len,
            src_pagetable,
            tgt_pagetable,
        )

    src_pagetable, _, _ = \
        pop_paged_kv_cache(
            src_pagetable,
            src_paged_kv_indices,
            src_paged_kv_last_page_len,
            0,
        )
    
    return src_pagetable, tgt_pagetable, tgt_paged_kv_indices, tgt_paged_kv_last_page_len