import queue
from typing import List

import torch
import flashinfer

from model.llama31 import SpeechLlamaModel

PAGE_SIZE = 16

class PageTable:
    def __init__(self, max_batch_size, max_steps, layer, kv_heads, kv_dim, device, dtype=torch.bfloat16, wrapper_type='prefill'):
        self.max_steps = max_steps
        max_num_pages = 2 * max_batch_size * (max_steps + PAGE_SIZE - 1) // PAGE_SIZE

        max_num_pages *= 8 # TODO: remove this

        self.paged_kv_cache = torch.zeros(
            layer, max_num_pages, 2, PAGE_SIZE, kv_heads, kv_dim, 
            dtype=dtype, device=device
        ) # NHD
        self.paged_queue = list(range(max_num_pages))
        self.workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)

        self.wrapper_type = wrapper_type
        if wrapper_type == 'prefill':
            self.wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD"
            )
        else:
            self.wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                self.workspace_buffer, "NHD", use_tensor_cores=True
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
    max_llm_steps, llm_layer, llm_kv_heads, llm_kv_dim, 
    dtype=torch.bfloat16, device_prefill='cuda:0', device_decode='cuda:1'
):
    # speech prefill
    speech_pagetable = PageTable(
        max_batch_size, max_speech_steps, speech_layer, speech_kv_heads, speech_kv_dim, 
        device_prefill, dtype=dtype, wrapper_type='prefill'
    )

    # llm prefill
    llm_prefill_pagetable = PageTable(
        max_batch_size, max_llm_steps, llm_layer, llm_kv_heads, llm_kv_dim, 
        device_prefill, dtype=dtype, wrapper_type='prefill'
    )

    # llm decode
    llm_decode_pagetable = PageTable(
        max_batch_size, max_llm_steps, llm_layer, llm_kv_heads, llm_kv_dim, 
        device_decode, dtype=dtype, wrapper_type='decode'
    )

    return speech_pagetable, llm_prefill_pagetable, llm_decode_pagetable

def allocate_paged_kv_cache(
    paged_queue,
    paged_kv_indices,
    paged_kv_last_page_len,
    n,
):
    if paged_kv_last_page_len + n <= PAGE_SIZE:
        paged_kv_last_page_len += n
    else:
        num_new_page = (n - (PAGE_SIZE - paged_kv_last_page_len) + PAGE_SIZE - 1) // PAGE_SIZE
        while num_new_page > 0:
            paged_kv_indices.append(paged_queue.pop(0))
            num_new_page -= 1
        paged_kv_last_page_len = (n - (PAGE_SIZE - paged_kv_last_page_len) - 1) % PAGE_SIZE + 1
    return paged_queue, paged_kv_indices, paged_kv_last_page_len

def pop_paged_kv_cache(
    paged_queue,
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
        paged_queue.extend(paged_kv_indices[num_pages_start : num_pages_start + num_pages_to_pop])
        paged_kv_indices = paged_kv_indices[:num_pages_start] + paged_kv_indices[num_pages_start + num_pages_to_pop:]

    return paged_queue, paged_kv_indices, paged_kv_last_page_len


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
    tgt_pagetable.paged_queue, tgt_paged_kv_indices, tgt_paged_kv_last_page_len = \
        allocate_paged_kv_cache(
            tgt_pagetable.paged_queue,
            tgt_paged_kv_indices,
            tgt_paged_kv_last_page_len,
            src_size
        )

    tgt_pagetable.paged_kv_cache[:, tgt_paged_kv_indices] = \
        src_pagetable.paged_kv_cache[:, src_paged_kv_indices].to(tgt_device)
    
    return tgt_pagetable, tgt_paged_kv_indices, tgt_paged_kv_last_page_len


def move_paged_kv_cache(
    src_paged_kv_indices,
    src_paged_kv_last_page_len,
    src_pagetable,
    tgt_pagetable,
):
    tgt_pagetable, tgt_paged_kv_indices, tgt_paged_kv_last_page_len = \
        copy_paged_kv_cache(
            src_paged_kv_indices,
            src_paged_kv_last_page_len,
            src_pagetable,
            tgt_pagetable,
        )

    src_pagetable.paged_queue, _, _ = \
        pop_paged_kv_cache(
            src_pagetable.paged_queue,
            src_paged_kv_indices,
            src_paged_kv_last_page_len,
            0,
        )
    
    return src_pagetable, tgt_pagetable, tgt_paged_kv_indices, tgt_paged_kv_last_page_len