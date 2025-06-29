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
        # 🔥 紧急修复：增加页面池大小，添加更多缓冲
        # 4 * max_batch_size
        page_multiplier = 4  # 增加页面池倍数
        max_num_pages = page_multiplier * max_batch_size * (max_steps + PAGE_SIZE - 1) // PAGE_SIZE
        
        # 🔍 记录页面池大小
        print(f"🔧 [PageTable-{wrapper_type}] 初始化页面池:")
        print(f"   - max_batch_size: {max_batch_size}")
        print(f"   - max_steps: {max_steps}")
        print(f"   - page_multiplier: {page_multiplier}")
        print(f"   - 总页面数: {max_num_pages}")
        print(f"   - 页面大小: {PAGE_SIZE}")
        print(f"   - 理论最大并发数: {max_num_pages // 6} sessions (假设每session需要6页)")

        self.paged_kv_cache = torch.zeros(
            layer, max_num_pages, 2, PAGE_SIZE, kv_heads, kv_dim, 
            dtype=dtype, device=device
        ) # NHD
        self.paged_queue = list(range(max_num_pages))
        self.page_cnt = torch.zeros(max_num_pages, dtype=torch.int32)
        self.workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
        
        # 🔍 添加页面池监控变量
        self.initial_pages = max_num_pages
        self.peak_usage = 0
        self.allocation_count = 0
        
        # 🔥 新增：session类型追踪
        self.page_session_map = {}  # {page_id: session_info}
        self.session_pages = {}     # {session_id: [page_ids]}
        self.last_access_time = {}  # {session_id: timestamp}

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
                rope_scale=1.0,
                rope_theta=1e4,
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
    
    def _emergency_page_reclaim(self, needed_pages: int) -> int:
        """紧急页面回收：从不活跃的session中回收页面"""
        import time
        current_time = time.time()
        freed_pages = 0
        
        print(f"🚨 [EMERGENCY] 开始紧急页面回收，需要 {needed_pages} 页")
        
        # 获取所有session的最后访问时间，按不活跃程度排序
        inactive_sessions = []
        for session_id, last_access in self.last_access_time.items():
            inactive_time = current_time - last_access
            if session_id in self.session_pages:
                page_count = len(self.session_pages[session_id])
                inactive_sessions.append((session_id, inactive_time, page_count))
        
        # 按不活跃时间排序（最不活跃的优先回收）
        inactive_sessions.sort(key=lambda x: x[1], reverse=True)
        
        print(f"🔍 [EMERGENCY] 找到 {len(inactive_sessions)} 个session可供回收")
        
        # 优先回收5分钟以上不活跃的session
        for session_id, inactive_time, page_count in inactive_sessions:
            if freed_pages >= needed_pages:
                break
                
            # 只回收超过5分钟不活跃的session
            if inactive_time > 300:  # 5分钟
                print(f"🔄 [EMERGENCY] 回收session {session_id} ({page_count} 页，不活跃 {inactive_time:.1f}s)")
                
                # 释放该session的所有页面
                if session_id in self.session_pages:
                    pages_to_free = self.session_pages[session_id].copy()
                    
                    # 减少页面引用计数
                    for page_id in pages_to_free:
                        if self.page_cnt[page_id] > 0:
                            self.page_cnt[page_id] -= 1
                            
                            # 如果引用计数为0，放回页面池
                            if self.page_cnt[page_id] == 0:
                                self.paged_queue.append(page_id)
                                freed_pages += 1
                                
                                # 清理映射
                                if page_id in self.page_session_map:
                                    del self.page_session_map[page_id]
                    
                    # 清理session记录
                    del self.session_pages[session_id]
                    if session_id in self.last_access_time:
                        del self.last_access_time[session_id]
                    
                    print(f"✅ [EMERGENCY] Session {session_id} 释放了 {len(pages_to_free)} 页")
        
        # 如果还是不够，考虑回收较新的session（但给出警告）
        if freed_pages < needed_pages:
            remaining_needed = needed_pages - freed_pages
            print(f"⚠️ [EMERGENCY] 仍需 {remaining_needed} 页，考虑回收较新session")
            
            for session_id, inactive_time, page_count in inactive_sessions:
                if freed_pages >= needed_pages:
                    break
                    
                # 回收1分钟以上不活跃的session
                if inactive_time > 60 and session_id in self.session_pages:  # 1分钟
                    print(f"🔄 [EMERGENCY] 强制回收session {session_id} ({page_count} 页，不活跃 {inactive_time:.1f}s)")
                    
                    pages_to_free = self.session_pages[session_id].copy()
                    for page_id in pages_to_free:
                        if self.page_cnt[page_id] > 0:
                            self.page_cnt[page_id] -= 1
                            if self.page_cnt[page_id] == 0:
                                self.paged_queue.append(page_id)
                                freed_pages += 1
                                if page_id in self.page_session_map:
                                    del self.page_session_map[page_id]
                    
                    del self.session_pages[session_id]
                    if session_id in self.last_access_time:
                        del self.last_access_time[session_id]
        
        print(f"🏁 [EMERGENCY] 紧急回收完成，释放了 {freed_pages} 页")
        return freed_pages
    
    def track_session_page_usage(self, session_id: str, allocated_pages: list):
        """追踪session的页面使用"""
        import time
        
        if session_id not in self.session_pages:
            self.session_pages[session_id] = []
        
        # 添加新分配的页面
        self.session_pages[session_id].extend(allocated_pages)
        
        # 更新页面到session的映射
        for page_id in allocated_pages:
            self.page_session_map[page_id] = {
                'session_id': session_id,
                'allocated_time': time.time()
            }
        
        # 更新最后访问时间
        self.last_access_time[session_id] = time.time()
        
        print(f"📈 [TRACKING] Session {session_id} 现在使用 {len(self.session_pages[session_id])} 页")
    
    def update_session_access_time(self, session_id: str):
        """更新session的最后访问时间"""
        import time
        self.last_access_time[session_id] = time.time()

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
    n,  # n是需要的token数量
    session_priority='normal',  # 🔥 新增：session优先级
    is_existing_session=True,   # 🔥 新增：是否为已有session
    session_id=None,            # 🔥 新增：session标识符
):
    # 🔍 页面分配前的状态检查
    available_pages = len(pagetable.paged_queue)
    used_pages = pagetable.initial_pages - available_pages
    pagetable.allocation_count += 1
    
    print(f"🔍 [MEMORY] 页面分配请求 #{pagetable.allocation_count}:")
    print(f"   - 需要token数: {n}")
    print(f"   - 当前页面数: {len(paged_kv_indices)}")
    print(f"   - 最后页剩余: {PAGE_SIZE - paged_kv_last_page_len} slots")
    print(f"   - 可用页面池: {available_pages} 页")
    print(f"   - 已使用页面: {used_pages} 页")
    print(f"   - 总页面数: {pagetable.initial_pages}")
    print(f"   - 使用率: {100*used_pages/pagetable.initial_pages:.1f}%")
    print(f"   - Session类型: {'已有' if is_existing_session else '新建'}")
    print(f"   - 优先级: {session_priority}")
    print(f"   - Session ID: {session_id or 'Unknown'}")
    
    # 🔥 更新session访问时间
    if session_id and hasattr(pagetable, 'update_session_access_time'):
        pagetable.update_session_access_time(session_id)
    
    # 计算是否需要新页面
    if paged_kv_last_page_len + n <= PAGE_SIZE:
        # 当前页面足够
        paged_kv_last_page_len += n
        print(f"✅ [MEMORY] 当前页足够，更新最后页长度: {paged_kv_last_page_len}")
    else:
        # 需要新页面
        num_new_pages = (n - (PAGE_SIZE - paged_kv_last_page_len) + PAGE_SIZE - 1) // PAGE_SIZE
        print(f"📈 [MEMORY] 需要新页面: {num_new_pages} 页")
        
        # 🔥 智能页面不足处理策略
        if available_pages < num_new_pages:
            # 计算页面使用率
            usage_rate = used_pages / pagetable.initial_pages
            
            print(f"❌ [MEMORY] 页面池不足：需要 {num_new_pages} 页，但只有 {available_pages} 页可用")
            print(f"🔍 [MEMORY] 页面使用统计:")
            print(f"   - 使用率: {usage_rate:.1%}")
            
            # 🔥 策略1：已有session优先保护
            if is_existing_session and usage_rate > 0.9:  # 90%以上使用率
                print(f"🛡️ [MEMORY] 已有session优先保护策略启用")
                
                # 尝试紧急回收页面
                emergency_freed = pagetable._emergency_page_reclaim(num_new_pages)
                if emergency_freed >= num_new_pages:
                    print(f"✅ [MEMORY] 紧急回收成功，释放了 {emergency_freed} 页")
                    available_pages = len(pagetable.paged_queue)
                else:
                    print(f"⚠️ [MEMORY] 紧急回收不足，已有session请求将被延迟处理")
                    raise RuntimeError(f"GPU内存页面池耗尽：已有session需要 {num_new_pages} 页但只能回收 {emergency_freed} 页")
            
            # 🔥 策略2：新session降级处理
            elif not is_existing_session:
                print(f"🔄 [MEMORY] 新session降级策略：延迟创建")
                raise RuntimeError(f"GPU内存页面池耗尽：新session创建被阻止以保护已有session（需要 {num_new_pages} 页但只有 {available_pages} 页）")
            
            # 🔥 策略3：系统过载保护
            else:
                print(f"🚨 [MEMORY] 系统内存严重不足")
                if available_pages == 0:
                    print(f"❌ [MEMORY] 页面池完全耗尽，无法分配内存")
                
                # 显示引用计数分布
                unique_counts, count_frequencies = torch.unique(pagetable.page_cnt, return_counts=True)
                print(f"📊 [MEMORY] 页面引用计数分布:")
                for count, freq in zip(unique_counts.cpu().numpy(), count_frequencies.cpu().numpy()):
                    print(f"   - 引用计数 {count}: {freq} 页")
                
                raise RuntimeError(f"GPU内存页面池耗尽：需要 {num_new_pages} 页但无可用页面")
        
        # 分配页面
        allocated_indices = []
        for _ in range(num_new_pages):
            if len(pagetable.paged_queue) == 0:
                raise RuntimeError("页面分配过程中队列意外为空")
            
            page_idx = pagetable.paged_queue.pop(0)
            allocated_indices.append(page_idx)
            pagetable.page_cnt[page_idx] += 1
        
        paged_kv_indices.extend(allocated_indices)
        
        # 🔥 新增：追踪session页面使用
        if session_id and hasattr(pagetable, 'track_session_page_usage'):
            pagetable.track_session_page_usage(session_id, allocated_indices)
        
        # 计算新的最后页长度
        paged_kv_last_page_len = (n - (PAGE_SIZE - paged_kv_last_page_len) - 1) % PAGE_SIZE + 1
        
        # 更新峰值使用统计
        current_usage = pagetable.initial_pages - len(pagetable.paged_queue)
        if current_usage > pagetable.peak_usage:
            pagetable.peak_usage = current_usage
        
        print(f"✅ [MEMORY] 页面分配成功:")
        print(f"   - 分配的页面: {allocated_indices}")
        print(f"   - 新的最后页长度: {paged_kv_last_page_len}")
        print(f"   - 当前使用: {current_usage}/{pagetable.initial_pages} 页")
        print(f"   - 峰值使用: {pagetable.peak_usage} 页")
        print(f"   - 剩余可用: {len(pagetable.paged_queue)} 页")
    
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