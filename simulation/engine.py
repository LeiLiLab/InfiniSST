import time
import logging
import torch
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import random
import threading
from dataclasses import dataclass, field
from datetime import datetime
from queue import Queue
from concurrent.futures import ThreadPoolExecutor

from scheduler import Request, RequestType

logger = logging.getLogger("engine")

# Configure event logging for engine events
engine_event_logger = logging.getLogger("engine_events")
engine_event_logger.setLevel(logging.INFO)
engine_event_handler = logging.FileHandler('engine_events.txt')
engine_event_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
engine_event_logger.addHandler(engine_event_handler)

@dataclass
class SpeechCache:
    """Simplified speech cache for simulation"""
    indices: List[int] = field(default_factory=list)
    last_page_len: int = 0

@dataclass
class LLMCache:
    """Simplified LLM cache for simulation"""
    indices: List[int] = field(default_factory=list)
    last_page_len: int = 0

@dataclass
class BeamState:
    """Simplified beam search state for simulation"""
    num_beams: int = 4
    num_remaining_beams: int = 4
    sum_logps: Optional[torch.Tensor] = None
    generated_ids: Optional[torch.Tensor] = None
    results: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class BatchTask:
    """Task for batch processing"""
    request_type: RequestType
    batch: List[Request]
    results_callback: Any = None

class Engine:
    """Engine for processing speech and language model requests with beam search"""
    
    def __init__(
        self,
        prefill_time: float = 0.1,  # Simulated time for prefill processing
        decode_time: float = 0.05,  # Simulated time for decode step
        tokens_per_step: int = 1,  # Tokens generated per decode step
        max_steps: int = 5,  # Max decode steps before forcing completion
        beam_size: int = 4,  # Beam size for beam search
        vocab_size: int = 30000,  # Vocabulary size for random token generation
        eos_token_id: int = 2,  # End of sequence token ID
        realtime: bool = True,  # Whether to run in real time
        pseudo_batch_size: int = 1,  # Number of duplicate requests per batch
        num_prefill_workers: int = 1,  # Number of prefill worker threads
        num_decode_workers: int = 1,  # Number of decode worker threads
        num_workers: int = None  # Total number of worker threads to distribute
    ):
        """
        Initialize the engine
        
        Args:
            prefill_time: Simulated time for prefill processing (seconds)
            decode_time: Simulated time for a decode step (seconds)
            tokens_per_step: Tokens generated per decode step
            max_steps: Maximum decode steps before forcing completion
            beam_size: Beam size for beam search
            vocab_size: Vocabulary size for random token generation
            eos_token_id: End of sequence token ID
            realtime: Whether to run in real time
            pseudo_batch_size: Number of duplicate requests per batch
            num_prefill_workers: Number of worker threads for prefill processing
            num_decode_workers: Number of worker threads for decode processing
            num_workers: If provided, total number of worker threads to distribute
        """
        self.prefill_time = prefill_time
        self.decode_time = decode_time
        self.tokens_per_step = tokens_per_step
        self.max_steps = max_steps
        self.beam_size = beam_size
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id
        self.realtime = realtime
        self.pseudo_batch_size = pseudo_batch_size
        
        # Handle worker distribution if num_workers is provided
        if num_workers is not None:
            # Distribute workers with more for decode (typically more decode steps than prefill)
            total_workers = num_workers
            self.num_prefill_workers = max(1, total_workers // 3)
            self.num_decode_workers = max(1, total_workers - self.num_prefill_workers)
        else:
            self.num_prefill_workers = num_prefill_workers
            self.num_decode_workers = num_decode_workers
        
        # Setup page tables for KV caches
        self.speech_pagetable = self._init_pagetable()
        self.llm_prefill_pagetable = self._init_pagetable()
        self.llm_decode_pagetable = self._init_pagetable()
        
        # Stats
        self.total_prefill_requests = 0
        self.total_decode_requests = 0
        self.total_decode_steps = 0
        self.total_tokens_generated = 0
        
        # Set up worker queues and threads
        self.prefill_queue = Queue()
        self.decode_queue = Queue()
        self.running = True
        
        # Start worker threads
        self.prefill_workers = []
        self.decode_workers = []
        
        # Start prefill workers
        for i in range(self.num_prefill_workers):
            worker_thread = threading.Thread(target=self._prefill_worker_thread, args=(i,))
            worker_thread.daemon = True
            worker_thread.start()
            self.prefill_workers.append(worker_thread)
        
        # Start decode workers
        for i in range(self.num_decode_workers):
            worker_thread = threading.Thread(target=self._decode_worker_thread, args=(i,))
            worker_thread.daemon = True
            worker_thread.start()
            self.decode_workers.append(worker_thread)
        
        engine_event_logger.info(f"ENGINE_INITIALIZED: realtime={realtime}, prefill_time={prefill_time}s, " +
                                f"decode_time={decode_time}s, tokens_per_step={tokens_per_step}, " +
                                f"beam_size={beam_size}, prefill_workers={self.num_prefill_workers}, " +
                                f"decode_workers={self.num_decode_workers}")
        
        logger.info(
            f"Engine initialized with: prefill_time={prefill_time}s, decode_time={decode_time}s, "
            f"tokens_per_step={tokens_per_step}, max_steps={max_steps}, beam_size={beam_size}, "
            f"realtime={realtime}, pseudo_batch_size={pseudo_batch_size}, "
            f"prefill_workers={self.num_prefill_workers}, decode_workers={self.num_decode_workers}"
        )
    
    def _init_pagetable(self):
        """Initialize a page table for KV cache"""
        pagetable = {"paged_kv_cache": {"device": "cpu"}, "paged_queue": []}
        return pagetable
    
    def _prefill_worker_thread(self, worker_id):
        """Worker thread dedicated to processing prefill batches"""
        logger.info(f"Prefill worker {worker_id} started")
        
        while self.running:
            try:
                if not self.prefill_queue.empty():
                    task = self.prefill_queue.get(block=False)
                    logger.debug(f"Prefill worker {worker_id} processing batch of size {len(task.batch)}")
                    results = self._process_prefill_batch(task.batch)
                    if task.results_callback:
                        task.results_callback(results)
                    self.prefill_queue.task_done()
                else:
                    # If queue is empty, sleep a bit
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Prefill worker {worker_id} encountered an error: {e}", exc_info=True)
        
        logger.info(f"Prefill worker {worker_id} stopped")
    
    def _decode_worker_thread(self, worker_id):
        """Worker thread dedicated to processing decode batches"""
        logger.info(f"Decode worker {worker_id} started")
        
        while self.running:
            try:
                if not self.decode_queue.empty():
                    task = self.decode_queue.get(block=False)
                    logger.debug(f"Decode worker {worker_id} processing batch of size {len(task.batch)}")
                    results = self._process_decode_batch(task.batch)
                    if task.results_callback:
                        task.results_callback(results)
                    self.decode_queue.task_done()
                else:
                    # If queue is empty, sleep a bit
                    time.sleep(0.01)
            except Exception as e:
                logger.error(f"Decode worker {worker_id} encountered an error: {e}", exc_info=True)
        
        logger.info(f"Decode worker {worker_id} stopped")
    
    def shutdown(self):
        """Shut down the engine and workers"""
        self.running = False
        
        # Wait for workers to finish
        for worker in self.prefill_workers + self.decode_workers:
            worker.join(timeout=1.0)
        
        logger.info("Engine shut down")
    
    def run_batch(self, request_type: RequestType, batch: List[Request], callback=None) -> List[Dict]:
        """
        Process a batch of requests by adding to appropriate queue
        
        Args:
            request_type: Type of requests in the batch
            batch: List of Request objects
            callback: Optional callback function to call with results
            
        Returns:
            Empty list (actual results will be delivered via callback)
        """
        if not batch:
            if callback:
                callback([])
            return []
        
        # Ensure all requests have user_id set
        for request in batch:
            if not hasattr(request, 'user_id'):
                request.user_id = -1  # Set a default user_id if not present
        
        # Create task for batch processing
        task = BatchTask(request_type=request_type, batch=batch, results_callback=callback)
        
        # Add task to appropriate queue
        if request_type == RequestType.PREFILL:
            self.prefill_queue.put(task)
            logger.debug(f"Added prefill batch of size {len(batch)} to queue")
        elif request_type == RequestType.DECODE:
            self.decode_queue.put(task)
            logger.debug(f"Added decode batch of size {len(batch)} to queue")
        else:
            logger.error(f"Unknown request type: {request_type}")
            if callback:
                callback([])
        
        return []
    
    def run_batch_sync(self, request_type: RequestType, batch: List[Request]) -> List[Dict]:
        """
        Process a batch of requests synchronously (for backward compatibility)
        
        Args:
            request_type: Type of requests in the batch
            batch: List of Request objects
            
        Returns:
            List of result dictionaries for each request
        """
        if not batch:
            return []
        
        # Ensure all requests have user_id set
        for request in batch:
            if not hasattr(request, 'user_id'):
                request.user_id = -1  # Set a default user_id if not present
        
        if request_type == RequestType.PREFILL:
            return self._process_prefill_batch(batch)
        elif request_type == RequestType.DECODE:
            return self._process_decode_batch(batch)
        else:
            logger.error(f"Unknown request type: {request_type}")
            return []
    
    def _process_prefill_batch(self, batch: List[Request]) -> List[Dict]:
        """Process a batch of prefill requests using beam search"""
        unique_batch_size = len(set(r.user_id for r in batch if hasattr(r, 'user_id')))
        self.total_prefill_requests += unique_batch_size
        batch_id = self.total_prefill_requests // max(1, unique_batch_size)
        
        start_time = time.time()
        
        engine_event_logger.info(f"PREFILL_BATCH_START: id={batch_id}, size={unique_batch_size}, " +
                               f"expected_duration={self.prefill_time}s")
        
        logger.info(f"Processing {unique_batch_size} prefill requests (batch #{batch_id})")
        
        # Simulate processing time in real time
        if self.realtime:
            # Actually sleep for the simulated processing time
            time.sleep(self.prefill_time)
        
        # Apply beam search prefill to each request
        for request in batch:
            # Ensure user_id is set
            if not hasattr(request, 'user_id'):
                request.user_id = -1
            
            # Mark request as not prefill finished before processing
            request.prefill_finished = False
        
        # Run beam search prefill
        processed_requests, self.speech_pagetable, self.llm_prefill_pagetable, self.llm_decode_pagetable = self._beam_search_prefill(batch)
        
        results = []
        
        # Process each request result
        for request in processed_requests:
            # For simplicity, always continue to decode
            continue_decode = True
            
            # Ensure result token ids are 1D tensors for consistent handling
            if hasattr(request.beam_state, 'generated_ids') and request.beam_state.generated_ids is not None:
                # Get a flattened version for consistent format
                result_tokens = request.beam_state.generated_ids.view(-1)
            else:
                # Default empty result if no generated ids
                result_tokens = torch.tensor([], dtype=torch.long)
                
            # Calculate actual processing time
            processing_time = time.time() - start_time
            
            results.append({
                "user_idx": batch.index(request),
                "user_id": request.user_id,
                "continue_decode": continue_decode,
                "result": result_tokens,  # Ensure result is a 1D tensor
                "speech_cache": request.speech_cache,
                "processing_time": processing_time
            })
            
            engine_event_logger.info(f"PREFILL_REQUEST_COMPLETED: user_id={request.user_id}, " +
                                  f"continue_decode={continue_decode}")
            
            logger.debug(f"Prefill completed for user {request.user_id}")
        
        batch_processing_time = time.time() - start_time
        
        engine_event_logger.info(f"PREFILL_BATCH_COMPLETED: id={batch_id}, size={unique_batch_size}, " +
                               f"actual_duration={batch_processing_time:.3f}s")
        
        logger.info(f"Processed {unique_batch_size} prefill requests in {batch_processing_time:.3f}s")
        return results
    
    def _process_decode_batch(self, batch: List[Request]) -> List[Dict]:
        """Process a batch of decode requests using beam search"""
        unique_batch_size = len(set(r.user_id for r in batch if hasattr(r, 'user_id')))
        self.total_decode_requests += unique_batch_size
        self.total_decode_steps += unique_batch_size
        batch_id = self.total_decode_requests // max(1, unique_batch_size)
        
        start_time = time.time()
        
        engine_event_logger.info(f"DECODE_BATCH_START: id={batch_id}, size={unique_batch_size}, " +
                               f"expected_duration={self.decode_time}s")
        
        logger.info(f"Processing {unique_batch_size} decode requests (batch #{batch_id})")
        
        # Simulate processing time in real time
        if self.realtime:
            # Actually sleep for the simulated processing time
            time.sleep(self.decode_time)
        
        # Apply beam search decode to each request
        for request in batch:
            # Ensure user_id is set
            if not hasattr(request, 'user_id'):
                request.user_id = -1
                
            # Mark request as prefill finished before processing
            request.prefill_finished = True
        
        # Run beam search decode
        processed_requests, self.speech_pagetable, self.llm_prefill_pagetable, self.llm_decode_pagetable = self._beam_search_decode(batch)
        
        results = []
        
        # Process each request result
        for request in processed_requests:
            # Generate output tokens
            if hasattr(request, 'decode_finished') and request.decode_finished:
                # Get the best beam result - ensure it's 1D
                output_tokens = torch.tensor(request.results["sequence"][-self.tokens_per_step:]).view(-1)
                continue_decode = False
            else:
                # Get the latest generated tokens from the first beam - ensure it's 1D
                output_tokens = torch.tensor([request.beam_state.generated_ids[0][-1].item()]).view(-1)
                continue_decode = True
            
            # Validate output_tokens is 1D
            if len(output_tokens.shape) > 1:
                output_tokens = output_tokens.view(-1)
                
            self.total_tokens_generated += len(output_tokens)
            
            # Calculate actual processing time
            processing_time = time.time() - start_time
            
            results.append({
                "user_idx": batch.index(request),
                "user_id": request.user_id,
                "continue_decode": continue_decode,
                "output_tokens": output_tokens,
                "past_key_values": request.llm_cache,
                "processing_time": processing_time
            })
            
            tokens_str = ", ".join(str(t.item()) for t in output_tokens)
            engine_event_logger.info(f"DECODE_REQUEST_COMPLETED: user_id={request.user_id}, " +
                                  f"tokens=[{tokens_str}], continue_decode={continue_decode}")
            
            logger.debug(
                f"Decode step for user {request.user_id}: "
                f"{'continue' if continue_decode else 'complete'} with {len(output_tokens)} tokens"
            )
        
        batch_processing_time = time.time() - start_time
        total_tokens = sum(len(r["output_tokens"]) for r in results)
        
        engine_event_logger.info(f"DECODE_BATCH_COMPLETED: id={batch_id}, size={unique_batch_size}, " +
                               f"tokens_generated={total_tokens}, " +
                               f"actual_duration={batch_processing_time:.3f}s")
        
        logger.info(
            f"Processed {unique_batch_size} decode requests in {batch_processing_time:.3f}s, "
            f"generated {total_tokens} tokens"
        )
        return results
    
    def _beam_search_prefill(self, requests: List[Request]):
        """Simulate prefill processing for beam search"""
        # Simulate speech encoding
        for request in requests:
            request.speech_cache = SpeechCache(
                indices=list(range(10)),  # Dummy indices
                last_page_len=request.blocksize
            )
        
        # Simulate LLM processing with random logits
        for request in requests:
            # Initialize beam search state
            request.beam_state = BeamState(num_beams=self.beam_size)
            
            # Generate random logits and get top-k
            logits = torch.randn(1, self.vocab_size)
            logps = torch.log_softmax(logits, dim=-1)
            topk_logps, topk_indices = torch.topk(logps, self.beam_size, dim=-1)
            
            request.beam_state.sum_logps = topk_logps.view(-1)
            request.beam_state.generated_ids = topk_indices.view(-1, 1)
            
            # Create LLM cache
            main_cache = LLMCache(
                indices=list(range(5)),
                last_page_len=len(request.input_ids)
            )
            
            # Create beam caches
            beam_caches = [main_cache]
            for _ in range(1, self.beam_size):
                cache_i = LLMCache(
                    indices=list(range(5)),
                    last_page_len=len(request.input_ids)
                )
                beam_caches.append(cache_i)
            
            request.llm_cache = beam_caches
            request.prefill_finished = True

        return requests, self.speech_pagetable, self.llm_prefill_pagetable, self.llm_decode_pagetable
    
    def _collect_finished_beams(self, request):
        """Collect finished beams from request"""
        remaining_llm_cache = []
        mask = [True] * request.beam_state.num_remaining_beams
        
        for j in range(request.beam_state.num_remaining_beams):
            gen_j = request.beam_state.generated_ids[j]
            # Randomly decide if the beam is finished
            if random.random() < 0.2 or gen_j[-1] == self.eos_token_id or len(gen_j) >= request.max_new_tokens:
                request.beam_state.num_remaining_beams -= 1
                mask[j] = False
                request.beam_state.results.append({
                    "sequence": gen_j.tolist(),
                    "logp": request.beam_state.sum_logps[j] / len(gen_j),
                    "cache": request.llm_cache[j]
                })
            else:
                remaining_llm_cache.append(request.llm_cache[j])
        
        if request.beam_state.num_remaining_beams > 0:
            request.llm_cache = remaining_llm_cache
            # Note: In a real implementation, we would properly handle the masking
            # Here we just keep track of which beams are still active
            if any(mask):
                mask_tensor = torch.tensor(mask)
                request.beam_state.sum_logps = request.beam_state.sum_logps[mask_tensor]
                request.beam_state.generated_ids = request.beam_state.generated_ids[mask_tensor]
    
    def _finish_beam_search(self, request):
        """Finish beam search when all beams are done"""
        assert request.beam_state.num_remaining_beams == 0
        
        # Sort results by log probability and keep the best one
        if request.beam_state.results:
            results = sorted(request.beam_state.results, key=lambda x: x["logp"], reverse=True)
            request.results = results[0]
            request.llm_cache = results[0]['cache']
        else:
            # Handle case where no beams finished naturally (all forced to end)
            default_result = {
                "sequence": request.beam_state.generated_ids[0].tolist(),
                "logp": request.beam_state.sum_logps[0] / len(request.beam_state.generated_ids[0]),
                "cache": request.llm_cache[0] if request.llm_cache else None
            }
            request.results = default_result
            request.llm_cache = default_result['cache']
        
        # Mark request as finished
        request.decode_finished = True
    
    def _beam_search_decode(self, requests: List[Request]):
        """Simulate decode processing for beam search"""
        # Collect finished beams
        remaining_requests = []
        completed_requests = []
        
        for request in requests:
            if not hasattr(request, 'beam_state') or request.beam_state is None:
                # Initialize beam state if not present
                request.beam_state = BeamState(num_beams=self.beam_size)
                
                # Generate random logits and get top-k
                logits = torch.randn(1, self.vocab_size)
                logps = torch.log_softmax(logits, dim=-1)
                topk_logps, topk_indices = torch.topk(logps, self.beam_size, dim=-1)
                
                request.beam_state.sum_logps = topk_logps.view(-1)
                request.beam_state.generated_ids = topk_indices.view(-1, 1)
                
                # Create LLM cache
                main_cache = LLMCache(
                    indices=list(range(5)),
                    last_page_len=len(request.input_ids)
                )
                
                # Create beam caches
                beam_caches = [main_cache]
                for _ in range(1, self.beam_size):
                    cache_i = LLMCache(
                        indices=list(range(5)),
                        last_page_len=len(request.input_ids)
                    )
                    beam_caches.append(cache_i)
                
                request.llm_cache = beam_caches
            
            self._collect_finished_beams(request)
            if request.beam_state.num_remaining_beams > 0:
                remaining_requests.append(request)
            else:
                self._finish_beam_search(request)
                completed_requests.append(request)
        
        # Process remaining requests
        for request in remaining_requests:
            # Generate random logits for each beam
            num_beams = request.beam_state.num_remaining_beams
            
            logp = torch.randn(num_beams, self.vocab_size)
            logp = torch.log_softmax(logp, dim=-1)
            topk_logp, topk_indices = torch.topk(logp, num_beams, dim=-1)
            
            # Create new generated IDs by appending new tokens
            new_generated_ids = []
            for j in range(num_beams):
                prev_ids = request.beam_state.generated_ids[j]
                # Pick a random token as the next one
                new_id = topk_indices[j, random.randint(0, num_beams-1)]
                new_generated_ids.append(torch.cat([prev_ids, new_id.unsqueeze(0)], dim=0))
            
            # Update the generated IDs
            request.beam_state.generated_ids = torch.stack(new_generated_ids)
            
            # Check if any beams finished with this new token
            self._collect_finished_beams(request)
            if request.beam_state.num_remaining_beams == 0:
                self._finish_beam_search(request)
                completed_requests.append(request)
        
        # Return all processed requests
        return requests, self.speech_pagetable, self.llm_prefill_pagetable, self.llm_decode_pagetable
    
    def get_stats(self) -> Dict:
        """Get engine statistics"""
        return {
            "total_prefill_requests": self.total_prefill_requests,
            "total_decode_requests": self.total_decode_requests,
            "total_decode_steps": self.total_decode_steps,
            "total_tokens_generated": self.total_tokens_generated,
            "tokens_per_second": self.total_tokens_generated / max(1, self.total_decode_steps * self.decode_time),
            "prefill_queue_size": self.prefill_queue.qsize(),
            "decode_queue_size": self.decode_queue.qsize()
        } 