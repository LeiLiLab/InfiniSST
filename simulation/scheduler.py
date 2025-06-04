import time
import logging
import torch
import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from enum import Enum
from dataclasses import dataclass
import threading

logger = logging.getLogger("scheduler")

class RequestType(Enum):
    """Types of requests that can be processed"""
    PREFILL = "prefill"
    DECODE = "decode"

class RequestState(Enum):
    """States that a request can be in"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class States:
    """State container for maintaining caches"""
    speech_cache: Any = None
    past_key_values: Any = None

@dataclass
class Request:
    """A request for speech processing as expected by the engine"""
    input_ids: torch.Tensor
    speech: torch.Tensor
    blocksize: int 
    max_new_tokens: int
    max_cache_size: int
    speech_cache: Any
    max_llm_cache_size: int
    system_prompt_size: int
    past_key_values: Any
    user_id: int = -1  # Added user_id field with default value
    prefill_finished: bool = False  # Flag to track if prefill is finished
    decode_finished: bool = False  # Flag to track if decode is finished
    beam_state: Any = None  # Beam search state
    llm_cache: Any = None  # LLM cache for beam search
    results: Any = None  # Results of beam search

class UserSession:
    """Class representing a user's session"""
    
    def __init__(self, user_id: int, latency_multiplier: int = 1):
        """
        Initialize a user session
        
        Args:
            user_id: Unique identifier for this user
            latency_multiplier: Multiplier for segment timing (1, 2, or 4)
        """
        self.user_id = user_id
        self.latency_multiplier = latency_multiplier
        self.current_request_id = None  # ID of the currently processing request
        self.last_segment_time = 0.0  # Time when last segment was processed
        self.segment_interval = 0.96 * latency_multiplier  # Time between segments in seconds
        
        # State tracking
        self.is_prefill_stage = True  # True for prefill, False for decode
        self.is_processing = False  # Whether the session is currently being processed
        self.pending_segments = []  # Segments waiting to be processed
        self.generated_tokens = []  # Tokens generated so far
        
        # Engine state tracking
        self.states = States()
        # Initialize empty tensor with long dtype (for token ids)
        self.input_ids = torch.tensor([], dtype=torch.long)
        
        logger.info(f"User session {user_id} created with latency multiplier {latency_multiplier}")
    
    def can_process_next_segment(self, current_time: float) -> bool:
        """Check if it's time to process the next segment"""
        if self.is_processing:
            return False
        
        # Can process if it's been long enough since the last segment
        return current_time >= (self.last_segment_time + self.segment_interval)
    
    def mark_as_processing(self):
        """Mark the session as currently being processed"""
        self.is_processing = True
    
    def handle_prefill_completed(self, current_time: float, result_input_ids=None):
        """Handle completion of prefill stage"""
        self.is_processing = False
        self.is_prefill_stage = False  # Move to decode stage
        self.last_segment_time = current_time
        if result_input_ids is not None:
            self.input_ids = result_input_ids
    
    def handle_decode_completed(self, current_time: float, output_tokens: List[int]):
        """Handle completion of decode stage"""
        self.is_processing = False
        self.is_prefill_stage = True  # Back to prefill for next segment
        self.last_segment_time = current_time
        
        # Update input_ids with new tokens
        if isinstance(output_tokens, torch.Tensor):
            # Ensure we're working with 1D tensors for consistency
            flat_output = output_tokens.view(-1)
            
            # Initialize input_ids if empty
            if self.input_ids.numel() == 0:
                self.input_ids = flat_output
            else:
                # Ensure input_ids is also 1D
                flat_input = self.input_ids.view(-1)
                # Concatenate the tokens
                self.input_ids = torch.cat([flat_input, flat_output], dim=0)
                
            # Store generated tokens as a list
            self.generated_tokens.extend(flat_output.tolist())
        else:
            # Convert list to tensor then handle
            output_tensor = torch.tensor(output_tokens, dtype=torch.long).view(-1)
            
            # Initialize input_ids if empty
            if self.input_ids.numel() == 0:
                self.input_ids = output_tensor
            else:
                # Ensure input_ids is 1D
                flat_input = self.input_ids.view(-1)
                # Concatenate the tokens
                self.input_ids = torch.cat([flat_input, output_tensor], dim=0)
                
            # Store generated tokens
            self.generated_tokens.extend(output_tokens)

class Scheduler:
    """Scheduler for managing speech processing requests"""
    
    def __init__(self, max_batch_size: int = 4, blocksize: int = 48, max_new_tokens: int = 10,
                 max_cache_size: int = 1024, max_llm_cache_size: int = 2048,
                 system_prompt_size: int = 64, pseudo_batch_size: int = 1):
        """
        Initialize the scheduler
        
        Args:
            max_batch_size: Maximum number of requests in a batch
            blocksize: Default blocksize for requests
            max_new_tokens: Default max_new_tokens for requests
            max_cache_size: Maximum cache size for speech processing
            max_llm_cache_size: Maximum cache size for LLM
            system_prompt_size: Size of the system prompt
            pseudo_batch_size: Number of duplicate requests for pseudo-batching
        """
        self.max_batch_size = max_batch_size
        self.blocksize = blocksize
        self.max_new_tokens = max_new_tokens
        self.max_cache_size = max_cache_size
        self.max_llm_cache_size = max_llm_cache_size
        self.system_prompt_size = system_prompt_size
        self.pseudo_batch_size = pseudo_batch_size
        
        # Session management
        self.active_sessions = {}  # user_id -> UserSession
        
        # Request management
        self.prefill_queue = []  # List of prefill sessions (user_ids)
        self.decode_queue = []  # List of decode sessions (user_ids)
        
        # Request counter for tracking
        self.next_request_count = 0
        
        # Locks for thread safety
        self.prefill_lock = threading.Lock()
        self.decode_lock = threading.Lock()
        self.session_lock = threading.Lock()
        
        # Stats
        self.total_prefill_requests = 0
        self.total_decode_requests = 0
        self.total_output_tokens = 0
        
        # For engine compatibility
        self.current_batch_user_ids = []
        
        logger.info(f"Scheduler initialized with max batch size {max_batch_size}")
    
    def submit_request(self, user_id: int, speech_segment: np.ndarray):
        """
        Submit a speech segment for processing
        
        Args:
            user_id: User ID associated with the request
            speech_segment: Speech data (numpy array)
        """
        with self.session_lock:
            # Create user session if it doesn't exist
            if user_id not in self.active_sessions:
                # For simplicity, assign a random latency multiplier (1, 2, or 4)
                latency_multiplier = np.random.choice([1, 2, 4])
                self.active_sessions[user_id] = UserSession(user_id, latency_multiplier)
            
            session = self.active_sessions[user_id]
            
            # Convert speech segment to torch tensor and store in session
            speech_tensor = torch.tensor(speech_segment).float()
            session.pending_segments.append(speech_tensor)
            
            # Add to prefill queue if not already processing
            if not session.is_processing:
                with self.prefill_lock:
                    if user_id not in self.prefill_queue:
                        self.prefill_queue.append(user_id)
                        self.total_prefill_requests += 1
                        logger.debug(f"Added user {user_id} to prefill queue")
    
    def get_request_batch(self, batch_size: Optional[int] = None) -> Tuple[RequestType, List]:
        """
        Get a batch of requests for processing
        
        Args:
            batch_size: Maximum number of requests to include in the batch
            
        Returns:
            Tuple containing the request type and list of user_ids
        """
        if batch_size is None:
            batch_size = self.max_batch_size
        
        # Prioritize decode requests over prefill
        with self.decode_lock:
            if len(self.decode_queue) > 0:
                batch_user_ids = self.decode_queue[:batch_size]
                self.decode_queue = self.decode_queue[batch_size:]
                
                # Prepare the requests
                request_list = self._prepare_request_list(batch_user_ids, RequestType.DECODE)
                
                # Store the user IDs for the current batch
                self.current_batch_user_ids = batch_user_ids
                
                # Mark the sessions as processing
                with self.session_lock:
                    for user_id in batch_user_ids:
                        if user_id in self.active_sessions:
                            session = self.active_sessions[user_id]
                            session.mark_as_processing()
                
                return RequestType.DECODE, request_list
        
        # If no decode requests, check prefill
        with self.prefill_lock:
            if len(self.prefill_queue) > 0:
                batch_user_ids = self.prefill_queue[:batch_size]
                self.prefill_queue = self.prefill_queue[batch_size:]
                
                # Prepare the requests
                request_list = self._prepare_request_list(batch_user_ids, RequestType.PREFILL)
                
                # Store the user IDs for the current batch
                self.current_batch_user_ids = batch_user_ids
                
                # Mark the sessions as processing
                with self.session_lock:
                    for user_id in batch_user_ids:
                        if user_id in self.active_sessions:
                            session = self.active_sessions[user_id]
                            session.mark_as_processing()
                
                return RequestType.PREFILL, request_list
        
        # No requests in either queue
        return None, []
    
    def _prepare_request_list(self, user_ids: List[int], request_type: RequestType) -> List[Request]:
        """
        Prepare a list of Request objects in the format expected by the engine
        
        Args:
            user_ids: List of user IDs to prepare requests for
            request_type: Type of requests to prepare
            
        Returns:
            List of Request objects
        """
        requests = []
        
        for user_id in user_ids:
            if user_id not in self.active_sessions:
                continue
                
            session = self.active_sessions[user_id]
            
            # Ensure input_ids is a valid tensor with the right dtype
            if session.input_ids is None or session.input_ids.numel() == 0:
                session.input_ids = torch.tensor([], dtype=torch.long)
            
            if request_type == RequestType.PREFILL:
                # For prefill, use the first pending segment
                if not session.pending_segments:
                    continue
                    
                speech_batch = session.pending_segments.pop(0)
                
                # Create request for each item in pseudo batch
                for _ in range(self.pseudo_batch_size):
                    request = Request(
                        input_ids=session.input_ids.view(-1),
                        speech=speech_batch.view(-1),
                        blocksize=self.blocksize * session.latency_multiplier,
                        max_new_tokens=self.max_new_tokens,
                        
                        max_cache_size=self.max_cache_size,
                        speech_cache=session.states.speech_cache,

                        max_llm_cache_size=self.max_llm_cache_size,
                        system_prompt_size=self.system_prompt_size,
                        past_key_values=session.states.past_key_values,
                        user_id=user_id  # Set the user_id field
                    )
                    requests.append(request)
                
            elif request_type == RequestType.DECODE:
                # For decode, use the existing input_ids
                for _ in range(self.pseudo_batch_size):
                    request = Request(
                        input_ids=session.input_ids.view(-1),
                        speech=torch.tensor([], dtype=torch.float),  # Empty for decode
                        blocksize=self.blocksize * session.latency_multiplier,
                        max_new_tokens=self.max_new_tokens,
                        
                        max_cache_size=self.max_cache_size,
                        speech_cache=session.states.speech_cache,

                        max_llm_cache_size=self.max_llm_cache_size,
                        system_prompt_size=self.system_prompt_size,
                        past_key_values=session.states.past_key_values,
                        user_id=user_id  # Set the user_id field
                    )
                    requests.append(request)
        
        return requests
    
    def handle_prefill_result(self, user_id: int, continue_decode: bool, result=None, speech_cache=None):
        """
        Handle the result of prefill processing
        
        Args:
            user_id: User ID
            continue_decode: Whether to continue to decode stage
            result: The resulting tokens from prefill
            speech_cache: Updated speech cache
        """
        with self.session_lock:
            if user_id in self.active_sessions:
                session = self.active_sessions[user_id]
                
                # Update session state
                if speech_cache is not None:
                    session.states.speech_cache = speech_cache
                
                if continue_decode:
                    # Update input_ids if result is provided
                    if result is not None:
                        session.handle_prefill_completed(time.time(), result)
                    else:
                        session.handle_prefill_completed(time.time())
                    
                    # Add to decode queue
                    with self.decode_lock:
                        if user_id not in self.decode_queue:
                            self.decode_queue.append(user_id)
                            self.total_decode_requests += 1
                    
                    logger.debug(f"Prefill completed for user {user_id}, added to decode queue")
                else:
                    # Reset session state
                    session.is_processing = False
                    logger.debug(f"Prefill completed for user {user_id}, no decode needed")
    
    def handle_decode_result(self, user_id: int, output_tokens, continue_decode: bool, past_key_values=None):
        """
        Handle the result of decode processing
        
        Args:
            user_id: User ID
            output_tokens: Tokens generated in this decode step
            continue_decode: Whether to continue decoding (more tokens needed)
            past_key_values: Updated past key values
        """
        with self.session_lock:
            if user_id in self.active_sessions:
                session = self.active_sessions[user_id]
                
                # Update past key values if provided
                if past_key_values is not None:
                    session.states.past_key_values = past_key_values
                
                # Handle generated tokens
                session.handle_decode_completed(time.time(), output_tokens)
                self.total_output_tokens += len(output_tokens)
                
                if continue_decode:
                    # Need more decode steps - add back to decode queue
                    with self.decode_lock:
                        if user_id not in self.decode_queue:
                            self.decode_queue.append(user_id)
                    
                    logger.debug(f"Decode step completed for user {user_id}, need more decode steps")
                else:
                    # Decode complete - check if there are more segments
                    if session.pending_segments:
                        # More speech segments to process
                        with self.prefill_lock:
                            if user_id not in self.prefill_queue:
                                self.prefill_queue.append(user_id)
                        
                        logger.debug(f"Decode completed for user {user_id}, has more segments")
                    else:
                        # No more segments, session is idle
                        session.is_processing = False
                        logger.debug(f"Decode completed for user {user_id}, no more segments")
    
    def get_stats(self) -> Dict:
        """Get statistics about the scheduler state"""
        return {
            "active_sessions": len(self.active_sessions),
            "prefill_queue": len(self.prefill_queue),
            "decode_queue": len(self.decode_queue),
            "total_prefill_requests": self.total_prefill_requests,
            "total_decode_requests": self.total_decode_requests,
            "total_output_tokens": self.total_output_tokens
        } 