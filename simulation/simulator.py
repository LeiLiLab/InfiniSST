import time
import random
import numpy as np
import soundfile as sf
import logging
import threading
from typing import Callable, Dict, List
import os
from datetime import datetime

# Configure a special event logger for detailed timing
event_logger = logging.getLogger("event_log")
event_logger.setLevel(logging.INFO)
event_handler = logging.FileHandler('event_log.txt')
event_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
event_logger.addHandler(event_handler)

# Regular logger
logger = logging.getLogger("simulator")

class UserSession:
    """Represents a user's session in the simulation"""
    
    def __init__(
        self,
        user_id: int,
        wav_path: str,
        start_time: float,
        duration: float,
        latency_multiplier: int,
        submit_fn: Callable,
        realtime: bool = True
    ):
        """
        Initialize a user session
        
        Args:
            user_id: Unique identifier for this user
            wav_path: Path to the WAV file containing speech data
            start_time: Simulation time when this session starts
            duration: Duration of the session in seconds
            latency_multiplier: Multiplier for segment timing (1, 2, or 4)
            submit_fn: Function to call to submit speech segments
            realtime: Whether running in realtime mode
        """
        self.user_id = user_id
        self.wav_path = wav_path
        self.start_time = start_time
        self.end_time = start_time + duration
        self.latency_multiplier = latency_multiplier
        self.submit_fn = submit_fn
        self.realtime = realtime
        
        # Load audio data
        self.audio_data, self.sample_rate = sf.read(wav_path)
        self.current_position = 0
        
        # Segment timing (in real seconds)
        self.segment_interval = 0.96 * latency_multiplier  # in seconds
        
        # Set the first segment time
        if realtime:
            # For realtime mode, use wall clock time
            self.next_segment_time = time.time() + self.segment_interval
        else:
            # For non-realtime mode, use simulation time
            self.next_segment_time = start_time + self.segment_interval
            
        self.segment_count = 0
        
        event_logger.info(f"USER_CREATED: user_id={user_id}, latency_multiplier={latency_multiplier}, " +
                         f"first_segment_at={datetime.fromtimestamp(self.next_segment_time).strftime('%H:%M:%S.%f')[:-3]}, " +
                         f"segment_interval={self.segment_interval}s")
        
        logger.info(
            f"Created user {user_id}: start={start_time:.2f}s, duration={duration:.2f}s, "
            f"multiplier={latency_multiplier}, wav={os.path.basename(wav_path)}, "
            f"audio length={len(self.audio_data)/self.sample_rate:.2f}s, "
            f"segment_interval={self.segment_interval}s"
        )
    
    def time_to_next_segment(self) -> float:
        """Get the time until the next segment should be submitted (in seconds)"""
        now = time.time()
        if now >= self.next_segment_time:
            return 0.0
        else:
            return self.next_segment_time - now
    
    def get_next_segment_time(self) -> float:
        """Get the absolute time when the next segment should be submitted (in seconds)"""
        return self.next_segment_time
    
    def get_next_segment(self, is_realtime=True, current_sim_time=None) -> np.ndarray:
        """
        Get the next segment of speech data
        
        Args:
            is_realtime: Whether running in realtime mode
            current_sim_time: Current simulation time (for non-realtime mode)
        
        Returns:
            Speech segment as numpy array
        """
        if self.current_position >= len(self.audio_data):
            return None
        
        # Calculate segment size based on latency multiplier
        # 960ms = 16000 * 0.96 = 15360 samples for multiplier=1
        segment_size = int(self.sample_rate * 0.96 * self.latency_multiplier)
        
        end_pos = min(self.current_position + segment_size, len(self.audio_data))
        segment = self.audio_data[self.current_position:end_pos]
        self.current_position = end_pos
        self.segment_count += 1
        
        # Update the next segment time
        if is_realtime:
            # For realtime simulation, use wall clock time
            self.next_segment_time = time.time() + self.segment_interval
        else:
            # For non-realtime simulation, use simulation time
            if current_sim_time is not None:
                self.next_segment_time = current_sim_time + self.segment_interval
            else:
                self.next_segment_time += self.segment_interval
        
        event_logger.info(f"SEGMENT_RETRIEVED: user_id={self.user_id}, segment_num={self.segment_count}, " +
                         f"next_segment_at={datetime.fromtimestamp(self.next_segment_time).strftime('%H:%M:%S.%f')[:-3]}")
        
        return segment
    
    def is_finished(self) -> bool:
        """Check if the session has finished (all data processed)"""
        return self.current_position >= len(self.audio_data)
    
    def has_expired(self, current_time: float) -> bool:
        """Check if the session has expired (time exceeded)"""
        return current_time >= self.end_time

class Simulator:
    """Simulates users arriving and submitting speech segments"""
    
    def __init__(
        self,
        rate_lambda: float,
        session_duration_mean: float,
        session_duration_stddev: float,
        wav_paths: List[str],
        submit_fn: Callable,
        max_concurrent_users: int = 32,
        initial_users: int = 0,  # Number of users to create at the start
        realtime: bool = True    # Whether to run in real time
    ):
        """
        Initialize the simulator
        
        Args:
            rate_lambda: Rate parameter for Poisson process (arrivals per second)
            session_duration_mean: Mean session duration in seconds
            session_duration_stddev: Standard deviation of session duration
            wav_paths: List of paths to WAV files for speech data
            submit_fn: Function to call to submit speech segments
            max_concurrent_users: Maximum number of concurrent users
            initial_users: Number of users to create at the start of simulation
            realtime: Whether to run in real time (True) or simulated time (False)
        """
        self.rate_lambda = rate_lambda
        self.session_duration_mean = session_duration_mean
        self.session_duration_stddev = session_duration_stddev
        self.wav_paths = wav_paths
        self.submit_fn = submit_fn
        self.max_concurrent_users = max_concurrent_users
        self.initial_users = initial_users
        self.realtime = realtime
        
        self.next_user_id = 0
        self.active_sessions = {}  # user_id -> UserSession
        self.running = False
        self.simulation_thread = None
        self.current_time = 0.0
        
        # For realtime simulation
        self.simulation_start_time = 0.0
        
        event_logger.info(f"SIMULATOR_INITIALIZED: realtime={realtime}, rate={rate_lambda}, " +
                         f"max_users={max_concurrent_users}, initial_users={initial_users}")
        
        logger.info(
            f"Simulator initialized: rate={rate_lambda} users/sec, "
            f"duration={session_duration_mean}±{session_duration_stddev}s, "
            f"max_users={max_concurrent_users}, initial_users={initial_users}, "
            f"wav_files={len(wav_paths)}, realtime={realtime}"
        )
    
    def start(self):
        """Start the simulation in a background thread"""
        if self.running:
            return
        
        self.running = True
        self.simulation_start_time = time.time()
        
        event_logger.info(f"SIMULATION_STARTED: time={datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        
        # Create initial users immediately before starting the thread
        if self.initial_users > 0:
            logger.info(f"Creating {self.initial_users} initial users...")
            for _ in range(min(self.initial_users, self.max_concurrent_users)):
                self._create_new_user()
        
        self.simulation_thread = threading.Thread(target=self._run_simulation)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()
        logger.info("Simulation started")
    
    def stop(self):
        """Stop the simulation"""
        self.running = False
        if self.simulation_thread:
            self.simulation_thread.join(timeout=1.0)
        
        event_logger.info(f"SIMULATION_STOPPED: time={datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
        logger.info("Simulation stopped")
    
    def _run_simulation(self):
        """Main simulation loop - runs in background thread"""
        # Schedule first user arrival
        next_arrival_time = self._get_next_arrival_time()
        next_arrival_wall_time = time.time() + next_arrival_time if self.realtime else 0
        
        event_logger.info(f"NEXT_USER_ARRIVAL: time={datetime.fromtimestamp(next_arrival_wall_time).strftime('%H:%M:%S.%f')[:-3]}")
        
        while self.running:
            # For realtime mode, use actual wall clock time
            if self.realtime:
                current_wall_time = time.time()
                self.current_time = current_wall_time - self.simulation_start_time
                
                # Check for new user arrivals based on wall time
                if current_wall_time >= next_arrival_wall_time and len(self.active_sessions) < self.max_concurrent_users:
                    self._create_new_user()
                    next_arrival_time = self._get_next_arrival_time()
                    next_arrival_wall_time = current_wall_time + next_arrival_time
                    event_logger.info(f"NEXT_USER_ARRIVAL: time={datetime.fromtimestamp(next_arrival_wall_time).strftime('%H:%M:%S.%f')[:-3]}")
                
                # Small sleep to avoid CPU spinning
                time.sleep(0.01)
            else:
                # Original simulated time approach
                # Small sleep to avoid CPU spinning
                time.sleep(0.01)
                
                # Update simulation time
                self.current_time += 0.01
                
                # Check for new user arrivals
                if self.current_time >= next_arrival_time and len(self.active_sessions) < self.max_concurrent_users:
                    self._create_new_user()
                    next_arrival_time = self.current_time + self._get_next_arrival_time()
            
            # Process active sessions
            self._process_active_sessions()
    
    def _get_next_arrival_time(self) -> float:
        """Get time until next user arrival (exponential distribution)"""
        return np.random.exponential(scale=1.0/self.rate_lambda)
    
    def _create_new_user(self):
        """Create a new user session"""
        user_id = self.next_user_id
        self.next_user_id += 1
        
        # Select random WAV file
        wav_path = random.choice(self.wav_paths)
        
        # Select random latency multiplier (1, 2, or 4)
        latency_multiplier = random.choice([1, 2, 4])
        
        # Calculate session duration from normal distribution
        duration = max(5.0, np.random.normal(
            loc=self.session_duration_mean,
            scale=self.session_duration_stddev
        ))
        
        logger.info(f"Creating user {user_id} (active users: {len(self.active_sessions)}/{self.max_concurrent_users})")
        
        # Create user session
        session = UserSession(
            user_id=user_id,
            wav_path=wav_path,
            start_time=self.current_time,
            duration=duration,
            latency_multiplier=latency_multiplier,
            submit_fn=self.submit_fn,
            realtime=self.realtime
        )
        
        self.active_sessions[user_id] = session
        event_logger.info(f"USER_ADDED_TO_SIMULATION: user_id={user_id}, active_users={len(self.active_sessions)}")
    
    def _process_active_sessions(self):
        """Process all active sessions - submit segments when it's time"""
        expired_sessions = []
        
        for user_id, session in self.active_sessions.items():
            # Check if session has expired
            if session.has_expired(self.current_time):
                expired_sessions.append(user_id)
                event_logger.info(f"SESSION_EXPIRED: user_id={user_id}")
                continue
            
            # In realtime mode, check if it's time to submit the next segment
            if self.realtime:
                # Check if we've reached the next segment time
                if session.time_to_next_segment() <= 0:
                    self._submit_speech_segment(user_id)
            else:
                # For non-realtime mode, we need to check against the simulation time
                # instead of wall clock time
                if self.current_time >= session.get_next_segment_time():
                    self._submit_speech_segment(user_id)
        
        # Remove expired sessions
        for user_id in expired_sessions:
            logger.info(f"Session expired for user {user_id}")
            self.active_sessions.pop(user_id)
    
    def _submit_speech_segment(self, user_id: int):
        """Submit a speech segment for processing"""
        session = self.active_sessions.get(user_id)
        if session is None:
            logger.warning(f"Tried to submit segment for non-existent user {user_id}")
            return
            
        # Get next segment
        segment = session.get_next_segment(is_realtime=self.realtime, current_sim_time=self.current_time)
        
        if segment is None or session.is_finished():
            logger.info(f"End of audio data for user {user_id}")
            event_logger.info(f"END_OF_AUDIO: user_id={user_id}")
            self.active_sessions.pop(user_id)
            return
        
        # Submit the segment
        event_logger.info(f"SEGMENT_SUBMITTED: user_id={user_id}, segment_num={session.segment_count}, " +
                         f"samples={len(segment)}, next_segment_in={session.segment_interval}s")
        
        logger.debug(f"Submitting segment for user {user_id} at time {self.current_time:.2f}s")
        self.submit_fn(user_id, segment) 