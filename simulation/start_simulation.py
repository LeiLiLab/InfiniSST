#!/usr/bin/env python3
import os
import time
import logging
import argparse
from typing import List
import glob
import threading
import sys
from datetime import datetime

from simulator import Simulator
from scheduler import Scheduler, RequestType
from engine import Engine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger("start_simulation")

# Configure event logging for the main process
event_logger = logging.getLogger("main_events")
event_logger.setLevel(logging.INFO)
event_handler = logging.FileHandler('main_events.txt')
event_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
event_logger.addHandler(event_handler)

def load_wav_paths(wav_dir: str, source_file: str = None) -> List[str]:
    """
    Load paths to WAV files for simulation
    
    Args:
        wav_dir: Directory containing WAV files
        source_file: Optional file containing paths to WAV files
        
    Returns:
        List of paths to WAV files
    """
    if source_file and os.path.exists(source_file):
        # Read paths from source file
        with open(source_file, 'r') as f:
            paths = [line.strip() for line in f.readlines()]
            
        # If paths are relative, make them absolute
        if paths and not os.path.isabs(paths[0]):
            paths = [os.path.join(wav_dir, os.path.basename(p)) for p in paths]
            
        logger.info(f"Loaded {len(paths)} WAV paths from {source_file}")
        return paths
    else:
        # Find WAV files in directory
        wav_paths = glob.glob(os.path.join(wav_dir, "**/*.wav"), recursive=True)
        logger.info(f"Found {len(wav_paths)} WAV files in {wav_dir}")
        return wav_paths

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Simulate speech processing with InfiniSST")
    
    # Simulation parameters
    parser.add_argument("--simulation-time", type=float, default=60.0,
                        help="Simulation duration in seconds (default: 60.0)")
    parser.add_argument("--user-rate", type=float, default=0.2,
                        help="User arrival rate (users per second) (default: 0.2)")
    parser.add_argument("--session-duration-mean", type=float, default=120.0,
                        help="Mean session duration in seconds (default: 120.0)")
    parser.add_argument("--session-duration-stddev", type=float, default=30.0,
                        help="Standard deviation of session duration (default: 30.0)")
    parser.add_argument("--max-users", type=int, default=32,
                        help="Maximum number of concurrent users (default: 32)")
    parser.add_argument("--initial-users", type=int, default=0,
                        help="Number of users to create at start (default: 0)")
    parser.add_argument("--realtime", action="store_true",
                        help="Run in real time (default: false)")
    
    # Engine parameters
    parser.add_argument("--prefill-time", type=float, default=0.1,
                        help="Simulated time for prefill processing (default: 0.1s)")
    parser.add_argument("--decode-time", type=float, default=0.05,
                        help="Simulated time for decode step (default: 0.05s)")
    parser.add_argument("--tokens-per-step", type=int, default=1,
                        help="Tokens generated per decode step (default: 1)")
    parser.add_argument("--max-steps", type=int, default=5,
                        help="Max decode steps before forcing completion (default: 5)")
    parser.add_argument("--beam-size", type=int, default=4,
                        help="Beam size for beam search (default: 4)")
    parser.add_argument("--vocab-size", type=int, default=30000,
                        help="Vocabulary size (default: 30000)")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of worker threads for processing (default: 4)")
    parser.add_argument("--prefill-workers", type=int, default=None,
                        help="Number of prefill worker threads (default: auto-distributed)")
    parser.add_argument("--decode-workers", type=int, default=None,
                        help="Number of decode worker threads (default: auto-distributed)")
    
    # Scheduler parameters
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Maximum batch size (default: 4)")
    parser.add_argument("--blocksize", type=int, default=48,
                        help="Blocksize for requests (default: 48)")
    parser.add_argument("--max-new-tokens", type=int, default=10,
                        help="Max new tokens to generate (default: 10)")
    parser.add_argument("--pseudo-batch-size", type=int, default=1,
                        help="Number of duplicate requests per batch (default: 1)")
    
    # Input/output parameters
    parser.add_argument("--wav-dir", type=str, default="wavs",
                        help="Directory containing WAV files (default: 'wavs')")
    parser.add_argument("--source-file", type=str, default=None,
                        help="File containing list of WAV files to use (optional)")
    
    # Misc parameters
    parser.add_argument("--stats-interval", type=float, default=5.0,
                        help="Statistics printing interval (default: 5.0s)")
    parser.add_argument("--poll-interval", type=float, default=0.01,
                        help="Polling interval (default: 0.01s)")
    
    return parser.parse_args()

def print_stats(scheduler, engine, start_time, simulation_start_time):
    """Print statistics about the simulation"""
    scheduler_stats = scheduler.get_stats()
    engine_stats = engine.get_stats()
    
    wall_time = time.time() - start_time
    
    logger.info("-" * 60)
    logger.info(f"Simulation Stats (after {wall_time:.2f}s real time):")
    logger.info(f"  Simulation time:       {time.time() - simulation_start_time:.2f}s")
    logger.info(f"  Active sessions:       {scheduler_stats['active_sessions']}")
    logger.info(f"  Prefill queue:         {scheduler_stats['prefill_queue']}")
    logger.info(f"  Decode queue:          {scheduler_stats['decode_queue']}")
    logger.info(f"  Total prefill requests: {scheduler_stats['total_prefill_requests']}")
    logger.info(f"  Total decode requests: {scheduler_stats['total_decode_requests']}")
    logger.info(f"  Total tokens generated: {scheduler_stats['total_output_tokens']}")
    logger.info(f"  Tokens per second:     {scheduler_stats['total_output_tokens'] / wall_time:.2f}")
    logger.info("-" * 60)

def handle_batch_results(results, scheduler, request_type):
    """
    Handle the results of a batch processing
    
    Args:
        results: List of result dictionaries
        scheduler: Scheduler instance
        request_type: Type of request that was processed
    """
    if not results:
        return
    
    # Log batch info
    if request_type == RequestType.PREFILL:
        event_logger.info(f"PREFILL_BATCH_RESULTS: size={len(results)}, " +
                         f"user_ids=[{', '.join(str(r['user_id']) for r in results)}]")
    else:
        event_logger.info(f"DECODE_BATCH_RESULTS: size={len(results)}, " +
                         f"user_ids=[{', '.join(str(r['user_id']) for r in results)}]")
    
    # Handle individual results
    for result in results:
        user_id = result["user_id"]
        
        if request_type == RequestType.PREFILL:
            event_logger.info(f"PREFILL_RESULT_HANDLING: user_id={user_id}, " +
                             f"continue_decode={result['continue_decode']}")
            
            scheduler.handle_prefill_result(
                user_id,
                result["continue_decode"],
                result["result"],
                result["speech_cache"]
            )
        elif request_type == RequestType.DECODE:
            token_count = len(result["output_tokens"])
            event_logger.info(f"DECODE_RESULT_HANDLING: user_id={user_id}, " + 
                             f"tokens={token_count}, " +
                             f"continue_decode={result['continue_decode']}")
            
            scheduler.handle_decode_result(
                user_id,
                result["output_tokens"],
                result["continue_decode"],
                result["past_key_values"]
            )

def main():
    """Main function"""
    # Parse command line arguments
    args = parse_args()
    
    # Configure logging
    log_file = f"simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger("start_simulation")
    
    # Load WAV files
    wav_paths = load_wav_paths(args.wav_dir, args.source_file)
    if not wav_paths:
        logger.error(f"No WAV files found in {args.wav_dir}")
        return
    logger.info(f"Loaded {len(wav_paths)} WAV files")
    
    # Create scheduler
    scheduler = Scheduler(
        max_batch_size=args.batch_size,
        blocksize=args.blocksize,
        max_new_tokens=args.max_new_tokens,
        pseudo_batch_size=args.pseudo_batch_size
    )
    
    # Create engine
    engine = Engine(
        prefill_time=args.prefill_time,
        decode_time=args.decode_time,
        tokens_per_step=args.tokens_per_step,
        max_steps=args.max_steps,
        beam_size=args.beam_size,
        vocab_size=args.vocab_size,
        realtime=args.realtime,
        pseudo_batch_size=args.pseudo_batch_size,
        num_workers=args.num_workers,
        num_prefill_workers=args.prefill_workers,
        num_decode_workers=args.decode_workers
    )
    
    # Create simulator
    simulator = Simulator(
        rate_lambda=args.user_rate,
        session_duration_mean=args.session_duration_mean,
        session_duration_stddev=args.session_duration_stddev,
        wav_paths=wav_paths,
        submit_fn=scheduler.submit_request,
        max_concurrent_users=args.max_users,
        initial_users=args.initial_users,
        realtime=args.realtime
    )
    
    # Start simulation
    start_time = time.time()
    simulation_start_time = time.time()
    
    # Set up periodic stats printing
    stop_stats = False
    
    def print_stats_periodically():
        """Print statistics periodically"""
        while not stop_stats:
            time.sleep(args.stats_interval)
            print_stats(scheduler, engine, start_time, simulation_start_time)
    
    stats_thread = threading.Thread(target=print_stats_periodically)
    stats_thread.daemon = True
    stats_thread.start()
    
    # Start the simulator
    simulator.start()
    event_logger.info(f"SIMULATION_STARTED: time={datetime.now().strftime('%H:%M:%S.%f')[:-3]}, " +
                    f"user_rate={args.user_rate}, max_users={args.max_users}")
    
    try:
        # Main loop
        while time.time() - start_time < args.simulation_time:
            # Get a batch of requests
            batch_result = scheduler.get_request_batch()
            request_type, batch = batch_result
            
            if batch:
                if request_type == RequestType.PREFILL:
                    event_logger.info(f"PROCESSING_PREFILL_BATCH: size={len(batch)}, " +
                                     f"user_ids=[{', '.join(str(r.user_id) for r in batch)}]")
                else:
                    event_logger.info(f"PROCESSING_DECODE_BATCH: size={len(batch)}, " +
                                     f"user_ids=[{', '.join(str(r.user_id) for r in batch)}]")
                
                # Submit the batch to the engine for asynchronous processing
                # Create a callback to handle results for this specific batch
                callback = lambda results, rt=request_type: handle_batch_results(results, scheduler, rt)
                
                # Process the batch
                engine.run_batch(request_type, batch, callback)
            
            # Small sleep to avoid CPU spinning
            time.sleep(args.poll_interval)
    
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
        event_logger.info("SIMULATION_INTERRUPTED: by_user=True")
    
    finally:
        # Stop the simulation
        stop_stats = True
        simulator.stop()
        engine.shutdown()  # Shut down the engine workers
        stats_thread.join(timeout=1.0)
        
        # Print final statistics
        logger.info("Final Statistics:")
        print_stats(scheduler, engine, start_time, simulation_start_time)
        event_logger.info("SIMULATION_COMPLETED")

if __name__ == "__main__":
    main() 