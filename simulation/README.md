# InfiniSST Simulation

A system for simulating concurrent speech processing with beam search in an SST (Speech-to-Text) pipeline.

## Overview

InfiniSST Simulation models the processing of multiple speech streams in a batched, asynchronous manner. The system simulates speech processing without requiring actual models, focusing on the scheduling and batching logic.

## Components

- **Simulator**: Generates synthetic user speech segments based on a Poisson process
- **Scheduler**: Manages user sessions and batches requests for processing
- **Engine**: Simulates speech and language model inference with beam search

## Processing Flow

```
[User Speech Segments] → [Scheduler]
                          |
[Prefill Queue (Batched)] → [Engine (Prefill Workers)]
                             |
                             ↓
[Decode Queue (Batched)] ← [Scheduler] ← [Prefill Results]
                            |
                            ↓
                         [Engine (Decode Workers)]
                            |
                            ↓
                         [Scheduler] ← [Decode Results]
```

1. **Speech Submission**: Users submit speech segments to the Scheduler
2. **Prefill Stage**: 
   - Batched prefill requests are processed by the Engine
   - Results are returned to the Scheduler
3. **Decode Stage**:
   - Scheduler creates batched decode requests
   - Engine performs beam search steps
   - Results either continue decoding or move to next segment

## Running the Simulation

```bash
# Navigate to the simulation directory
cd simulation

# Run with default settings
./run_simulation.sh

# With custom parameters
./run_simulation.sh --simulation-time 120 --user-rate 1.0 --max-users 32 --beam-size 4 --batch-size 8
```

## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `simulation-time` | Duration in seconds |
| `user-rate` | User arrival rate per second |
| `max-users` | Maximum concurrent users |
| `prefill-time` | Simulated time for prefill processing |
| `decode-time` | Simulated time for decode steps |
| `beam-size` | Number of beams for beam search |
| `batch-size` | Maximum batch size |
| `num-workers` | Total worker threads |

## Output

The simulation produces logs showing the processing flow, batching decisions, and performance metrics:

- `engine_events.txt`: Engine processing events
- `main_events.txt`: System-level events
- Console output: Runtime statistics 