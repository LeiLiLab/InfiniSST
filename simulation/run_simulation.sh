#!/bin/bash
set -e  # Exit on error

# Path to the source file containing WAV paths
SOURCE_FILE="/home/xixu/work/llmsys/InfiniSST/dev.source.corrected"

# Verify source file exists
if [ ! -f "$SOURCE_FILE" ]; then
    echo "Error: Source file $SOURCE_FILE not found."
    echo "Please make sure the path is correct."
    exit 1
fi

echo "Using WAV paths from: $SOURCE_FILE"

# Default parameters
SIM_TIME=60
USER_RATE=0.5
MAX_USERS=16
PREFILL_TIME=0.1
DECODE_TIME=0.05
BEAM_SIZE=4
BATCH_SIZE=8
NUM_WORKERS=4

# Parse command line arguments to override defaults
while [[ $# -gt 0 ]]; do
    case $1 in
        --simulation-time)
            SIM_TIME="$2"
            shift 2
            ;;
        --user-rate)
            USER_RATE="$2"
            shift 2
            ;;
        --max-users)
            MAX_USERS="$2"
            shift 2
            ;;
        --prefill-time)
            PREFILL_TIME="$2"
            shift 2
            ;;
        --decode-time)
            DECODE_TIME="$2"
            shift 2
            ;;
        --beam-size)
            BEAM_SIZE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --source-file)
            SOURCE_FILE="$2"
            shift 2
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

# Display configuration
echo "Starting InfiniSST simulation with the following parameters:"
echo "  Simulation time: ${SIM_TIME}s"
echo "  User arrival rate: ${USER_RATE} users/s"
echo "  Maximum concurrent users: ${MAX_USERS}"
echo "  Prefill processing time: ${PREFILL_TIME}s"
echo "  Decode step time: ${DECODE_TIME}s"
echo "  Beam size: ${BEAM_SIZE}"
echo "  Batch size: ${BATCH_SIZE}"
echo "  Worker threads: ${NUM_WORKERS}"
echo "  Source file: ${SOURCE_FILE}"
echo ""

# Clean up any previous log files
echo "Cleaning up previous log files..."
rm -f engine_events.txt main_events.txt

# Run the simulation
python start_simulation.py \
  --simulation-time "$SIM_TIME" \
  --user-rate "$USER_RATE" \
  --max-users "$MAX_USERS" \
  --prefill-time "$PREFILL_TIME" \
  --decode-time "$DECODE_TIME" \
  --beam-size "$BEAM_SIZE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --source-file "$SOURCE_FILE"

echo "Simulation completed. Check log files for results."
echo "Event logs have been written to:"
echo "- main_events.txt: Main process events"
echo "- engine_events.txt: Engine processing events" 