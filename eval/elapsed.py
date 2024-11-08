import json
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_speech_frame_list(delays):
    """
    Generate the speech frame list based on the delays. 
    This function calculates the frame duration for each speech segment.
    
    Args:
        delays (list): A list of delays in seconds.
    
    Returns:
        list: A list of frame durations (time between consecutive delays).
    """
    speech_frame_list = []
    
    for i in range(len(delays)):
        if i == len(delays) - 1:
            # Last element, no subsequent delay, assign 0
            speech_frame_list.append(0)
        else:
            next_different = next((delays[j] for j in range(i + 1, len(delays)) if delays[j] != delays[i]), None)
            if next_different is None:
                # No different delay found, assign 0
                speech_frame_list.append(0)
            else:
                # Calculate the difference between current and next different delay
                speech_frame_list.append(next_different - delays[i])
    
    return speech_frame_list

def adjust_elapsed_latency(delays, elapsed):
    """
    Adjust the elapsed time based on the delays and speech frame durations.
    
    Args:
        delays (list): List of speech delays in seconds.
        elapsed (list): List of raw elapsed times that need correction.
    
    Returns:
        list: A list of corrected elapsed times.
    """
    corrected_elapsed = []
    speech_frame_list = get_speech_frame_list(delays)
    buffer = 0.0
    previous_speech_start = delays[0]
    previous_wrong_elapsed = elapsed[0]
    frame_start_index = 0

    logging.info(f"Speech Frame List: {speech_frame_list}")

    for i in range(len(elapsed)):
        speech_start = delays[i]
        wrong_elapsed = elapsed[i]

        if i == 0 or speech_start != previous_speech_start:
            # New speech frame
            frame_start_index = i
            wrong_buffer_inference_time = previous_wrong_elapsed - previous_speech_start if i > 0 else 0
            frame_inference_time = wrong_elapsed - speech_start - wrong_buffer_inference_time
        else:
            # Same speech frame, calculate relative inference time
            frame_inference_time += wrong_elapsed - elapsed[frame_start_index]

        corrected_time = buffer + frame_inference_time + speech_start
        corrected_elapsed.append(corrected_time)

        # Update buffer for the next token
        if i < len(delays) - 1:
            next_speech_start = delays[i + 1]
            if next_speech_start != speech_start:
                buffer = max(0, buffer + frame_inference_time - (next_speech_start - speech_start))
                # buffer += max(0, frame_inference_time - (next_speech_start - speech_start))


        logging.debug(f"Token {i}:")
        logging.debug(f"  Speech Start: {speech_start}")
        logging.debug(f"  Wrong Elapsed: {wrong_elapsed}")
        logging.debug(f"  Corrected Time: {corrected_time}")
        logging.debug(f"  Buffer: {buffer}")
        logging.debug(f"  Frame Inference Time: {frame_inference_time}")
        logging.debug("--------------------")

        # Update previous values for the next iteration
        previous_speech_start = speech_start
        previous_wrong_elapsed = wrong_elapsed

    return corrected_elapsed

def update_log_file(log_file_path, new_file_path):
    """
    Load the log file, correct the elapsed times, and save the updated log to a new file.
    
    Args:
        log_file_path (str): Path to the original log file (instances.log).
        new_file_path (str): Path to save the corrected log file (instance_corrected.log).
    
    Returns:
        None
    """
    try:
        # Step 1: Load the log data (multiple instances)
        with open(log_file_path, 'r') as log_file:
            log_data = [json.loads(line) for line in log_file if line.strip()]

        # Step 2: Process each instance and adjust elapsed times
        for instance in log_data:
            delays = instance.get('delays', [])
            elapsed = instance.get('elapsed', [])

            if not delays or not elapsed:
                logging.error("Instance missing 'delays' or 'elapsed' data.")
                continue

            logging.info(f"Processing instance with delays: {delays} and elapsed: {elapsed}")
            corrected_elapsed = adjust_elapsed_latency(delays, elapsed)

            # Update the instance with the corrected elapsed time
            instance['elapsed'] = corrected_elapsed

        # Step 3: Write the updated instances to the new log file
        with open(new_file_path, 'w') as new_log_file:
            for instance in log_data:
                new_log_file.write(json.dumps(instance) + '\n')

        logging.info(f"Updated log file saved to {new_file_path} successfully.")
    
    except FileNotFoundError:
        logging.error(f"Log file {log_file_path} not found.")
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {log_file_path}.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")

# Example usage:
# log_file_path = "instances.log"
# new_file_path = "instance_corrected.log"
# update_log_file(log_file_path, new_file_path)