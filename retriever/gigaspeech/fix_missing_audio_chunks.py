import os
import json
import soundfile as sf
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-tsv", default="/mnt/gemini/data1/jiaxuanluo/train_xl_case_robust_asr-filtered_zh_metricx-qe3.0_align.tsv")
    parser.add_argument("--missing-info", default="/mnt/gemini/data1/jiaxuanluo/missing_chunks_info.json")
    parser.add_argument("--output-dir", default="/mnt/gemini/data1/jiaxuanluo/term_train_audio_chunks")
    args = parser.parse_args()

    if not os.path.exists(args.missing_info):
        logger.error(f"Missing info file not found: {args.missing_info}")
        return

    logger.info(f"Loading missing info from {args.missing_info}...")
    with open(args.missing_info, "r") as f:
        missing_map = json.load(f)

    if not missing_map:
        logger.info("No missing chunks to process.")
        return

    logger.info(f"Targeting {len(missing_map)} utterances for regeneration...")
    os.makedirs(args.output_dir, exist_ok=True)

    SAMPLE_RATE = 16000
    SAMPLES_PER_UNIT = 15360 # 0.96s

    count_regenerated = 0
    
    with open(args.input_tsv, "r", encoding="utf-8") as f:
        # Skip header if it exists (usually TSVs have headers)
        header = f.readline()
        if not header.startswith("id\taudio"):
            # If no header, seek back to start
            f.seek(0)
        
        for line in tqdm(f, desc="Processing TSV"):
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2: continue
            
            uid = parts[0]
            if uid not in missing_map:
                continue
            
            audio_info = parts[1]
            try:
                # Format: path:start_frame:total_frames
                info_parts = audio_info.split(":")
                if len(info_parts) != 3:
                    logger.warning(f"Invalid audio info for {uid}: {audio_info}")
                    continue
                    
                audio_path, start_frame, total_frames = info_parts
                start_frame, total_frames = int(start_frame), int(total_frames)
                
                # Load the full utterance audio
                full_audio_data, fs = sf.read(audio_path, start=start_frame, frames=total_frames)
                if fs != SAMPLE_RATE:
                    # In case the source is not 16k, we could resample, but GigaSpeech is 16k.
                    pass 
            except Exception as e:
                logger.error(f"Failed to read audio for {uid}: {e}")
                continue
            
            for j in missing_map[uid]:
                # chunk_idx is j, window is [j, j+2] units
                idx = int(j)
                start = idx * SAMPLES_PER_UNIT
                end = (idx + 2) * SAMPLES_PER_UNIT
                
                if start >= len(full_audio_data):
                    logger.warning(f"Chunk index {idx} out of range for {uid}")
                    continue
                    
                chunk_data = full_audio_data[start:min(end, len(full_audio_data))]
                
                out_path = os.path.join(args.output_dir, f"{uid}_chunk_{idx}.wav")
                sf.write(out_path, chunk_data, SAMPLE_RATE)
                count_regenerated += 1
            
            # Cleanup processed UID
            del missing_map[uid]
            if not missing_map: 
                break

    logger.info(f"Regeneration complete. Created {count_regenerated} chunks.")

if __name__ == "__main__":
    main()
