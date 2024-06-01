import os
import shutil
import tempfile

def update_audio_paths(input_tsv_path, old_prefix, new_prefix, audio_col_index=1):
    # Temporary file to store updates
    temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8')

    corrected_count = 0
    try:
        with open(input_tsv_path, 'r', encoding='utf-8') as infile, temp_file:
            header = next(infile).strip()
            temp_file.write(header + '\n')
    
            for line_number, line in enumerate(infile, start=2):
                columns = line.strip().split('\t')
                
                if len(columns) > audio_col_index:
                    audio_path = columns[audio_col_index]
                    
                    if audio_path.startswith(old_prefix):
                        corrected_path = audio_path.replace(old_prefix, new_prefix)
                        columns[audio_col_index] = corrected_path
                        corrected_count += 1
                    
                    temp_file.write('\t'.join(columns) + '\n')
                else:
                    print(f"Error: Line {line_number} does not have enough columns. Skipping this line.")

        # Close the temp_file to ensure all data is written
        temp_file.close()

        # Safely copy the temporary file contents to the original file location
        shutil.copy(temp_file.name, input_tsv_path)
    finally:
        # Ensure the temporary file is removed after copying
        os.unlink(temp_file.name)
    
    return corrected_count

# Example usage:
input_path = '/scratch/xixu/dataset/must-c-v1.0/en-es/train.tsv'
old_prefix = '/scratch/siqiouya/'
new_prefix = '/scratch/xixu/'
corrected_count = update_audio_paths(input_path, old_prefix, new_prefix)
print(f"Total corrected paths: {corrected_count}")

input_path = '/scratch/xixu/dataset/must-c-v1.0/en-es/tst-COMMON.tsv'
old_prefix = '/scratch/siqiouya/'
new_prefix = '/scratch/xixu/'
corrected_count = update_audio_paths(input_path, old_prefix, new_prefix)
print(f"Total corrected paths: {corrected_count}")

input_path = '/scratch/xixu/dataset/must-c-v1.0/en-es/dev.tsv'
old_prefix = '/scratch/siqiouya/'
new_prefix = '/scratch/xixu/'
corrected_count = update_audio_paths(input_path, old_prefix, new_prefix)
print(f"Total corrected paths: {corrected_count}")

input_path = '/scratch/xixu/dataset/must-c-v1.0/en-es/train_mfa_30s_mix_filtered.tsv'
old_prefix = '/scratch/siqiouya/'
new_prefix = '/scratch/xixu/'
corrected_count = update_audio_paths(input_path, old_prefix, new_prefix)
print(f"Total corrected paths: {corrected_count}")

input_path = '/scratch/xixu/dataset/must-c-v1.0/en-es/dev_mfa_30s_mix_filtered.tsv'
old_prefix = '/scratch/siqiouya/'
new_prefix = '/scratch/xixu/'
corrected_count = update_audio_paths(input_path, old_prefix, new_prefix)
print(f"Total corrected paths: {corrected_count}")