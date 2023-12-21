from transformers import AutoModelForCTC
import torch
import pandas as pd


# model = AutoModelForCTC.from_pretrained("LuisG07/wav2vec2-large-xlsr-53-spanish")

# torch.save(model.state_dict(), '/mnt/data/xixu/models/wav2vec2-large-xlsr-53-spanish.pt')

# checkpoint = torch.load("/mnt/data/xixu/models/wav2vec2-large-xlsr-53-spanish.pt", map_location='cpu')
# print(checkpoint.keys())

# import csv
# from tqdm import tqdm

# def modify_tsv(file_path, output_path, new_base_path):
#     with open(file_path, 'r', encoding='utf-8') as file, open(output_path, 'w', encoding='utf-8', newline='') as out_file:
#         reader = csv.reader(file, delimiter='\t')
#         writer = csv.writer(out_file, delimiter='\t')

#         # Write header
#         header = next(reader)
#         writer.writerow(header)

#         # Process each row
#         for row in tqdm(reader, desc="Processing rows"):
#             try:
#                 # The 'audio' column is at index 1
#                 row[1] = new_base_path + row[1]
#                 writer.writerow(row)
#             except IndexError:
#                 print(f"Error processing line: {reader.line_num}")

# file_path = '/mnt/data/xixu/datasets/must-c-v1.0/dev_st_es.tsv'
# output_path = '/mnt/data/xixu/datasets/must-c-v1.0/en-es/dev.tsv'
# new_base_path = '/mnt/data/xixu/datasets/must-c-v1.0/'

# modify_tsv(file_path, output_path, new_base_path)

# import csv
# from tqdm import tqdm

# def update_audio_paths(file_path, output_path, expected_base_path):
#     with open(file_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8', newline='') as outfile:
#         reader = csv.reader(infile, delimiter='\t')
#         writer = csv.writer(outfile, delimiter='\t')

#         header = next(reader)  
#         writer.writerow(header)  

#         for row in tqdm(reader, desc="Updating paths"):
#             audio_path = row[1]  

#             if not audio_path.startswith(expected_base_path):
#                 row[1] = expected_base_path + audio_path.split('/')[-1]

#             writer.writerow(row)

# file_path = '/mnt/data/xixu/datasets/must-c-v1.0/train_st_es.tsv'
# output_path = '/mnt/data/xixu/datasets/must-c-v1.0/en-es/train.tsv'
# expected_base_path = '/mnt/data/xixu/datasets/must-c-v1.0/'

# update_audio_paths(file_path, output_path, expected_base_path)

import os

def convert_to_absolute(input_file, output_file, root_dir):
    with open(input_file, 'r') as file, open(output_file, 'w') as outfile:
        header = file.readline()
        outfile.write(header)

        for line in file:
            columns = line.strip().split('\t')
            
            path = columns[1]

            if not path.startswith('/'):
                absolute_path = os.path.join(root_dir, path)
                columns[1] = absolute_path

            outfile.write('\t'.join(columns) + '\n')

input_tsv = '/mnt/data/xixu/datasets/must-c-v1.0/en-es/tst-COMMON_st_es.tsv' 
output_tsv = '/mnt/data/xixu/datasets/must-c-v1.0/en-es/tst-COMMON_st_es_1.tsv' 
root_directory = '/mnt/data/xixu/datasets/must-c-v1.0'  

convert_to_absolute(input_tsv, output_tsv, root_directory)


