import os
import argparse

from tqdm import tqdm
import soundfile as sf

from data_prep.utils import read_tsv, read_wav

splits = ['train', 'dev']

args = argparse.ArgumentParser()
args.add_argument('--data-root', type=str, required=True)
args = args.parse_args()

for split in splits:
    tsv_path = os.path.join(args.data_root, split + '.tsv')
    samples = read_tsv(tsv_path)

    mfa_dir = os.path.join(args.data_root, 'data', split, 'mfa')
    os.makedirs(mfa_dir, exist_ok=True)

    for sample in tqdm(samples, desc=f'Preparing MFA inputs for {split} set'):
        wav, sr = read_wav(sample['audio'])
        wav_path = os.path.join(mfa_dir, sample['id'] + '.wav')
        sf.write(wav_path, wav, sr)

        with open(os.path.join(mfa_dir, sample['id'] + '.txt'), 'w') as f:
            f.write(sample['src_text'])