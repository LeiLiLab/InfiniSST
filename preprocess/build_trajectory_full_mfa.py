import os
import csv
import argparse
import numpy as np
import textgrids
from tqdm import tqdm
from simalign import SentenceAligner
import jieba
import soundfile as sf

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=str, required=True)
parser.add_argument("--lang", type=str, required=True)
parser.add_argument("--split", type=str, required=True)
parser.add_argument("--mult", type=int, default=60)
parser.add_argument("--output-split", type=str, default='dev_traj_full')
parser.add_argument("--max-duration", type=float, default=43.2)
args = parser.parse_args()

myaligner = SentenceAligner(model="pvl/labse_bert", token_type="bpe", matching_methods="a", device='cuda')
# myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="a", device='cuda')

tsv_path = os.path.join(args.data_root, "{}.tsv".format(args.split))
with open(tsv_path) as f:
    reader = csv.DictReader(
        f,
        delimiter="\t",
        quotechar=None,
        doublequote=False,
        lineterminator="\n",
        quoting=csv.QUOTE_NONE,
    )
    samples = [dict(e) for e in reader]

stepsize = int(0.96 * 16000)
n_skipped = 0
data_split = 'train' if 'dev' not in args.split else 'dev'
textgrid_dir = os.path.join(args.data_root, "data", data_split, "mfa", "textgrids")
for sample in tqdm(samples):
    offset = int(sample['audio'].split(':')[1])
    offset_rounded = offset // stepsize * stepsize

    tg_path = os.path.join(textgrid_dir, sample['id'] + '.TextGrid')
    if not os.path.exists(tg_path):        
        sample['trajectory'] = [offset_rounded]
        n_skipped += 1
        continue
    tg = textgrids.TextGrid(tg_path)

    if int(sample['n_frames']) / 16000 > args.max_duration:
        sample['trajectory'] = [offset_rounded]
        n_skipped += 1
        continue

    src_text = sample['src_text'].replace("(Laughing)", "(Laughter)")
    tgt_text = sample['tgt_text']
    src_text_l = src_text.lower()

    # The source and target sentences should be tokenized to words.
    src_words = src_text.split(' ')
    tgt_words = tgt_text.split(' ') if args.lang != 'zh' else list(jieba.cut(tgt_text))

    # The output is a dictionary with different matching methods.
    # Each method has a list of pairs indicating the indexes of aligned words (The alignments are zero-indexed).
    try:
        alignments = myaligner.get_word_aligns(src_words, tgt_words)
    except Exception as e:
        print(e)
        sample['trajectory'] = [offset_rounded]
        n_skipped += 1
        continue

    alignments = sorted(alignments['inter'], key=lambda x: (x[1], x[0]))
    alignments.append((len(src_words) - 1, len(tgt_words) - 1))
    alignments_r = []
    for a in alignments:
        if len(alignments_r) > 0 and alignments_r[-1][1] == a[1]:
            alignments_r[-1] = a
        else:
            alignments_r.append(a)
    for i, a in enumerate(alignments_r):
        if i == 0:
            continue
        alignments_r[i] = (max(a[0], alignments_r[i - 1][0]), a[1])
    alignments_r = [(-1, -1)] + alignments_r

    # mapping
    mapping = []    
    p = 0
    flag = False
    for w in tg['words']:
        t = w.text

        if t.strip() == '':
            continue
        if t == "(bracketed)" or t == "[bracketed]":
            continue
        if t == "[laughter]":
            t = "(laughter)"
        
        if src_text_l.find(t, p) == -1 and "'" in t[1 : -1]:
            pos = t.rfind("'")
            t = t[pos + 1:]
        
        if src_text_l.find(t, p) == -1 and t.isdigit():
            t = f"{int(t):,}"
        
        if src_text_l.find(t, p) == -1:
            print(src_text_l, t, sep='----')
            flag = True
            break

        p = src_text_l.find(t, p) + len(t)
        idx = src_text_l[:p].count(' ')

        if len(mapping) > 0 and mapping[-1][1] == idx:
            mapping[-1] = (w.xmax, idx)
        else:
            mapping.append((w.xmax, idx))    

    if flag:
        sample['trajectory'] = [offset_rounded]
        n_skipped += 1
        continue

    mapping.append((tg.xmax, src_text_l.count(' ')))
    # sample['mapping'] = mapping

    j = k = -1
    r = 0
    src_segments = []
    trajectory = []

    n_frame = int(sample["n_frames"])
    
    for i in np.arange(offset_rounded, offset + n_frame, stepsize):
        rbound = min(i + stepsize, offset + n_frame) - offset
        while j < len(mapping) - 1 and int(mapping[j + 1][0] * 16000) <= rbound:
            j += 1
        if j >= 0 and int(mapping[j][0] * 16000) > i - offset:
            src_segments.append(' '.join(src_words[k + 1 : mapping[j][1] + 1]))
            k = mapping[j][1]

            old_r = r
            while r < len(alignments_r) - 1 and alignments_r[r + 1][0] <= k:
                r += 1
            tgt_segment = tgt_words[alignments_r[old_r][1] + 1 : alignments_r[r][1] + 1]
            trajectory.append(' '.join(tgt_segment) if args.lang != 'zh' else ''.join(tgt_segment))

        else:
            src_segments.append('')
            trajectory.append('')
    trajectory[-1] += ' '
    sample['src_segments'] = src_segments
    sample['trajectory'] = [offset_rounded, trajectory]

print("n_skipped", n_skipped)

samples = sorted(samples, key=lambda x: x['trajectory'][0])

id2samples = {}
for sample in samples:
    ted_id = sample['id'].split('_')[1]
    if ted_id not in id2samples:
        id2samples[ted_id] = [sample]
    else:
        id2samples[ted_id].append(sample)

max_step = args.mult
max_n_frame_per_slice = max_step * stepsize
slices = []
for id, _samples in id2samples.items():
    audio_path = _samples[0]['audio'].split(':')[0]
    wav, sr = sf.read(audio_path)
    n_frame = wav.shape[0]
    offset = 0
    idx_in_ted = 0
    i = -1

    while offset < n_frame:
        duration = min(max_n_frame_per_slice, n_frame - offset)
        slice_traj = [''] * ((duration + stepsize - 1) // stepsize)
        slice_src = ""
        new_offset = -1
        while i < len(_samples) - 1 and _samples[i + 1]['trajectory'][0] < offset + duration:
            i += 1
            if len(_samples[i]['trajectory']) == 1:
                duration = _samples[i]['trajectory'][0] - offset
                slice_traj = slice_traj[:((duration + stepsize - 1) // stepsize)]

                end_frame = int(_samples[i]['audio'].split(':')[1]) + int(_samples[i]['n_frames'])
                new_offset = end_frame // stepsize * stepsize
                
                break
            sample_offset_rounded, sample_traj = _samples[i]['trajectory']
            for j, seg in enumerate(sample_traj):
                if sample_offset_rounded + j * stepsize - offset < duration:
                    slice_traj[sample_offset_rounded // stepsize + j - offset // stepsize] += seg + (' ' if args.lang != 'zh' else '')
                    if _samples[i]['src_segments'][j] != '':
                        slice_src += _samples[i]['src_segments'][j] + ' '
            
        slices.append(
            {
                "id": "ted_{}_{}".format(id, idx_in_ted),
                "audio": "{}:{}:{}".format(audio_path, offset, duration),
                "n_frames": duration,
                'speaker': _samples[0]['speaker'],
                'src_text': slice_src,
                'tgt_text': ''.join(slice_traj),
                'src_lang': _samples[0]['src_lang'],
                'tgt_lang': _samples[0]['tgt_lang'],
                'trajectory': slice_traj
            }
        )
        idx_in_ted += 1
            
        if new_offset != -1:
            offset = new_offset
        else:
            if i >= 0 and _samples[i]['trajectory'][0] > offset:
                offset = _samples[i]['trajectory'][0]
                i -= 1
            else:
                offset += duration

output_path = os.path.join(args.data_root, "{}.tsv".format(args.output_split))
with open(output_path, "w") as w:
    writer = csv.DictWriter(
        w,
        slices[0].keys(),
        delimiter="\t",
        quotechar=None,
        doublequote=False,
        lineterminator="\n",
        quoting=csv.QUOTE_NONE,
        extrasaction='ignore'
    )
    writer.writeheader()
    writer.writerows(slices)