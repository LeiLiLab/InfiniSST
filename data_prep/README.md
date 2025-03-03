# Data Preparation

This directory contains scripts and utilities for preparing datasets for the InfiniSST training.

## Data Organization

We are using the [MuST-C](https://aclanthology.org/N19-1202/) dataset. Sadly it is not distributed by [FBK](https://ict.fbk.eu/must-c/) anymore due to some licensing issues, but if you downloaded it before, you can still use this script. The data should be organized as follows. 

```
$ROOT/en-$lang/
      train.tsv
      dev.tsv
      tst-COMMON.tsv
      data/
         train/
            wav/
            txt/
         dev/
            wav/
            txt/
         tst-COMMON/
            wav/
            txt/
```
`$ROOT` is the root directory of the dataset and `$lang` is the language code (e.g., `de`, `es`, `zh` and etc.).

The tsv files should have the following format (take train.tsv of en-de as an example):

| Column    | Value |
|-----------|--------------|
| id        | ted_1_0 |
| audio     | en-de/data/train/wav/ted_1.wav:1872800:25920 |
| n_frames  | 25920 |
| speaker   | spk.1 |
| src_text  | There was no motorcade back there. |
| tgt_text  | Hinter mir war gar keine Autokolonne. |
| src_lang  | en |
| tgt_lang  | de |


## Forced Alignment

Forced alignment is performed using the [Montreal Forced Aligner](https://montreal-forced-aligner.readthedocs.io/en/latest/index.html). 
First you need to install the MFA in a separate conda environment.

```bash
conda create -n mfa -y
conda activate mfa
conda config --add channels conda-forge
conda install -y montreal-forced-aligner
mfa model download dictionary english_mfa
mfa model download acoustic english_mfa
```

Then you need to conduct forced alignment on the utterances in the train and dev sets.

**You need to start at the root of the repository and run the following commands.**
```bash
# Prepare MFA inputs
conda activate infinisst
export PYTHONPATH=$PYTHONPATH:$(pwd) 
python data_prep/prep_mfa.py --data-root $ROOT/en-$lang/

# Conduct forced alignment
conda activate mfa

# Align train set
mfa align --clean --final_clean --single_speaker --num_jobs 8 --overwrite --output_format long_textgrid $ROOT/en-$lang/data/train/mfa english_mfa english_mfa $ROOT/en-$lang/data/train/mfa/textgrids

# Align dev set
mfa align --clean --final_clean --overwrite --output_format long_textgrid $ROOT/en-$lang/data/dev/mfa english_mfa english_mfa $ROOT/en-$lang/data/dev/mfa/textgrids
```

Textgrids are saved in the `$ROOT/en-$lang/data/{train,dev}/mfa/textgrids` directory, which are used later for trajectory construction.

## Remove speaker names

We remove speaker names from transcripts and translations and store the new tsv files as `train_nospeaker.tsv` and `dev_nospeaker.tsv`.

```bash
conda activate infinisst
python data_prep/remove_speakers.py \
   --tsv-path $ROOT/en-$lang/train.tsv
python data_prep/remove_speakers.py \
   --tsv-path $ROOT/en-$lang/dev.tsv
```

## Form trajectory and robust segments

We form trajectories and robust segments for the train and dev sets and store them as `train_nospeaker_traj_30.tsv` and `dev_nospeaker_traj_30.tsv`.

```bash
python data_prep/build_trajectory_full_mfa.py \
	--data-root $ROOT/en-$lang \
	--lang ${lang} --split train_nospeaker \
	--output-split train_nospeaker_traj_30 \
	--mult 30 --max-duration 28.8
python data_prep/build_trajectory_full_mfa.py \
	--data-root $ROOT/en-$lang \
	--lang ${lang} --split dev_nospeaker \
	--output-split dev_nospeaker_traj_30 \
	--mult 30 --max-duration 28.8
```

## Filter with Whisper ASR

Finally, we filter out the utterances whose transcripts are not aligned with Whisper ASR outputs and store the filtered data as `train_nospeaker_traj_30_filtered.tsv` and `dev_nospeaker_traj_30_filtered.tsv`.

```bash
# we split the data into 8 parts and run ASR on each part in parallel
# use this only if you have slurm, otherwise, run asr.py directly
sbatch data_prep/asr.sh $ROOT/en-$lang/train_nospeaker_traj_30.tsv
sbatch data_prep/asr.sh $ROOT/en-$lang/dev_nospeaker_traj_30.tsv

# filter the data
python data_prep/filter_by_asr.py \
	--tsv-path $ROOT/en-${lang}/train_nospeaker_traj_30.tsv
python data_prep/filter_by_asr.py \
	--tsv-path $ROOT/en-${lang}/dev_nospeaker_traj_30.tsv
```

## Prepare SimulEval Inputs for tst-COMMON set

We prepare the SimulEval inputs for the tst-COMMON set.
The output files are `tst-COMMON_full.source` and `tst-COMMON_full.target` in the `$ROOT/en-$lang` directory.

```bash
python data_prep/prepare_simuleval_inputs.py \
	--tsv-path $ROOT/en-$lang/tst-COMMON.tsv
```