# Data Preparation

This directory contains scripts and utilities for preparing datasets for the InfiniSST training.

## Data Organization

We are using the [MuST-C](https://aclanthology.org/N19-1202/) dataset. Sadly it is not distributed by [FBK](https://ict.fbk.eu/must-c/) anymore due to some licensing issues, but if you downloaded it before, you can still use this script. The data should be organized as follows (take en-de as an example):

```
$ROOT/en-de/
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
`$ROOT` is the root directory of the dataset.

The tsv files should have the following format (take train.tsv as an example):

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
```bash
# Prepare MFA inputs
conda activate infinisst
PYTHONPATH=$PYTHONPATH:$(pwd) python data_prep/prep_mfa.py --data-root $ROOT/en-de/

# Conduct forced alignment
conda activate mfa

# Align train set
# This might take a few hours, if you want to speed it up, you can split the dataset into smaller subsets and launch a process on each subset.
cd $ROOT/en-de/data/train/mfa
mfa align --clean --final_clean --single_speaker --num_jobs 8 --overwrite --output_format long_textgrid . english_mfa english_mfa textgrids

# Align dev set
cd $ROOT/en-de/data/dev/mfa
mfa align --clean --final_clean --overwrite --output_format long_textgrid . english_mfa english_mfa textgrids
```