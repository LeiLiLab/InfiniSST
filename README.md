# InfiniSST

## Online Demo

## Installation

```bash
conda create -n infinisst -y python=3.9
conda activate infinisst

# torch and related packages
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.47.0 evaluate jiwer lightning accelerate deepspeed rotary_embedding_torch torchtune sentence-transformers wandb tensorboardX matplotlib soundfile simuleval jupyter jieba unbabel-comet simalign praat-textgrids
pip install flash-attn --no-build-isolation

# fairseq for wav2vec2
git clone git@github.com:facebookresearch/fairseq.git
cd fairseq
git checkout 0.12.2-release
pip install pip==23.3
pip install -e .

# serving
pip install fastapi uvicorn python-multipart websockets
```

Finally, you can clone the repository and checkout to the release branch.

```bash
git clone git@github.com:siqiouya/InfiniSST.git
cd InfiniSST
git checkout release
```

Also you need to login wandb with `wandb login` to use the `wandb` package.

## Data Preparation

For detailed information about data preparation, please refer to the [Data Preparation README](preprocess/README.md).

## Training

You need to first download the pre-trained speech encoder [wav2vec 2.0](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_vox_960h_pl.pt) and [Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct).
Then you need to fill in the following variables in the `scripts/train/stage1.sh` script.

```bash
llama_path= # path to the Llama-3.1-8B-Instruct model
w2v2_path= # path to the wav2vec 2.0 model
ROOT= # path to the root directory of the data
lang_code= # language code, e.g. de, es, zh, etc.
lang= # language name, e.g. German, Spanish, Chinese, etc.
save_dir= # path to the directory to save the model
```

Then you can run the following script to train the model. By default, we assume you are at the root directory of the repository when running the script. If not, you need to set the `PYTHONPATH` variable to the root directory of the repository.

```bash
# use sbatch to run the script on the SLURM cluster
# if you are running on a single machine, you can run the script directly
sbatch scripts/train/stage1.sh
```

After the first stage of training, you need to set the aforementioned variables together with the `stage1_ckpt_dir` variable in the `scripts/train/stage2.sh` script to the path of the checkpoint saved in the first stage. Then you can run the following script to train the model for the second stage.

```bash
sbatch scripts/train/stage2.sh
```

## Inference

After the training is complete, you can use simuleval to perform inference on the tst-COMMON set.
You need to fill in the following variables in the `scripts/infer/infinisst.sh` script.

```bash
checkpoint_dir= # path to the stage 2 checkpoint directory
llama_path= # path to the Llama-3.1-8B-Instruct model
w2v2_path= # path to the wav2vec 2.0 model
w2v2_type= # wav2vec 2.0 type
ctc_finetuned= # whether the wav2vec 2.0 model is finetuned
ROOT= # path to the root directory of the data
lang_code= # language code, e.g. de, es, zh, etc.
lang= # language name, e.g. German, Spanish, Chinese, etc.
tokenizer= # tokenizer, e.g. 13a, zh, etc.
unit= # unit, e.g. word, char, etc.
```

Then you can run the following script
```bash
sbatch scripts/infer/infinisst.sh
```

## Evaluation with StreamLAAL

## Serving

## Citation

## Contact

