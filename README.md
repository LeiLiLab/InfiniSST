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

Also you need to login wandb with `wandb login` to use the `wandb` package.

## Data Construction

For detailed information about data preparation, please refer to the [Data Preparation README](data_prep/README.md).

## Training

## Evaluation

## Serving

## Citation

## Contact

