## Requirements
```bash
# conda env create -f environment.yml

# conda create -y -n sllama python=3.8
# conda activate sllama
# conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# conda install -y -c conda-forge fairseq pysoundfile sentencepiece
# conda install -y jupyter

# pip install transformers deepspeed
# pip install flash-attn --no-build-isolation
# pip install "fschat[model_worker,webui]"

use sid's conda env
pip install jupyter

git clone https://github.com/facebookresearch/SimulEval.git
cd SimulEval
pip install -e .

```

## Training
```bash
sbatch run.job
```

### Before Stage 2 Training
Ensure that the adapters have been extracted before proceeding with stage 2 training. Use the following commands:
```bash
cd train

python ./zero_to_fp32.py ${llm_model}/checkpoint-12000 ${llm_model}/checkpoint-12000/pytorch_model.bin

model_path=/mnt/data/xixu/runs/sllama/en-es/13b/stage1
python ./extract_adapter.py \
  --model_name_or_path ${model_path}/checkpoint-12000 \
  --extracted_name 'mm_length_adapter' \
  --output ${model_path}/length_adapter.bin 
python ./extract_adapter.py \
  --model_name_or_path ${model_path}/checkpoint-12000 \
  --extracted_name 'mm_mlp_adapter' \
  --output ${model_path}/mlp_adapter.bin 
```

### Train Speech Encoder together in Stage1
please set --freeze_speech_foundation to False

## Evaluation
```bash
sbatch evl.job
```

## Simultaneous Evaluation

```bash
simuleval \
  --agent agents/tt_waitk_sllama.py \
  --source-segment-size 640 \
  --waitk-lagging 3 \
  --model-dir /mnt/taurus/data/xixu/runs/sllama/en-es/7b/stage2/checkpoint-2100 \
  --prompt "<speech_here> Start by converting the English audio into Spanish written form." \
  --source tmp/source.txt \
  --target tmp/reference.txt \
  --output tmp/prediction \
  --quality-metrics BLEU --sacrebleu-tokenizer 13a
```

