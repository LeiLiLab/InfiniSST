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
conda install seaborn -c conda-forge
pip install jupyter

git clone https://github.com/facebookresearch/SimulEval.git
cd SimulEval
pip install -e .

```

## Training
```bash
sbatch run.job
```

### Before Stage 1 Training
Make sure Speech Encoder and the adapters have been extracted before proceeding with stage 1 training if using encoder trained on stage 0.
```bash
cd train
speech_encoder_dir = 'dirname_of_your_stage0_checkpoint'
speech_encoder_name = 'name_of_your_stage0_checkpoint'

python ./extract_adapter.py \
  --model_name_or_path ${speech_encoder_dir}/${speech_encoder_name} \
  --speech-encoder-only \
  --extracted_name 'mm_length_adapter' \
  --output ${speech_encoder_dir}/length_adapter.bin 

python ./extract_adapter.py \
  --model_name_or_path ${speech_encoder_dir}/${speech_encoder_name} \
  --speech-encoder-only \
  --extracted_name 'mm_mlp_adapter' \
  --output ${speech_encoder_dir}/mlp_adapter.bin 

python ./extract_adapter.py \
  --model_name_or_path ${speech_encoder_dir}/${speech_encoder_name} \
  --speech-encoder-only \
  --extracted_name 'speech_tower' \
  --output ${speech_encoder_dir}/speech_tower.bin 

```

### Before Stage 2 Training
Make sure Speech Encoder and the adapters have been extracted before proceeding with stage 2 training. Use the following commands:
```bash
cd train
llm_model='path_to_your_stage1_weights'

python ./zero_to_fp32.py ${llm_model} ${llm_model}/pytorch_model.bin

python ./extract_adapter.py \
  --model_name_or_path ${llm_model} \
  --extracted_name 'mm_length_adapter' \
  --output ${llm_model}/length_adapter.bin 

python ./extract_adapter.py \
  --model_name_or_path ${llm_model} \
  --extracted_name 'mm_mlp_adapter' \
  --output ${llm_model}/mlp_adapter.bin 

python ./extract_adapter.py \
    --model_name_or_path ${llm_model} \
    --extracted_name 'speech_tower' \
    --output ${llm_model}/speech_tower.bin
```

### Speech Encoder  
set `--freeze_speech_foundation` to False to train Speech Encoder togther \
add `replace_uni_train()` to enable uni-directional encoding

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

