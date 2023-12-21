## Requirements
```bash
conda activate sllama
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



