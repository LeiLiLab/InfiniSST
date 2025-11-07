# InfiniSST

This repository contains a demo and the implementation of our paper "InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model".

## Online Demo

The link to the online demo is [here](https://c79b-128-111-28-80.ngrok-free.app/).

## Installation

```bash

# swift training
apptainer shell \
  --nv \
  --env "MODELSCOPE_CACHE=/home/siqiouya/.cache/modelscope/" \
  --env "MEGATRON_LM_PATH=/home/siqiouya/code/Megatron-LM/" \
  --env "NCCL_P2P_DISABLE=1" \
  --env "NCCL_IB_DISABLE=1" \
  --env "WANDB_API_KEY=a4c4532b2468d62eda36ff381716ba8dad52b260" \
  docker://modelscope-registry.us-west-1.cr.aliyuncs.com/modelscope-repo/modelscope:ubuntu22.04-cuda12.8.1-py311-torch2.8.0-vllm0.11.0-modelscope1.31.0-swift3.9.1

swift export \
  --model /data/user_data/siqiouya/ckpts/pretrained/llm/Qwen3-Omni-30B-A3B-Instruct/ \
  --to_mcore true \
  --torch_dtype bfloat16 \
  --output_dir /data/user_data/siqiouya/ckpts/pretrained/llm/Qwen3-Omni-30B-A3B-Instruct-mcore

PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
ENABLE_AUDIO_OUTPUT=False \
megatron sft \
    --load /data/user_data/siqiouya/ckpts/pretrained/llm/Qwen3-Omni-30B-A3B-Instruct-mcore/ \
    --dataset /data/group_data/li_lab/siqiouya/datasets/gigaspeech/train_s_case_zh-qwen2.5-32b-instruct_marked_mfa_punc_asr.json \
    --val_dataset /data/group_data/li_lab/siqiouya/datasets/gigaspeech/dev_case_zh-qwen2.5-32b-instruct_marked_mfa_punc.json \
    --load_from_cache_file true \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_llm false \
    --freeze_vit true \
    --freeze_aligner true \
    --vit_gradient_checkpointing false \
    --packing true \
    --expert_model_parallel_size 2 \
    --moe_permute_fusion true \
    --moe_grouped_gemm true \
    --moe_shared_expert_overlap true \
    --moe_aux_loss_coeff 1e-3 \
    --micro_batch_size 1 \
    --global_batch_size 4 \
    --recompute_granularity full \
    --recompute_method uniform \
    --recompute_num_layers 1 \
    --finetune true \
    --cross_entropy_loss_fusion true \
    --lr 1e-4 \
    --lr_warmup_fraction 0.05 \
    --min_lr 1e-5 \
    --weight_decay 0.01 \
    --clip_grad 1.0 \
    --max_epochs 1 \
    --save /data/user_data/siqiouya/ckpts/test_swift/Qwen3-Omni-30B-A3B-Instruct-lora \
    --log_interval 10 \
    --eval_interval 200 \
    --save_interval 200 \
    --max_length 2048 \
    --num_workers 8 \
    --dataset_num_proc 8 \
    --no_save_optim true \
    --no_save_rng true \
    --attention_backend flash \
    --report_to wandb \
    --wandb_project gigaspeech_zh \
    --wandb_exp_name test-omni \
    --strict True # for debugging dataset

swift export \
    --mcore_adapters /data/user_data/siqiouya/ckpts/test_swift/Qwen3-Omni-30B-A3B-Instruct-lora/v1-20251104-033331/ \
    --to_hf true \
    --torch_dtype bfloat16 \
    --output_dir /data/user_data/siqiouya/ckpts/test_swift/Qwen3-Omni-30B-A3B-Instruct-lora/v1-20251104-033331-hf


# inference

apptainer shell \
  --nv \
  --env "VLLM_USE_V1=0" \
  --env "NCCL_P2P_DISABLE=1" \
  --env "NCCL_IB_DISABLE=1" \
  docker://qwenllm/qwen3-omni:3-cu124

vllm serve /data/user_data/siqiouya/ckpts/test_swift/Qwen3-Omni-30B-A3B-Instruct-lora/v1-20251104-033331-hf \
  --gpu-memory-utilization 0.95 \
  --tensor-parallel-size 2 \
  --limit-mm-per-prompt '{"audio": 60}' \
  --max-model-len 2048 \
  --enable-prefix-caching

pip install evaluate jiwer lightning deepspeed torchtune sentence-transformers wandb tensorboardX matplotlib soundfile simuleval jieba unbabel-comet simalign praat-textgrids peft

VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTHONPATH=/home/siqiouya/code/infinisst-omni \
/home/siqiouya/.local/bin/simuleval \
    --agent agents/infinisst_omni.py \
    --agent-class agents.InfiniSSTOmni \
    --source-segment-size 960 \
    --output /data/user_data/siqiouya/ckpts/test_swift/Qwen3-Omni-30B-A3B-Instruct-lora/v1-20251104-033331-hf/evaluation/RealSI/en2zh/seg960 \
    --max-new-tokens 10 \
    --max-cache-chunks 60 \
    --keep-cache-chunks 30 \
    --source-lang English \
    --target-lang Chinese \
    --min-start-sec 0 \
    --source /data/group_data/li_lab/siqiouya/datasets/RealSI/data/en2zh/sources.txt \
    --target /data/group_data/li_lab/siqiouya/datasets/RealSI/data/en2zh/targets.txt \
    --use-vllm 1 \
    --temperature 0.6 \
    --top-p 0.95 \
    --top-k 20 \
    --model-name /data/user_data/siqiouya/ckpts/test_swift/Qwen3-Omni-30B-A3B-Instruct-lora/v1-20251104-033331-hf \
    --quality-metrics BLEU \
    --eval-latency-unit char \
    --sacrebleu-tokenizer zh

# infinisst-omni
module load cuda-11.8

conda create -n infinisst-omni -y python=3.12
conda activate infinisst-omni
pip install uv

# torch and related packages
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu129
uv pip install transformers==4.57.1 accelerate qwen-omni-utils jupyter gradio
uv pip install flash-attn --no-build-isolation
uv pip install pyarrow==20.0.0

uv pip install evaluate jiwer lightning deepspeed torchtune sentence-transformers wandb tensorboardX matplotlib soundfile simuleval jieba unbabel-comet simalign praat-textgrids peft

# no need for fairseq for wav2vec2 anymore
# git clone git@github.com:facebookresearch/fairseq.git
# mv fairseq fairseq-0.12.2
# cd fairseq-0.12.2
# git checkout 0.12.2-release
# uv pip install pip==23.3
# uv pip install -e .
# cd ..

# flashinfer
git clone git@github.com:flashinfer-ai/flashinfer.git
cd flashinfer
# latest v2.5.0 version has numerical instability issue
git checkout v0.2.0 
uv pip install --no-build-isolation --verbose --editable .
cd ..

# serving
uv pip install fastapi uvicorn python-multipart websockets
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

For en-de direction, stage 1 takes around 6 hours and stage 2 takes around 4.5 hours on a single node of 8 NVIDIA L40S GPUs.

## Inference

After the training is complete, you can use simuleval to perform inference on the tst-COMMON set.
You need to fill in the following variables in the `scripts/infer/infinisst.sh` script.

```bash
checkpoint_dir= # path to the stage 2 checkpoint directory
llama_path= # path to the Llama-3.1-8B-Instruct model
w2v2_path= # path to the wav2vec 2.0 model
w2v2_type= # wav2vec 2.0 type
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

After the inference is complete, you can evaluate the resulting instance log following the instructions in the [StreamLAAL](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/STREAMATT_STREAMLAAL.md#-evaluation-streamlaal).

<!-- ## Citation

If you find this work useful, please consider citing:

```bibtex
@article{ouyang2025infinisst,
  title={InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model},
  author={Ouyang, Siqi and Zhang, Yong and Zhang, Yong and Zhang, Yong},
  journal={arXiv preprint arXiv:2503.00000},
  year={2025}
}
``` -->

## Contact

If you have any questions, please feel free to raise GitHub issues or contact me at siqiouya[at]andrew.cmu.edu.