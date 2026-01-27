# InfiniSST — High-Throughput Streaming Speech Translation Version

## TL;DR (What This System Solves)

InfiniSST is a **streaming speech translation system** that must operate on **unbounded audio streams** while keeping:

- **Low, deterministic latency** for real-time UX
- **High throughput** under multi-session load
- **Memory safety** when context grows without a fixed upper bound

The core challenge is **KV-cache growth** and **scheduling** under streaming constraints: unlike offline ASR/MT, the system cannot assume a known input length or a clean "end of utterance."

## Systems Contributions

I contributed as a **core systems engineer**, focusing on **optimization** and **inference engine design**:

- **Custom streaming inference engine on Ray**
  - Built a multi-tenant, GPU-efficient serving loop tailored for streaming speech-to-text translation workloads.
- **Paged Attention via FlashInfer kernels + Python memory manager**
  - Implemented a **specialized Python-based memory manager** to coordinate page allocation/reuse and support **up to 32 concurrent sessions per GPU** (configurable by workload and hardware).
- **Dynamic KV-cache eviction for unbounded streams**
  - Designed an eviction protocol to keep memory bounded while preserving translation quality under long-context conditions.
- **Deterministic sub-200ms latency by reducing serving overhead**
  - Achieved predictable latency by **bypassing generic LLM serving abstractions** and executing fine-grained **read/write policies** directly in the critical path.

## System Architecture (Bird’s-Eye View)

At a high level, the system is organized as a streaming pipeline:

```
Audio Stream
   |
   v
Speech Encoder (Wav2Vec2 / variants)
   |
   v
Streaming Policy / Agent (SimulEval-compatible)
   |
   v
LLM Decoder (Llama-family) + KV-Cache
   |
   v
Incremental Target Tokens -> UI / Log / Evaluation
```

The **streaming agent** controls:

- when to **READ** more audio frames
- when to **WRITE** target tokens
- how to allocate/evict KV state per session when the stream is effectively infinite

## Design Notes (System-Aware Highlights)

### Multi-Tenant GPU Serving on Streaming Workloads

Streaming translation is not just "LLM serving with smaller prompts":

- the system must interleave **many partial sequences** (sessions) on the same GPU
- each session has its own continuously growing context/state
- the scheduler must trade off **fairness**, **tail latency**, and **GPU utilization**

Ray provides a solid substrate for orchestration, but high-performance streaming requires minimizing per-request overhead and controlling the execution granularity.

### Paged Attention + Custom Memory Management

To make long-context streaming practical under concurrency, we use **Paged Attention**:

- KV-cache is stored in **fixed-size pages**
- the engine allocates and reuses pages across sessions
- page-level management enables predictable memory behavior and reduces fragmentation

FlashInfer kernels are used to accelerate attention on paged KV layouts; a Python-side memory manager coordinates session ownership, lifecycle, and eviction triggers.

### Dynamic KV-Cache Eviction for Unbounded Streams

In unbounded streams, "keep all KV forever" is not viable. We implement a **dynamic eviction protocol**:

- evict based on an explicit policy (read/write dynamics, session pressure, and bounded budget)
- keep the system stable under bursty workloads
- preserve translation quality by prioritizing the most useful context segments

### Deterministic Latency via Policy-First Execution

Generic LLM servers typically optimize for throughput but introduce variable overhead (routing, batching, framework layers).
For streaming speech translation, we prioritize **deterministic latency**:

- fine-grained policy execution (READ/WRITE) runs on a tight loop
- avoid unnecessary layers in the critical path
- keep scheduling decisions close to the data and KV state

## Repository Map (Where to Look First)

- **`agents/`**: streaming agents/policies (e.g., InfiniSST, StreamAtt, AlignAtt)
- **`model/`**: LLM + speech encoder integration and patches
  - **`model/patches/`**: hooks/patches for attention, LLM behavior, and speech encoder integration
- **`train/`**: dataset, training entrypoints, and utilities
- **`preprocess/`**: data preparation pipeline (ASR, filtering, alignment, SimulEval inputs)
- **`scripts/`**: SLURM-friendly training/inference scripts
- **`plots/`**: figures used for analysis (quality/latency, RTF, context length, etc.)

## Demos

### Online Demo

The link to the online demo is [here](https://infinisst.ngrok.app/).

### macOS Desktop Demo (Apple Silicon)

1. Download `InfiniSST Translation-1.0.0-arm64.dmg` from this repository.
2. Install the application and run the following command to bypass macOS certificate verification:

```bash
xattr -d com.apple.quarantine "/Applications/InfiniSST Translation.app"
```

3. Once launched, the translation window can float above other desktop applications (except those in full-screen mode).

![image](https://github.com/user-attachments/assets/552eafd2-5d22-4678-9ebf-9bd4951902b5)

## Checkpoints

We provide checkpoints for three language directions: English-German (en-de), English-Spanish (en-es), and English-Chinese (en-zh).
The "Offline" checkpoints correspond to the checkpoints used in StreamAtt and AlignAtt, while the "InfiniSST" checkpoints are for our proposed model.

| Language Direction | Offline | InfiniSST |
|--------------------|-------------------|----------------------|
| en-de              | [pytorch_model.bin](https://f005.backblazeb2.com/file/owaski-release/ckpts/infinisst/must-c/en-de/8B-s2-bi-v3.5/last.ckpt/pytorch_model.bin) | [pytorch_model.bin](https://f005.backblazeb2.com/file/owaski-release/ckpts/infinisst/must-c/en-de/8B-traj-s2-v3.6/pytorch_model.bin) |
| en-es              | [pytorch_model.bin](https://f005.backblazeb2.com/file/owaski-release/ckpts/infinisst/must-c/en-es/8B-s2-bi-v3.5/last.ckpt/pytorch_model.bin) | [pytorch_model.bin](https://f005.backblazeb2.com/file/owaski-release/ckpts/infinisst/must-c/en-es/8B-traj-s2-v3.6/pytorch_model.bin) |
| en-zh              |  | [pytorch_model.bin](https://f005.backblazeb2.com/file/owaski-release/ckpts/infinisst/must-c/en-zh/8B-traj-s2-v3.6/pytorch_model.bin) |

You can download the checkpoints from the links above and use them for evaluation or further experiments.

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

You also need to login wandb with `wandb login` to use the `wandb` package.

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

After training completes, you can use SimulEval to perform inference on the tst-COMMON set.
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

Then you can run the following script:

```bash
sbatch scripts/infer/infinisst.sh
```

## Evaluation with StreamLAAL

After inference completes, you can evaluate the resulting instance log following the instructions in [StreamLAAL](https://github.com/hlt-mt/FBK-fairseq/blob/master/fbk_works/STREAMATT_STREAMLAAL.md#-evaluation-streamlaal).

<!--
## Citation

If you find this work useful, please consider citing:

```bibtex
@article{ouyang2025infinisst,
  title={InfiniSST: Simultaneous Translation of Unbounded Speech with Large Language Model},
  author={Ouyang, Siqi and Zhang, Yong and Zhang, Yong and Zhang, Yong},
  journal={arXiv preprint arXiv:2503.00000},
  year={2025}
}
```
-->

## Contact

If you have any questions, please feel free to raise GitHub issues or contact me at jluo50@jh.edu
