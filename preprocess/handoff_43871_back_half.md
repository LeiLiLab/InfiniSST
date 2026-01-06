# Hand-off: 43871 TTS back half (shards 22-31)

给 teammate 跑 jiaxuan 的 SLURM job `43871` 的后 10 个 shard (`22-31`)，跑完回传 10 个 `.jsonl` + 对应 WAV 目录。合并 / MFA / 训练由 jiaxuan 这边做。

**假设**：你已经有能跑 CosyVoice 的 conda / python 环境，这里只讲数据和 sbatch。

---

## 1. TL;DR

- Jiaxuan 会在 Slack 发你 **3 个小文件**（sbatch + worker + 这份 README）+ **1 个 JSONL**（gzip 后 ~100 MB）。
- 把 3 个文件放**同一个目录**，JSONL 放**另一个目录**，设 5 个 env var → `sbatch --array=22-31 run_tts_3variant_gs_v2_full_TEAMMATE.sh`
- 单 shard 一张卡约 14h（vLLM + batch_size=16）。
- `TEAMMATE_OUTPUT_DIR` 需要 ≥ **250 GB** 空闲。
- 回传 ~210 MB JSONL + ~150-200 GB WAV。

---

## 2. 需要准备的数据

### 2.1 Jiaxuan 会 Slack 给你（总共 4 个文件）

| 文件 | 大小 | 放哪 |
|---|---|---|
| `run_tts_3variant_gs_v2_full_TEAMMATE.sh`（本 sbatch） | ~5 KB | 任意目录，设为 `${HANDOFF_DIR}` |
| `rag_tts_multispeaker_noise.py`（TTS worker） | ~18 KB | **和 sbatch 同一个目录** (`${HANDOFF_DIR}`) |
| `handoff_43871_back_half.md`（本文档） | ~8 KB | 随便放，方便你参考 |
| `wiki_synth_utterances_3variant.jsonl.gz`（输入 utterance，gzip 后 ~100 MB） | 343 MB 解压后 | 任意目录，解压成 `.jsonl` 后指向 `${TEAMMATE_DATA}`（**注意 §6 的 quirk**） |

解压输入 JSONL：

```bash
gunzip wiki_synth_utterances_3variant.jsonl.gz
# 得到 wiki_synth_utterances_3variant.jsonl (343 MB, 2,998,703 行)
```

### 2.2 你自己搞定

| 东西 | 大小 | 怎么拿 |
|---|---|---|
| Speaker prompts pool | 2.3 GB | HF public dataset [`gavinlaw/gigaspeech_speaker_prompts_v2`](https://huggingface.co/datasets/gavinlaw/gigaspeech_speaker_prompts_v2) |
| `Fun-CosyVoice3-0.5B/` 模型 | 13 GB | 你本地应该已经有；或 HF [`FunAudioLLM/Fun-CosyVoice3-0.5B`](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B) |

下载 speaker prompts：

```bash
cd $(dirname ${TEAMMATE_SPEAKER_DIR})
hf download --repo-type dataset gavinlaw/gigaspeech_speaker_prompts_v2 \
    gigaspeech_speaker_prompts_v2.tar \
    gigaspeech_speaker_prompts_v2.tar.sha256 \
    --local-dir ./
sha256sum -c gigaspeech_speaker_prompts_v2.tar.sha256   # 必须 OK
tar -xf gigaspeech_speaker_prompts_v2.tar               # 解压出 gigaspeech_speaker_prompts/ 目录
# 然后把 TEAMMATE_SPEAKER_DIR 指向那个解压后的目录即可。
```

### 2.3 sanity check（拿到后随手看下）

- `wc -l $TEAMMATE_DATA` 应该是 **2,998,703**
- `ls $TEAMMATE_SPEAKER_DIR/*.wav | wc -l` 应该是 **9989**
- `$TEAMMATE_SPEAKER_DIR/speaker_index.json` 是 9989 条 array
- `ls ${HANDOFF_DIR}/rag_tts_multispeaker_noise.py` 能找到（sbatch 会自动去同目录找 worker）

不需要 `wham_wav/`（这轮是 CLEAN-only，`--noise-dir ""`）。

---

## 3. 环境变量

```bash
export COSYVOICE_ROOT=/path/to/your/CosyVoice
export TEAMMATE_MODEL_DIR=${COSYVOICE_ROOT}/pretrained_models/Fun-CosyVoice3-0.5B
export TEAMMATE_DATA=/path/to/wiki_synth_utterances_3variant.jsonl
export TEAMMATE_SPEAKER_DIR=/path/to/gigaspeech_speaker_prompts
export TEAMMATE_OUTPUT_DIR=/path/with/>=250GB/wiki_synth_tts_3variant_gs_v2_teammate

# 可选：如果 worker 脚本没和 sbatch 放同目录，用这个显式指定路径
# export TEAMMATE_WORKER=/abs/path/to/rag_tts_multispeaker_noise.py
```

Worker 脚本 (`rag_tts_multispeaker_noise.py`) self-contained —— 只 import CosyVoice，不依赖 InfiniSST 其它文件。sbatch 默认会去**自己所在目录**找 worker，所以 3 个小文件放一起就行，不需要 clone InfiniSST repo。

---

## 4. 启动

**先冒烟测一个 shard**（确认你机器上 CosyVoice + 数据都 OK）：

```bash
cd ${HANDOFF_DIR}
sbatch -p <your_partition> -o <log>/%A-%a.out -e <log>/%A-%a.err \
  --array=31 run_tts_3variant_gs_v2_full_TEAMMATE.sh
```

跑完检查：
- `$(dirname ${TEAMMATE_DATA})/wiki_synth_3variant_gs_v2_clean_with_tts_shard31.jsonl` 生成，行数 > 90,000
- `${TEAMMATE_OUTPUT_DIR}/clean/0XXX/*.wav` 有文件，16 kHz mono

**批量**（在同一个 `${HANDOFF_DIR}` 里跑）：

```bash
sbatch -p <your_partition> -o <log>/%A-%a.out -e <log>/%A-%a.err \
  --array=22-31 run_tts_3variant_gs_v2_full_TEAMMATE.sh
```

GPU 少的话限流：`--array=22-31%4`（最多并发 4 个 shard）。

---

## 5. Hard constraints（错一个数据就废）

1. `--num-shards 32` 固定，**不要**因为只跑 10 个 shard 就改成 10。sharding 是 `range(shard_id, 2998703, 32)`，改了会和前半 shard 0-21 撞号或漏数据。
2. `--shard-id` 必须等于 `SLURM_ARRAY_TASK_ID`。sbatch 里已经这么写，不要把 22-31 remap 成 0-9。
3. `--array=22-31`。前半 14-21 是 jiaxuan 的。脚本顶部有 `SHARD_ID < 22 || > 31 → exit 2` 的守卫，是保护层。
4. `--no_dedup` 必须传（sbatch 里已经传了）。默认去重会把 3M 压到 ~1M。
5. `--noise-dir ""` 必须空。CLEAN-only 是这轮的改进点。
6. `OUTPUT_JSONL_PREFIX=wiki_synth_3variant_gs_v2_clean_with_tts` 不要改。jiaxuan 这边 merge 脚本按这个前缀找 32 个 shard。
7. `--no-load-trt` 必须传（已传）。别开 TRT，除非你愿意等 15-25 min 首次编译。

---

## 6. Quirk: JSONL 输出**不在** `--output-dir`

`rag_tts_multispeaker_noise.py` 把 shard jsonl 写到：

```
os.path.dirname(TEAMMATE_DATA) + / + wiki_synth_3variant_gs_v2_clean_with_tts_shard{N}.jsonl
```

也就是 jsonl 落到**你 `${TEAMMATE_DATA}` 所在的目录**，不是 `${TEAMMATE_OUTPUT_DIR}`。所以回传时要从**两个目录**各拿一份：

- `$(dirname ${TEAMMATE_DATA})/wiki_synth_3variant_gs_v2_clean_with_tts_shard{22..31}.jsonl`（10 个，~210 MB）
- `${TEAMMATE_OUTPUT_DIR}/clean/`（整个目录，~150-200 GB）

没法用 CLI 参数改（脚本里硬编码的 `dirname(args.data)`）。最省事：把 `${TEAMMATE_DATA}` 放到你希望 jsonl 落地的那个目录里。

---

## 7. 回传给 jiaxuan

1. **10 个 shard JSONL** `wiki_synth_3variant_gs_v2_clean_with_tts_shard{22..31}.jsonl`（各 ~21 MB，`wc -l` 各 ≈ 94,000 行）。
2. **WAV 目录** `${TEAMMATE_OUTPUT_DIR}/clean/`（~150-200 GB）。推荐 `rsync -av --info=progress2 --partial`（断线续传）或 `tar` 后走 S3 / 内网盘。
3. **一个 run log 小卡片** `teammate_run_log.txt`：
   - `TEAMMATE_OUTPUT_DIR` 的**绝对路径**（jiaxuan 做 clean_audio_path 前缀重写要用）
   - `sacct -j <job_id> --format=JobID%-20,State%-12,Elapsed,ExitCode` 输出
   - `nvidia-smi -L`（GPU 型号，万一结果异常要对照）

---

## 8. 断点续传 & 常见坑

- **shard 跑一半机器挂了**：直接 `sbatch --array=<那个 shard>` 重跑。worker 有 `os.path.exists(clean_path) → skipped`，不会重复算。
- **vLLM OOM**：把 sbatch 里的 `BATCH_SIZE=16` 降到 `8` 或 `4`。
- **单条 utterance 报 CosyVoice 前端错**：正常，会被计入 `failed`。如果 `failed > 100` 再贴 log 给 jiaxuan。
- **某 shard `failed` / `TIMEOUT`**：看 err log；默认 `--time=2-12:00:00`（60h）理论很宽松，超时通常意味着 vLLM 卡死，重跑一次即可。

---

## 附录 A: Slack 发给你的 4 个文件

| 文件名 | 用途 | 你放哪 |
|---|---|---|
| `run_tts_3variant_gs_v2_full_TEAMMATE.sh` | sbatch 入口 | `${HANDOFF_DIR}` |
| `rag_tts_multispeaker_noise.py` | TTS worker，被 sbatch 调用 | `${HANDOFF_DIR}`（**必须和 sbatch 同目录**，否则用 `TEAMMATE_WORKER` 显式指） |
| `handoff_43871_back_half.md` | 本文档 | 任意 |
| `wiki_synth_utterances_3variant.jsonl.gz` | 输入 utterance | 任意，`gunzip` 后指向 `${TEAMMATE_DATA}` |

## 附录 B: jiaxuan 这边的辅助工具（你不用管）

- `rewrite_teammate_wav_paths.py` — jiaxuan 收到你回传的 jsonl 后做 `clean_audio_path` 前缀重写
- `run_tts_3variant_gigaspeech_full_taurus.sh` — 原版 taurus sbatch（43871 本体），仅参考
