@tts_gigaspeech_threshold_ablation_k1_10.py @tts_sbatch_gigaspeech_threshold_ablation_k1_10_aries.sh 目前这些脚本不还是之前那套speech_utt encode后跟bge m3的text terms做cosine similarity, 需要做如下修改:
1.term retriever模型文件要换成:/mnt/gemini/data/jiaxuanluo/q3rag_tts_lora-r32-tr16_bs4k_ttsw0.5_ttm=query key value_temperature=0.03_epoch_16.pt
2.超参数配置支持只对text 模式做recall和threshold的similarity score; 只对term_tts模式做recall和threshold;两者取交集(也就是只有都出现才算recall, 来提升精度), 再sweep threshold.

如果需要数据读取等逻辑, 可以参考训练脚本:
/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/tt_term_train_aries.sh
