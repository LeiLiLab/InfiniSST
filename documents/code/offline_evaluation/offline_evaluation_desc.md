我需要一个在gigaspeech上recall随着k1变化的图, 看k1达到多大值后recall趋近于饱和.
新增个脚本, 用gigaspeech, 数据集来源是:
DEV_JSONL="/mnt/gemini/data1/jiaxuanluo/term_dev_dataset_final.jsonl"
根据这个测试集数据, 过滤掉term为空的samples, 提取terms作为glossary, 然后去判断对于这个1.92秒的chunk audio来说, recall@K1, K1达到多少时, recall命中率开始趋近饱和.
recall hit的代码可以参考/home/jiaxuanluo/InfiniSST/retriever/gigaspeech/qwen3_AuT_BGE_M3_train_lora_unfrozen_text.py
