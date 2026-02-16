/home/jiaxuanluo/InfiniSST/documents/code/train/term_train/tt_term_train_aries.sh
这个是老脚本, 需要改两个逻辑:
1.train dataset读取/mnt/gemini/data/siqiouyang/term_train_dataset_final_with_tts.jsonl, 关注tts_audio_path字段, 来取tts的audio. 例如:
{"term": "product team", "term_key": "product team", "chunk_src_text": "leader of the product team for", "utter_id": "POD0000006330_2", "chunk_idx": 4, "chunk_audio_path": "/mnt/gemini/data1/jiaxuanluo/term_train_audio_chunks/POD0000006330_2_chunk_4.wav", "tts_audio_path": "/mnt/gemini/data/siqiouyang/term_train_tts/POD0000006330/2/chunk_4_0.wav"}

2.eval_dataset用/mnt/gemini/data/siqiouyang/term_dev_dataset_final_with_tts.jsonl
同样是去读tts_audio_path.