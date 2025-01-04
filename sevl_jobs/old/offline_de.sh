export PYTHONPATH=/home/siqiouya/work/sllama
bash eval/offline.sh \
    /compute/babel-5-23/siqiouya/runs/3.1-8B-s2-english-german-w2v2-rope-freeze-noxpos/checkpoint-1760/ \
    /data/user_data/siqiouya/runs/pretrained/wav2_vec_vox_960h_pl.pt \
    w2v2 \
    /compute/babel-6-17/xixu/datasets/must-c-v1.0/en-de \
    English German \
    0