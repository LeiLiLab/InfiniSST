dirname=$1
seg_src_path=/data/user_data/siqiouya/dataset/must-c-v1.0/en-de/tst-COMMON_full_segmented.source
seg_ref_path=/data/user_data/siqiouya/dataset/must-c-v1.0/en-de/tst-COMMON_full_segmented.target

source /home/siqiouya/anaconda3/bin/activate sllama_lightning
cd /home/siqiouya/work/sllama/eval
python pre_seg.py \
    --dir $dirname \
    --segmented-refs $seg_ref_path

source /home/siqiouya/anaconda3/bin/activate py2
cd /home/siqiouya/download/mwerSegmenter

for hyp_file in $dirname/tmp/hyp.*; do
    number="${hyp_file##*.}"
    ref_file="$dirname/tmp/ref.$number"
    echo $hyp_file
    ./mwerSegmenter -hypfile $hyp_file -MRefFile $ref_file
    mv __segments "$dirname/tmp/hyp.$number.seg"
done

source /home/siqiouya/anaconda3/bin/activate sllama_lightning
cd /home/siqiouya/work/sllama/eval
python post_seg.py \
    --dir $dirname \
    --segmented-srcs $seg_src_path \
    --segmented-refs $seg_ref_path