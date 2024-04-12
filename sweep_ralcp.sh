src_segment_sizes=(320 640 960 1280)
min_start_secs=(1.0 1.5 2.0 2.5)

for src_segment_size in "${src_segment_sizes[@]}"; do
    for min_start_sec in "${min_start_secs[@]}"; do
        sbatch sevl_de_ralcp.sh $src_segment_size $min_start_sec
    done
done