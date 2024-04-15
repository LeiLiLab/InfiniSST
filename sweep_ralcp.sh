src_segment_sizes=(960 2000)
min_start_secs=(2.0)
beam=(4)

for src_segment_size in "${src_segment_sizes[@]}"; do
    for min_start_sec in "${min_start_secs[@]}"; do
        sbatch sevl_de_ralcp.sh $src_segment_size $beam $min_start_sec
    done
done