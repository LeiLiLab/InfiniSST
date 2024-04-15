src_segment_sizes=(2000 2500)
min_start_secs=(2.0)
beams=(4 10)

for src_segment_size in "${src_segment_sizes[@]}"; do
    for min_start_sec in "${min_start_secs[@]}"; do
        for beam in "${beams[@]}"; do
            sbatch sevl_de_holdn.sh $src_segment_size $beam $min_start_sec 
        done
    done
done