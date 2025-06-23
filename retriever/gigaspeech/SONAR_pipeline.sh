LOG_FILE="logs/pipeline_$(date +%Y%m%d_%H%M%S).log"
# === 2. Preprocess and save training samples
sample_save_job=$(sbatch samples_pre_handle.sh l term  | awk '{print $4}')
echo "samples_pre_handle: $sample_save_job" >> "$LOG_FILE"

# === 3. Merge training samples
merge_samples=$(sbatch --dependency=afterok:$sample_save_job merge_samples.sh l term | awk '{print $4}')
echo "merge_samples: $merge_samples" >> "$LOG_FILE"

# === 4. Train the model
train=$(sbatch --dependency=afterok:$merge_samples train.sh l term | awk '{print $4}')
echo "train: $train" >> "$LOG_FILE"
