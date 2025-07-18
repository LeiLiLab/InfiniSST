#!/bin/bash

timestamp=$(date "+%Y-%m-%d_%H-%M-%S")
LOG_FILE="slurm_job_chain_$timestamp.log"
echo "🔗 Job submission log - $timestamp" > $LOG_FILE

# === 1. Split glossary into 4 chunks (by short_description field)
split_glossary=$(sbatch split_glossary.sh short_description | awk '{print $4}')
echo "split_glossary: $split_glossary" >> $LOG_FILE

# === 2. Preprocess and save training samples
sample_save_job=$(sbatch samples_pre_handle.sh m | awk '{print $4}')
echo "samples_pre_handle: $sample_save_job" >> $LOG_FILE

# === 3. Merge training samples
merge_samples=$(sbatch --dependency=afterok:$sample_save_job merge_samples.sh m | awk '{print $4}')
echo "merge_samples: $merge_samples" >> $LOG_FILE

# === 4. Train the model
train=$(sbatch --dependency=afterok:$merge_samples train.sh m | awk '{print $4}')
echo "train: $train" >> $LOG_FILE

# === 5. Embed glossary chunks
glossary_embedding=$(sbatch --dependency=afterok:$train:$split_glossary glossary_embedding.sh short_description | awk '{print $4}')
echo "glossary_embedding: $glossary_embedding" >> $LOG_FILE

# === 6. Build FAISS index
build_glossary_index=$(sbatch --dependency=afterok:$glossary_embedding build_glossary_index.sh short_description | awk '{print $4}')
echo "build_glossary_index: $build_glossary_index" >> $LOG_FILE

# === 7. Generate test samples
generate_test_sample=$(sbatch generate_test_sample.sh short_description | awk '{print $4}')
echo "generate_test_sample: $generate_test_sample" >> $LOG_FILE

# === 8. Evaluate retrieval
evaluate=$(sbatch --dependency=afterok:$generate_test_sample:$build_glossary_index evaluate.sh short_description | awk '{print $4}')
echo "evaluate: $evaluate" >> $LOG_FILE

echo "✅ All jobs submitted successfully. Job chain log saved to: $LOG_FILE"