#!/bin/bash

# Create output_logs directory if it doesn't exist
mkdir -p output_logs

# Loop through each .txt file in the ../10core directory
for input_file in ../10core/*.txt; do
    filename=$(basename "$input_file" .txt)
    
    log_file="output_logs/${filename}_log.txt"
    
    # Run the command with the input_file and redirect output to log file
    python run.py \
        --input_files "$input_file" \
        --output_dir ./output \
        --do_train \
        --do_eval \
        --eval_strategy epoch \
        --per_gpu_train_batch_size 256 \
        --per_gpu_eval_batch_size 256 \
        --num_train_epochs 100 \
        --logging_steps 500 \
        --hidden_size 64 \
        &> "$log_file"
    
    echo "Finished processing $input_file. Log saved to $log_file."
done

