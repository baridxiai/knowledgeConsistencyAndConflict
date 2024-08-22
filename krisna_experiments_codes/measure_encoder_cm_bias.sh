#!/bin/bash

#SBATCH --job-name=mlp-bias-mt0-large-de-en # Job name
#SBATCH --error=./logs/%j%x.err # error file
#SBATCH --output=./logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=16000 # 16 GB of RAM
#SBATCH --nodelist=ws-l6-007


echo "Starting......................"
python measure_encoder_cm_bias.py  --probed_layers 8 9 10 11 12 13 14 15 --model_name bigscience/mt0-large --source_lang en --target_lang de --output_prefix ./analysis/mlama-mt0-large-08-15 --model_type encoder-decoder
 
