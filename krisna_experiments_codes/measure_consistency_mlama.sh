#!/bin/bash

#SBATCH --job-name=mlama-consistency-ar-en-xlmr # Job name
#SBATCH --error=./logs/%j%x.err # error file
#SBATCH --output=./logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=16000 # 16 GB of RAM
#SBATCH --nodelist=ws-l6-007


echo "Starting......................"
python measure_consistency_mlama.py --batch_size 32 --probed_layers 0 1 2 3 4 5 6 7 8 9 10 11 --model_name xlm-roberta-base --source_lang en --target_lang ar --output_prefix evaluations/mlama-xlmr-consistency --beam_topk 5 --ranking_topk 5
