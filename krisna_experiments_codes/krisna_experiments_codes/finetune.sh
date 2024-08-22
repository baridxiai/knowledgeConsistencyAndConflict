#!/bin/bash

#SBATCH --job-name=xnli-xlmr-base# Job name
#SBATCH --error=./logs/%j%x.err # error file
#SBATCH --output=./logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=16000 # 16 GB of RAM
#SBATCH --nodelist=ws-l6-012


echo "Starting......................"
python finetune.py --model_name xlm-roberta-base --training_config_json configs/xlmr-base-xnli-config.json
