#!/bin/bash

#SBATCH --job-name=inference_baseline-en_de-en# Job name
#SBATCH --error=./logs/%j%x.err # error file
#SBATCH --output=./logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=16000 # 16 GB of RAM
#SBATCH --nodelist=ws-l6-012


echo "Starting......................"
python inference.py --dataset_file data/mkqa-test/mkqa_test_en-queries_de-en-answers_filtered.pkl --model_name bigscience/mt0-base --probed_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 --source_lang en --target_lang de
