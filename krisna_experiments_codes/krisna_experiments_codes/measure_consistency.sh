#!/bin/bash

#SBATCH --job-name=consistency-de-en-xnli # Job name
#SBATCH --error=./logs/%j%x.err # error file
#SBATCH --output=./logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=16000 # 16 GB of RAM
#SBATCH --nodelist=ws-l6-006


echo "Starting......................"
python measure_consistency.py --task_type nli --checked_dataset_file data/xnli-test/xnli_test_random-cm-de-en.pkl --pivot_dataset_file data/xnli-test/xnli_test_en-de_filtered.pkl --batch_size 32 --model_name MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 --pivot_lang en --probed_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 --metrics avg_acc --output_file ./evaluations/xnli-mdeberta-consistency_random-cm-de-en.pkl
