#!/bin/bash

#PBS -P H100004
###PBS -j oe
###PBS -k oed
#PBS -N mlama_overall_xlm-r-CS
#PBS -l walltime=48:00:00
#PBS -l select=1:ngpus=1
##----- CPU/Mem will be allocated at 10/200gb per GPU. -----
##----- sample config for ngpus of 2, 4, 8, 16 via either line below ----
###PBS -l select=1:ngpus=1
###PBS -l select=1:ngpus=4
###PBS -l select=1:ngpus=8
###PBS -l select=2:ngpus=8 


echo "Starting......................"
python measure_consistency_mlama.py --batch_size 32 --probed_layers 0 1 2 3 4 5 6 7 8 9 10 11 --model_name xlm-roberta-base --source_lang fr --target_lang ar --output_prefix evaluations/mlama-xlmr-consistency --beam_topk 5 --ranking_topk 5
python measure_consistency_mlama.py --batch_size 32 --probed_layers 0 1 2 3 4 5 6 7 8 9 10 11 --model_name xlm-roberta-base --source_lang fr --target_lang de --output_prefix evaluations/mlama-xlmr-consistency --beam_topk 5 --ranking_topk 5
python measure_consistency_mlama.py --batch_size 32 --probed_layers 0 1 2 3 4 5 6 7 8 9 10 11 --model_name xlm-roberta-base --source_lang fr --target_lang id --output_prefix evaluations/mlama-xlmr-consistency --beam_topk 5 --ranking_topk 5
python measure_consistency_mlama.py --batch_size 32 --probed_layers 0 1 2 3 4 5 6 7 8 9 10 11 --model_name xlm-roberta-base --source_lang fr --target_lang ta --output_prefix evaluations/mlama-xlmr-consistency --beam_topk 5 --ranking_topk 5
##nohup python  measure_consistency_mlama.py --batch_size 64 --probed_layers 0 4 8 12 16 20 24 28 32 36 40 44 47 --model_name facebook/xlm-roberta-xxl --matrix_lang en --embedded_lang ta --output_prefix evaluations/mlama-xlmr-XXL-consistency --beam_topk 5 --ranking_topk 5 &
