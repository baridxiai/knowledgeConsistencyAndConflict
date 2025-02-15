#!/bin/bash

#SBATCH --job-name=mlama-subject-alignments # Job name
#SBATCH --error=./logs/%j%x.err # error file
#SBATCH --output=./logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=16000 # 16 GB of RAM
#SBATCH --nodelist=ws-l4-020


echo "Starting......................"
echo "Decoder"
python measure_subjects_alignment.py --batch_size 8 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --model_name meta-llama/Llama-3.1-8B --matrix_lang en --embedded_lang de --model_type decoder --output_prefix analysis/mlama-llama3.1-8b-sim
python measure_subjects_alignment.py --batch_size 8 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --model_name meta-llama/Llama-3.1-8B --matrix_lang en --embedded_lang id --model_type decoder --output_prefix analysis/mlama-llama3.1-8b-sim
python measure_subjects_alignment.py --batch_size 8 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --model_name meta-llama/Llama-3.1-8B --matrix_lang en --embedded_lang ar --model_type decoder --output_prefix analysis/mlama-llama3.1-8b-sim
python measure_subjects_alignment.py --batch_size 8 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --model_name meta-llama/Llama-3.1-8B --matrix_lang en --embedded_lang ta --model_type decoder --output_prefix analysis/mlama-llama3.1-8b-sim
python measure_subjects_alignment.py --batch_size 8 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --model_name meta-llama/Llama-3.1-8B --matrix_lang en --embedded_lang baseline-decoder --model_type decoder --output_prefix analysis/mlama-llama3.1-8b-sim
python measure_subjects_alignment.py --batch_size 8 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --model_name meta-llama/Llama-3.1-8B-Instruct --matrix_lang en --embedded_lang ar --model_type decoder --output_prefix analysis/mlama-llama3.1-8b-instruct-sim
python measure_subjects_alignment.py --batch_size 8 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --model_name meta-llama/Llama-3.1-8B-Instruct --matrix_lang en --embedded_lang ta --model_type decoder --output_prefix analysis/mlama-llama3.1-8b-instruct-sim
python measure_subjects_alignment.py --batch_size 8 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --model_name meta-llama/Llama-3.1-8B-Instruct --matrix_lang en --embedded_lang baseline-decoder --model_type decoder --output_prefix analysis/mlama-llama3.1-8b-instruct-sim
python measure_subjects_alignment.py --batch_size 16 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --model_name meta-llama/Llama-3.2-1B --matrix_lang en --embedded_lang de --model_type decoder --output_prefix analysis/mlama-llama3.2-1b-sim
python measure_subjects_alignment.py --batch_size 16 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --model_name meta-llama/Llama-3.2-1B --matrix_lang en --embedded_lang id --model_type decoder --output_prefix analysis/mlama-llama3.2-1b-sim
python measure_subjects_alignment.py --batch_size 16 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --model_name meta-llama/Llama-3.2-1B --matrix_lang en --embedded_lang ar --model_type decoder --output_prefix analysis/mlama-llama3.2-1b-sim
python measure_subjects_alignment.py --batch_size 16 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --model_name meta-llama/Llama-3.2-1B --matrix_lang en --embedded_lang ta --model_type decoder --output_prefix analysis/mlama-llama3.2-1b-sim
python measure_subjects_alignment.py --batch_size 16 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --model_name meta-llama/Llama-3.2-1B --matrix_lang en --embedded_lang baseline-decoder --model_type decoder --output_prefix analysis/mlama-llama3.2-1b-sim

echo "Encoder-Decoder"
python measure_subjects_alignment.py --batch_size 16 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 --model_name bigscience/mt0-large --matrix_lang en --embedded_lang de --model_type encoder-decoder --output_prefix analysis/mlama-mt0-large-sim
python measure_subjects_alignment.py --batch_size 16 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 --model_name bigscience/mt0-large --matrix_lang en --embedded_lang id --model_type encoder-decoder --output_prefix analysis/mlama-mt0-large-sim
python measure_subjects_alignment.py --batch_size 16 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 --model_name bigscience/mt0-large --matrix_lang en --embedded_lang ar --model_type encoder-decoder --output_prefix analysis/mlama-mt0-large-sim
python measure_subjects_alignment.py --batch_size 16 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 --model_name bigscience/mt0-large --matrix_lang en --embedded_lang ta --model_type encoder-decoder --output_prefix analysis/mlama-mt0-large-sim
python measure_subjects_alignment.py --batch_size 16 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 --model_name bigscience/mt0-large --matrix_lang en --embedded_lang baseline-encoder-decoder --model_type encoder-decoder --output_prefix analysis/mlama-mt0-large-sim

python measure_subjects_alignment.py --batch_size 32 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 --model_name bigscience/mt0-base --matrix_lang en --embedded_lang de --model_type encoder-decoder --output_prefix analysis/mlama-mt0-base-sim
python measure_subjects_alignment.py --batch_size 32 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 --model_name bigscience/mt0-base --matrix_lang en --embedded_lang id --model_type encoder-decoder --output_prefix analysis/mlama-mt0-base-sim
python measure_subjects_alignment.py --batch_size 32 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 --model_name bigscience/mt0-base --matrix_lang en --embedded_lang ar --model_type encoder-decoder --output_prefix analysis/mlama-mt0-base-sim
python measure_subjects_alignment.py --batch_size 32 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 --model_name bigscience/mt0-base --matrix_lang en --embedded_lang ta --model_type encoder-decoder --output_prefix analysis/mlama-mt0-base-sim
python measure_subjects_alignment.py --batch_size 32 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 --model_name bigscience/mt0-base --matrix_lang en --embedded_lang baseline-encoder-decoder --model_type encoder-decoder --output_prefix analysis/mlama-mt0-base-sim

python measure_subjects_alignment.py --batch_size 16 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 --model_name google/mt5-large --matrix_lang en --embedded_lang ar --model_type encoder-decoder --output_prefix analysis/mlama-mt5-large-sim
python measure_subjects_alignment.py --batch_size 16 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 --model_name google/mt5-large --matrix_lang en --embedded_lang ta --model_type encoder-decoder --output_prefix analysis/mlama-mt5-large-sim
python measure_subjects_alignment.py --batch_size 16 --selected_layers 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 --model_name google/mt5-large --matrix_lang en --embedded_lang baseline-encoder-decoder --model_type encoder-decoder --output_prefix analysis/mlama-mt5-large-sim

echo "Finished"