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

cd /home/svu/t0932723/workspace/alpha/knowledgeConsistencyAndConflict/experiments_codes
sh batch_llama-1b-ig2-4.sh
