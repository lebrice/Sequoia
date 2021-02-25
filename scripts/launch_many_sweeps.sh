#!/bin/bash

# module load anaconda/3
# conda activate sequoia

cd ~/Sequoia
# pip install -e .[hpo,monsterkong]


settings="incremental_rl task_incremental_rl rl"
nb_tasks="10 20"
methods="baseline ewc" #a2c ppo dqn"
for setting in $settings ;do
for nb_task in $nb_tasks ;do
for method in $methods ;do
sbatch scripts/sweep.sh --setting $setting --dataset monsterkong --steps_per_task 100_000 --nb_tasks $nb_task --method $method
done #methods
done #nb_tasks 
done #settings