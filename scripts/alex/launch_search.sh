#!/bin/bash
#SBATCH --gres=gpu:1                           # Ask for 1 GPU --gres=gpu:1,titanxp:1
#SBATCH --mem=20G                                  # Ask for 10 GB of RAM
#SBATCH --time=32:00:00                            # The job will run for 3 hours
##SBATCH -o log/slurm-%N-%j.out  # Write the log on tmp1
#SBATCH --output=logs/job_output_$USER.txt
#SBATCH --error=logs/job_error_$USER.txt
#SBATCH -p long                                    # --partition=unkillable, long
#SBATCH --cpus-per-task=2                     	   # Ask for 2 CPUs


source $CONDA_ACTIVATE
mkdir /network/home/ostapeno/dev/SSCL/results/wandb_$USER
export WANDB_DIR=/network/home/ostapeno/dev/SSCL/results/wandb_$USER
conda activate /network/home/ostapeno/.conda/envs/falr
python -u hp_search.py