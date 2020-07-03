#!/bin/bash
#SBATCH -p long
#SBATCH --gres=gpu:2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=2
#SBATCH --mem=16G
#SBATCH --time=0-02:00:00
#SBATCH --cpus-per-task=4              # Cores proportional to GPUs

module load anaconda/3
conda activate pytorch

cd ~/SSCL

python setups/mnist.py