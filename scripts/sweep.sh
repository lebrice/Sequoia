#!/bin/bash
#SBATCH --array=0-10%2
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --mem=10GB
#SBATCH --time=11:59:00
#SBATCH -p long
#SBATCH -o logs/slurm-%j.out

# module load anaconda/3
# conda activate sequoia

cd ~/dev/Sequoia

module purge  
module load python/3.7    
module load python/3.7/cuda/11.0/cudnn/8.0/pytorch/1.6.0
source /home/mila/o/ostapeno/ENV/SSCL/bin/activate

pip install -e .[hpo,monsterkong]
# TODO: Change the setting, the number of tasks, the method, etc.
sequoia_sweep "$@"