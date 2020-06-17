#!/bin/bash
#SBATCH --gres=gpu::1                       # Ask for 1 GPU --gres=gpu:1,titanxp:1
#SBATCH --mem=25G                                  # Ask for 10 GB of RAM
#SBATCH --time=48:00:00                            # The job will run for 3 hours
#SBATCH --output=logs/job_output.txt
#SBATCH --error=logs/job_error.txt
<<<<<<< HEAD
#SBATCH -p long                                  # --partition=unkillable, long
=======
#SBATCH -p main                                    # --partition=unkillable, long
>>>>>>> temp_ssl
#SBATCH --cpus-per-task=2                     	   # Ask for 2 CPUs


source $CONDA_ACTIVATE
conda activate /network/home/ostapeno/.conda/envs/falr
#cd /network/home/ostapeno/dev/SSCL

mkdir /network/home/ostapeno/dev/SSCL_copy/results/wandb_$USER
export WANDB_DIR=/network/home/ostapeno/dev/SSCL_copy/results/wandb_$USER


#function cleanup(){
    #echo "Cleaning up and transfering files from $SLURM_TMPDIR to $SCRATCH/SSCL"
    #rsync -r -u $SCRATCH/SSCL/* $SCRATCH/SSCL
    ##wandb sync $WANDB_DIR # Not guaranteed to work given CC's network restrictions.
#}

#trap cleanup EXIT
echo "Calling python -u main.py task-incremental \
    --data_dir $SLURM_TMPDIR/data \
    --log_dir_root $SCRATCH/results \
    --run_number ${SLURM_ARRAY_TASK_ID:-0} \
    '${@:1}'"


python -u ../../main.py task-incremental-semi-sup \
    --data_dir $SLURM_TMPDIR/data \
<<<<<<< HEAD
    --log_dir_root /network/home/ostapeno/dev/SSCL \
=======
    --log_dir_root /network/home/ostapeno/dev/SSCL_copy \
>>>>>>> temp_ssl
    --run_number ${SLURM_ARRAY_TASK_ID:-0} \
    "${@:1}"

exit

