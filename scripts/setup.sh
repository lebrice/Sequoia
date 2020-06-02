#!/bin/bash

# Make the script fail whenever something bugs out or an unset variable is used or when a piped command has an error
#set -euo pipefail
#IFS=$'\n\t'

function create_load_environment(){    
    b=`pwd` # save the current directory.
    module load python/3.7

    if [[ $HOSTNAME == *"cedar"* ]]; then
        echo "Creating the environment locally on the compute node."
        virtualenv --no-download $SLURM_TMPDIR/env
        source $SLURM_TMPDIR/env/bin/activate
    elif [[ $HOSTNAME == *"blg"* ]]; then
        echo "Loading up the virtualenv at ~/ENV since we're on Beluga."
        source ~/ENV/bin/activate
    fi
    
    cd $SCRATCH/repos/SSCL
    # Install the packages that *do* need an internet connection.
    pip install -r scripts/requirements/normal.txt
    # Install the required packages that don't need to be downloaded from the internet.
    pip install -r scripts/requirements/no_index.txt	--no-index

    cd $b  # go back to the original directory.
}

function download_required_stuff(){
    b=`pwd` # save the current directory.

    cd $SCRATCH/repos/SSCL

    # Download the datasets to the $SCRATCH/data directory (if not already downloaded).
    python -m scripts.download_datasets --data_dir "$SCRATCH/data"
    
    # Download the pretrained model weights to the ~/.cache/(...) directory
    # (accessible from the compute node)
    export TORCH_HOME="$SCRATCH/.torch"
    mkdir -p $TORCH_HOME
    python -m scripts.download_pretrained_models # --save_dir "$SCRATCH/checkpoints"

    # Zip up the data folder (if it isn't already there)
    zip -u $SCRATCH/data.zip $SCRATCH/data

    # 2. Copy your dataset on the compute node
    # IMPORTANT: Your dataset must be compressed in one single file (zip, hdf5, ...)!!!
    cp --update $SCRATCH/data.zip -d $SLURM_TMPDIR
    # 3. Eventually unzip your dataset
    unzip -n $SLURM_TMPDIR/data.zip -d $SLURM_TMPDIR

    # go back to the original directory.
    cd $b
}

# Make the output directory for the slurm files, if not already present
mkdir -p $SCRATCH/slurm_out

HOSTNAME=`hostname`
echo "HOSTNAME is $HOSTNAME"

if [[ $HOSTNAME == *"cedar"* ]]; then
    echo "Running on Cedar!"
    export BELUGA=0
elif [[ $HOSTNAME == *"blg"* ]]; then
    echo "Running on Beluga!"
    export BELUGA=1
    module load httpproxy
fi

create_load_environment
download_required_stuff
git submodule init
git submodule update
