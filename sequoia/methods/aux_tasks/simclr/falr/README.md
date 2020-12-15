## Setup
```bash
# Conda setup (on Mila cluster)
module load anaconda/3
conda-activate

# Build environment (only need to do once)
conda env create -f environment.yml
conda activate falr

# Prepare data (on Mila cluster compute node)
chmod 700 prepare.sh
./prepare.sh

# Run experiment (on Mila cluster compute node)
python run.py
```

## Metrics
Experiment data is logged to wandb by default: https://app.wandb.ai/nitarshan/falr
