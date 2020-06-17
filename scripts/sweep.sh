#!/bin/bash

NAME="${1:?'Name must be set'}"
N_JOBS="${2:?'N_JOBS must be set'}"

OUT="$SCRATCH/slurm_out/$NAME/%x-%j.out"
ARGS=${@:3}

echo "Sweep with name '$NAME' and with args '$ARGS'"
# Create the slurm output dir if it doesn't exist already.

# activate the virtual environment (only used to download the datasets)
source ~/ENV/bin/activate
python -m scripts.download_datasets --data_dir "$SCRATCH/data"
python -m scripts.download_pretrained_models # --save_dir "$SCRATCH/checkpoints"
deactivate

zip -u "$SCRATCH/data.zip" "$SCRATCH/data"

mkdir -p "$SCRATCH/slurm_out/$NAME"

./scripts/sbatch_job.sh "${NAME}_baseline"           $N_JOBS --run_name "${NAME}_baseline"           $ARGS
./scripts/sbatch_job.sh "${NAME}_irm_001"            $N_JOBS --run_name "${NAME}_irm_001"            $ARGS --irm.coef 0.01
./scripts/sbatch_job.sh "${NAME}_irm_01"             $N_JOBS --run_name "${NAME}_irm_01"             $ARGS --irm.coef 0.1
./scripts/sbatch_job.sh "${NAME}_irm_1"              $N_JOBS --run_name "${NAME}_irm_1"              $ARGS --irm.coef 1
./scripts/sbatch_job.sh "${NAME}_mixup_001"          $N_JOBS --run_name "${NAME}_mixup_001"          $ARGS --mixup.coef 0.01
./scripts/sbatch_job.sh "${NAME}_mixup_01"           $N_JOBS --run_name "${NAME}_mixup_01"           $ARGS --mixup.coef 0.1
./scripts/sbatch_job.sh "${NAME}_mixup_1"            $N_JOBS --run_name "${NAME}_mixup_1"            $ARGS --mixup.coef 1
./scripts/sbatch_job.sh "${NAME}_manifold_mixup_001" $N_JOBS --run_name "${NAME}_manifold_mixup_001" $ARGS --manifold_mixup.coef 0.01
./scripts/sbatch_job.sh "${NAME}_manifold_mixup_01"  $N_JOBS --run_name "${NAME}_manifold_mixup_01"  $ARGS --manifold_mixup.coef 0.1
./scripts/sbatch_job.sh "${NAME}_manifold_mixup_1"   $N_JOBS --run_name "${NAME}_manifold_mixup_1"   $ARGS --manifold_mixup.coef 1
./scripts/sbatch_job.sh "${NAME}_rotation_001"       $N_JOBS --run_name "${NAME}_rotation_001"       $ARGS --rotation.coef 0.01
./scripts/sbatch_job.sh "${NAME}_rotation_001_nc"    $N_JOBS --run_name "${NAME}_rotation_001_nc"    $ARGS --rotation.coef 0.01 --rotation.compare False
./scripts/sbatch_job.sh "${NAME}_rotation_01"        $N_JOBS --run_name "${NAME}_rotation_01"        $ARGS --rotation.coef 0.1
./scripts/sbatch_job.sh "${NAME}_rotation_01_nc"     $N_JOBS --run_name "${NAME}_rotation_01_nc"     $ARGS --rotation.coef 0.1  --rotation.compare False
./scripts/sbatch_job.sh "${NAME}_rotation_1"         $N_JOBS --run_name "${NAME}_rotation_1"         $ARGS --rotation.coef 1
./scripts/sbatch_job.sh "${NAME}_rotation_1_nc"      $N_JOBS --run_name "${NAME}_rotation_1_nc"      $ARGS --rotation.coef 1    --rotation.compare False
./scripts/sbatch_job.sh "${NAME}_brightness_001"     $N_JOBS --run_name "${NAME}_brightness_001"     $ARGS --brightness.coef 0.01
./scripts/sbatch_job.sh "${NAME}_brightness_001_nc"  $N_JOBS --run_name "${NAME}_brightness_001_nc"  $ARGS --brightness.coef 0.01 --brightness.compare False
./scripts/sbatch_job.sh "${NAME}_brightness_01"      $N_JOBS --run_name "${NAME}_brightness_01"      $ARGS --brightness.coef 0.1
./scripts/sbatch_job.sh "${NAME}_brightness_01_nc"   $N_JOBS --run_name "${NAME}_brightness_01_nc"   $ARGS --brightness.coef 0.1  --brightness.compare False
./scripts/sbatch_job.sh "${NAME}_brightness_1"       $N_JOBS --run_name "${NAME}_brightness_1"       $ARGS --brightness.coef 1
./scripts/sbatch_job.sh "${NAME}_brightness_1_nc"    $N_JOBS --run_name "${NAME}_brightness_1_nc"    $ARGS --brightness.coef 1    --brightness.compare False
./scripts/sbatch_job.sh "${NAME}_simclr_001"         $N_JOBS --run_name "${NAME}_simclr_001"         $ARGS --simclr.coef 0.01
./scripts/sbatch_job.sh "${NAME}_simclr_01"          $N_JOBS --run_name "${NAME}_simclr_01"          $ARGS --simclr.coef 0.1
./scripts/sbatch_job.sh	"${NAME}_simclr_1"           $N_JOBS --run_name "${NAME}_simclr_1"           $ARGS --simclr.coef 1
./scripts/sbatch_job.sh "${NAME}_vae_001"            $N_JOBS --run_name "${NAME}_vae_001"            $ARGS --vae.coef 0.01
./scripts/sbatch_job.sh "${NAME}_vae_01"             $N_JOBS --run_name "${NAME}_vae_01"             $ARGS --vae.coef 0.1
./scripts/sbatch_job.sh	"${NAME}_vae_1"              $N_JOBS --run_name "${NAME}_vae_1"              $ARGS --vae.coef 1
./scripts/sbatch_job.sh "${NAME}_ae_001"             $N_JOBS --run_name "${NAME}_ae_001"             $ARGS --ae.coef 0.01
./scripts/sbatch_job.sh "${NAME}_ae_01"              $N_JOBS --run_name "${NAME}_ae_01"              $ARGS --ae.coef 0.1
./scripts/sbatch_job.sh "${NAME}_ae_1"               $N_JOBS --run_name "${NAME}_ae_1"               $ARGS --ae.coef 1