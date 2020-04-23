#!/bin/bash

NAME="cifar10_mh_d"
OUT="$SCRATCH/$NAME/%x-%j.out"

sbatch --output $OUT --job-name irm_001      	./scripts/sbatch_job.sh --run_name "${NAME}_irm_001"     	--dataset cifar10   --multihead --detach --irm.coef 0.01
sbatch --output $OUT --job-name irm_01       	./scripts/sbatch_job.sh --run_name "${NAME}_irm_01"      	--dataset cifar10   --multihead --detach --irm.coef 0.1
sbatch --output $OUT --job-name irm_1        	./scripts/sbatch_job.sh --run_name "${NAME}_irm_1"       	--dataset cifar10   --multihead --detach --irm.coef 1
sbatch --output $OUT --job-name mixup_001  	./scripts/sbatch_job.sh --run_name "${NAME}_mixup_001"   	--dataset cifar10   --multihead --detach --mixup.coef 0.01
sbatch --output $OUT --job-name mixup_01  	./scripts/sbatch_job.sh --run_name "${NAME}_mixup_01"   	--dataset cifar10   --multihead --detach --mixup.coef 0.1
sbatch --output $OUT --job-name mixup_1		./scripts/sbatch_job.sh --run_name "${NAME}_mixup_1"   		--dataset cifar10   --multihead --detach --mixup.coef 1
sbatch --output $OUT --job-name rotation_001	./scripts/sbatch_job.sh --run_name "${NAME}_rotation_001"	--dataset cifar10   --multihead --detach --rotation.coef 0.01
sbatch --output $OUT --job-name rotation_01	./scripts/sbatch_job.sh --run_name "${NAME}_rotation_01"	--dataset cifar10   --multihead --detach --rotation.coef 0.1
sbatch --output $OUT --job-name rotation_1	./scripts/sbatch_job.sh --run_name "${NAME}_rotation_1"		--dataset cifar10   --multihead --detach --rotation.coef 1
sbatch --output $OUT --job-name rotation_002_nc	./scripts/sbatch_job.sh --run_name "${NAME}_rotation_001_nc"	--dataset cifar10   --multihead --detach --rotation.coef 0.01 --rotation.compare False
sbatch --output $OUT --job-name rotation_01_nc	./scripts/sbatch_job.sh --run_name "${NAME}_rotation_01_nc"	--dataset cifar10   --multihead --detach --rotation.coef 0.1  --rotation.compare False
sbatch --output $OUT --job-name rotation_1_nc	./scripts/sbatch_job.sh --run_name "${NAME}_rotation_1_nc"	--dataset cifar10   --multihead --detach --rotation.coef 1    --rotation.compare False
sbatch --output $OUT --job-name simclr_001	./scripts/sbatch_job.sh --run_name "${NAME}_simclr_001"		--dataset cifar10   --multihead --detach --simclr.coef 0.01
sbatch --output $OUT --job-name simclr_01	./scripts/sbatch_job.sh --run_name "${NAME}_simclr_01"		--dataset cifar10   --multihead --detach --simclr.coef 0.1
sbatch --output $OUT --job-name simclr_1	./scripts/sbatch_job.sh	--run_name "${NAME}_simclr_1"		--dataset cifar10   --multihead --detach --simclr.coef 1

