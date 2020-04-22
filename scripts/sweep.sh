#!/bin/bash

sbatch  --job-name irm_001      	./scripts/sbatch_job.sh 	--run_name cifar10_mh_irm_001     		--dataset cifar10   --multihead --irm.coefficient 0.01
sbatch  --job-name irm_01       	./scripts/sbatch_job.sh 	--run_name cifar10_mh_irm_01      		--dataset cifar10   --multihead --irm.coefficient 0.1
sbatch  --job-name irm_1        	./scripts/sbatch_job.sh 	--run_name cifar10_mh_irm_1       		--dataset cifar10   --multihead --irm.coefficient 1
sbatch  --job-name mixup_001  		./scripts/sbatch_job.sh 	--run_name cifar10_mh_mixup_001   		--dataset cifar10   --multihead --mixup.coefficient 0.01
sbatch  --job-name mixup_01  		./scripts/sbatch_job.sh 	--run_name cifar10_mh_mixup_01   		--dataset cifar10   --multihead --mixup.coefficient 0.1
sbatch  --job-name mixup_1  		./scripts/sbatch_job.sh 	--run_name cifar10_mh_mixup_1   		--dataset cifar10   --multihead --mixup.coefficient 1
sbatch  --job-name rotation_001		./scripts/sbatch_job.sh 	--run_name cifar10_mh_rotation_001		--dataset cifar10   --multihead --rotation.coefficient 0.01
sbatch  --job-name rotation_01		./scripts/sbatch_job.sh 	--run_name cifar10_mh_rotation_01		--dataset cifar10   --multihead --rotation.coefficient 0.1
sbatch  --job-name rotation_1		./scripts/sbatch_job.sh 	--run_name cifar10_mh_rotation_1		--dataset cifar10   --multihead --rotation.coefficient 1
sbatch  --job-name rotation_001_nc	./scripts/sbatch_job.sh 	--run_name cifar10_mh_rotation_001_nc	--dataset cifar10   --multihead --rotation.coefficient 0.01	--rotation.compare_with_original False
sbatch  --job-name rotation_01_nc	./scripts/sbatch_job.sh 	--run_name cifar10_mh_rotation_01_nc	--dataset cifar10   --multihead --rotation.coefficient 0.1	--rotation.compare_with_original False
sbatch  --job-name rotation_1_nc	./scripts/sbatch_job.sh 	--run_name cifar10_mh_rotation_1_nc		--dataset cifar10   --multihead --rotation.coefficient 1	--rotation.compare_with_original False
sbatch  --job-name simclr_001		./scripts/sbatch_job.sh 	--run_name cifar10_mh_simclr_001		--dataset cifar10   --multihead --simclr.coefficient 0.01
sbatch  --job-name simclr_01		./scripts/sbatch_job.sh 	--run_name cifar10_mh_simclr_01			--dataset cifar10   --multihead --simclr.coefficient 0.1
sbatch  --job-name simclr_1			./scripts/sbatch_job.sh 	--run_name cifar10_mh_simclr_1			--dataset cifar10   --multihead --simclr.coefficient 1

