#!/bin/###bash
module load anaconda/3
conda-activate
export SCRATCH=/network/tmp1/ostapeno/SSCl
#cp -r /home/ostapeno/dev/SSCL/* /home/ostapeno/scratch/repos/SSCL/

bash sbatch_job_semi.sh 'ict_hp' 1 --wandb_project SSCL_hp_2 --dataset cifar100 --mixup.coefficient 1 --ratio_labelled 0.05 --learning_rate 0.01 --lr_rampdown_epochs 150 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 1. --mixup_consistency 10 --consistency_rampup_ends 50 --multihead 1 --n_classes_per_task 20 --supervised_epochs_per_task 100  --batch_size 128 --run_name ict_hp --patience 5 --converge_after_epoch 100 --random_class_ordering 0

bash sbatch_job_semi.sh 'ict_hp' 1 --wandb_project SSCL_hp_2 --dataset cifar100 --mixup.coefficient 1 --ratio_labelled 0.05 --learning_rate 0.01 --lr_rampdown_epochs 150 --consistency_rampup_starts 1 --mixup_usup_alpha 0.1 --mixup_sup_alpha 0.1 --mixup_consistency 100 --consistency_rampup_ends 50 --multihead 1 --n_classes_per_task 20 --supervised_epochs_per_task 100  --batch_size 128 --run_name ict_hp --patience 5 --converge_after_epoch 100 --random_class_ordering 0

bash sbatch_job_semi.sh 'ict_hp' 1 --wandb_project SSCL_hp_2 --dataset cifar100 --mixup.coefficient 1 --ratio_labelled 0.05 --learning_rate 0.01 --lr_rampdown_epochs 150 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 1. --mixup_consistency 100 --consistency_rampup_ends 50 --multihead 1 --n_classes_per_task 20 --supervised_epochs_per_task 100  --batch_size 128 --run_name ict_hp --patience 5 --converge_after_epoch 100 --random_class_ordering 0

bash sbatch_job_semi.sh 'ict_hp' 1 --wandb_project SSCL_hp_2 --dataset cifar100 --mixup.coefficient 1 --ratio_labelled 0.05 --learning_rate 0.01 --lr_rampdown_epochs 150 --consistency_rampup_starts 1 --mixup_usup_alpha 0.1. --mixup_sup_alpha 0.1. --mixup_consistency 10 --consistency_rampup_ends 50 --multihead 1 --n_classes_per_task 20 --supervised_epochs_per_task 100  --batch_size 128 --run_name ict_hp --patience 5 --converge_after_epoch 100 --random_class_ordering 0


bash sbatch_job_semi.sh 'ict_hp' 1 --wandb_project SSCL_hp_2 --dataset cifar100 --mixup.coefficient 1 --ratio_labelled 0.05 --learning_rate 0.001 --lr_rampdown_epochs 150 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 1. --mixup_consistency 10 --consistency_rampup_ends 50 --multihead 1 --n_classes_per_task 20 --supervised_epochs_per_task 100  --batch_size 128 --run_name ict_hp --patience 5 --converge_after_epoch 100 --random_class_ordering 0

bash sbatch_job_semi.sh 'ict_hp' 1 --wandb_project SSCL_hp_2 --dataset cifar100 --mixup.coefficient 1 --ratio_labelled 0.05 --learning_rate 0.001 --lr_rampdown_epochs 150 --consistency_rampup_starts 1 --mixup_usup_alpha 0.1 --mixup_sup_alpha 0.1 --mixup_consistency 100 --consistency_rampup_ends 50 --multihead 1 --n_classes_per_task 20 --supervised_epochs_per_task 100  --batch_size 128 --run_name ict_hp --patience 5 --converge_after_epoch 100 --random_class_ordering 0

bash sbatch_job_semi.sh 'ict_hp' 1 --wandb_project SSCL_hp_2 --dataset cifar100 --mixup.coefficient 1 --ratio_labelled 0.05 --learning_rate 0.001 --lr_rampdown_epochs 150 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 1. --mixup_consistency 100 --consistency_rampup_ends 50 --multihead 1 --n_classes_per_task 20 --supervised_epochs_per_task 100  --batch_size 128 --run_name ict_hp --patience 5 --converge_after_epoch 100 --random_class_ordering 0

bash sbatch_job_semi.sh 'ict_hp' 1 --wandb_project SSCL_hp_2 --dataset cifar100 --mixup.coefficient 1 --ratio_labelled 0.05 --learning_rate 0.001 --lr_rampdown_epochs 150 --consistency_rampup_starts 1 --mixup_usup_alpha 0.1. --mixup_sup_alpha 0.1. --mixup_consistency 10 --consistency_rampup_ends 50 --multihead 1 --n_classes_per_task 20 --supervised_epochs_per_task 100  --batch_size 128 --run_name ict_hp --patience 5 --converge_after_epoch 100 --random_class_ordering 0


bash sbatch_job_semi.sh 'ict_hp' 1 --wandb_project SSCL_hp_2 --dataset cifar100 --mixup.coefficient 1 --ratio_labelled 0.05 --learning_rate 0.01 --lr_rampdown_epochs 150 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 1. --mixup_consistency 10 --consistency_rampup_ends 50 --multihead 1 --n_classes_per_task 20 --supervised_epochs_per_task 100  --batch_size 128 --run_name ict_hp --patience 5 --converge_after_epoch 100 --random_class_ordering 0

bash sbatch_job_semi.sh 'ict_hp' 1 --wandb_project SSCL_hp_2 --dataset cifar100 --mixup.coefficient 1 --ratio_labelled 0.05 --learning_rate 0.01 --lr_rampdown_epochs 150 --consistency_rampup_starts 1 --mixup_usup_alpha 0.1 --mixup_sup_alpha 0.1 --mixup_consistency 100 --consistency_rampup_ends 50 --multihead 1 --n_classes_per_task 20 --supervised_epochs_per_task 100  --batch_size 128 --run_name ict_hp --patience 5 --converge_after_epoch 100 --random_class_ordering 0

bash sbatch_job_semi.sh 'ict_hp' 1 --wandb_project SSCL_hp_2 --dataset cifar100 --mixup.coefficient 1 --ratio_labelled 0.05 --learning_rate 0.01 --lr_rampdown_epochs 150 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 1. --mixup_consistency 100 --consistency_rampup_ends 50 --multihead 1 --n_classes_per_task 20 --supervised_epochs_per_task 100  --batch_size 128 --run_name ict_hp --patience 5 --converge_after_epoch 100 --random_class_ordering 0

bash sbatch_job_semi.sh 'ict_hp' 1 --wandb_project SSCL_hp_2 --dataset cifar100 --mixup.coefficient 1 --ratio_labelled 0.05 --learning_rate 0.01 --lr_rampdown_epochs 150 --consistency_rampup_starts 1 --mixup_usup_alpha 0.1. --mixup_sup_alpha 0.1. --mixup_consistency 10 --consistency_rampup_ends 50 --multihead 1 --n_classes_per_task 20 --supervised_epochs_per_task 100  --batch_size 128 --run_name ict_hp --patience 5 --converge_after_epoch 100 --random_class_ordering 0

