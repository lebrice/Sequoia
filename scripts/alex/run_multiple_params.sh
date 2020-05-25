#!/bin/###bash
module load anaconda/3
conda-activate
export SCRATCH=/network/tmp1/ostapeno/SSCl
#cp -r /home/ostapeno/dev/SSCL/* /home/ostapeno/scratch/repos/SSCL/
#sup
bash sbatch_job_semi.sh 'sup' 1 --dataset cifar100 --multihead 1 --ewc_lamda 0 --n_classes_per_task 10 --supervised_epochs_per_task 200   --encoder_model resnet18 --batch_size 128 --run_name sup
#sup ewc
bash sbatch_job_semi.sh 'sup_ewc' 1 --dataset cifar100 --multihead 1 --ewc_lamda 100 --n_classes_per_task 10 --supervised_epochs_per_task 200 --encoder_model resnet18 --batch_size 128 --run_name sup_ewc
#ICT
bash sbatch_job_semi.sh 'ict' 1 --dataset cifar100 --mixup.coefficient 1 --multihead 1 --ewc_lamda 0 --n_classes_per_task 10 --supervised_epochs_per_task 200 --encoder_model resnet18 --batch_size 128 --run_name ict
#ICT EWC
bash sbatch_job_semi.sh 'ict_ewc' 1 --dataset cifar100 --mixup.coefficient 1 --multihead 1 --ewc_lamda 100 --n_classes_per_task 10 --supervised_epochs_per_task 200 --encoder_model resnet18 --batch_size 128 --run_name ict_ewc
#SimCLR
bash sbatch_job_semi.sh 'SimCLR' 1 --dataset cifar100 --simclr.coefficient 1 --multihead 1 --ewc_lamda 0 --n_classes_per_task 10 --supervised_epochs_per_task 200 --encoder_model resnet18 --batch_size 128 --run_name simclr
#SimCLR + EWC
bash sbatch_job_semi.sh 'simclr_ewc' 1 --dataset cifar100 --simclr.coefficient 1 --multihead 1 --ewc_lamda 100 --n_classes_per_task 10 --supervised_epochs_per_task 200 --encoder_model resnet18 --batch_size 128 --run_name simclr_ewc
#SimCLR
bash sbatch_job_semi.sh 'simCLR_detached' 1 --detach_classifier --dataset cifar100 --simclr.coefficient 1 --multihead 1 --ewc_lamda 0 --n_classes_per_task 10 --supervised_epochs_per_task 200 --encoder_model resnet18 --batch_size 128 --run_name simclr_etached
#SimCLR + EWC
bash sbatch_job_semi.sh 'simclr_ewc_detached' 1 --detach_classifier  --dataset cifar100 --simclr.coefficient 1 --multihead 1 --ewc_lamda 100 --n_classes_per_task 10 --supervised_epochs_per_task 200 --encoder_model resnet18 --batch_size 128 --run_name simclr_ewc_detached
#SimCLR + ICT
bash sbatch_job_semi.sh 'simclr_ict' 1 --dataset cifar100 --mixup.coefficient 1 --simclr.coefficient 1 --multihead 1 --ewc_lamda 0 --n_classes_per_task 10 --supervised_epochs_per_task 200 --encoder_model resnet18 --batch_size 128 --run_name simclr_ict
#SimCLR + ICT + EWC
bash sbatch_job_semi.sh 'simclr_ict_ewc' 1 --dataset cifar100 --mixup.coefficient 1 --simclr.coefficient 1 --multihead 1 --ewc_lamda 100 --n_classes_per_task 10 --supervised_epochs_per_task 200 --encoder_model resnet18 --batch_size 128 --run_name simclr_ict_ewc