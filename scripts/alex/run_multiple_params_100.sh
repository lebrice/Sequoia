#!/bin/bash
module load anaconda/3
conda-activate
export SCRATCH=/network/tmp1/ostapeno/SSCl_$USER

#cp -r /home/ostapeno/dev/SSCL/* /home/ostapeno/scratch/repos/SSCL/

#0.2
#sup
##bash sbatch_job_semi.sh 'sup_loss' 3 --wandb_project SSCL_6_test_01 --dataset cifar100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group sup_loss --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'sup_acc' 3 --wandb_project SSCL_6_test_01 --dataset cifar100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group sup_acc --use_accuracy_as_metric 1 --random_class_ordering 0
#bash sbatch_job_semi.sh 'sup_acc' 3 --wandb_project SSCL_6_test_02_full --simclr_augment 0 --dataset cifar100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group sup_acc --use_accuracy_as_metric 1 --random_class_ordering 0 --semi_setup_full 1


#sup ewc
#bash sbatch_job_semi.sh 'sup_ewc_loss' 3 --wandb_project SSCL_6_test_01 --dataset cifar100 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group sup_ewc_loss --use_accuracy_as_metric 0 --random_class_ordering 0
##bash sbatch_job_semi.sh 'sup_ewc_acc' 3 --wandb_project SSCL_6_test_01 --dataset cifar100 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group sup_ewc_acc --use_accuracy_as_metric 1 --random_class_ordering 0
#bash sbatch_job_semi.sh 'sup_ewc_acc' 3 --wandb_project SSCL_6_test_02_full --simclr_augment 0 --dataset cifar100 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group sup_ewc_acc --use_accuracy_as_metric 1 --random_class_ordering 0 --semi_setup_full 1

#ICT
#bash sbatch_job_semi.sh 'ict_loss' 3 --wandb_project SSCL_6_test_01 --dataset cifar100 --mixup.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.2 --n_classes_per_task 20 --supervised_epochs_per_task 200  --batch_size 128 --run_group ict_loss_metric_3 --use_accuracy_as_metric 0 --random_class_ordering 0
##bash sbatch_job_semi.sh 'ict_acc' 3 --wandb_project SSCL_6_test_01 --dataset cifar100 --mixup.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.2 --n_classes_per_task 20 --supervised_epochs_per_task 200  --batch_size 128 --run_group ict_acc_metric_3 --use_accuracy_as_metric 1 --random_class_ordering 0
bash sbatch_job_semi.sh 'ict_acc' 1 --wandb_project SSCL_6_test_02_full --simclr_augment 0 --dataset cifar100 --mixup.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.2 --n_classes_per_task 20 --supervised_epochs_per_task 200  --batch_size 128 --run_group ict_acc_metric_3 --use_accuracy_as_metric 1 --random_class_ordering 0 --semi_setup_full 1


##ICT EWC
#bash sbatch_job_semi.sh 'ict_ewc_loss' 3 --wandb_project SSCL_6_test_01  --dataset cifar100 --mixup.coefficient 1 --ewc.coefficient 100  --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.2 --n_classes_per_task 20  --supervised_epochs_per_task 200 --batch_size 128 --run_group ict_ewc_loss_metric --use_accuracy_as_metric 0  --random_class_ordering 0
##bash sbatch_job_semi.sh 'ict_ewc_acc' 3 --wandb_project SSCL_6_test_01  --dataset cifar100 --mixup.coefficient 1 --ewc.coefficient 100  --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.2 --n_classes_per_task 20  --supervised_epochs_per_task 200 --batch_size 128 --run_group ict_ewc_acc_metric --use_accuracy_as_metric 1  --random_class_ordering 0
bash sbatch_job_semi.sh 'ict_ewc_acc' 2 --wandb_project SSCL_6_test_02_full --simclr_augment 0 --dataset cifar100 --mixup.coefficient 1 --ewc.coefficient 100  --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.2 --n_classes_per_task 20  --supervised_epochs_per_task 200 --batch_size 128 --run_group ict_ewc_acc_metric --use_accuracy_as_metric 1  --random_class_ordering 0 --semi_setup_full 1

#SimCLR
#bash sbatch_job_semi.sh 'SimCLR_conv_loss' 3 --wandb_project SSCL_6_test_01  --dataset cifar100 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_loss_metric --use_accuracy_as_metric 0 --random_class_ordering 0
##bash sbatch_job_semi.sh 'SimCLR_conv_acc' 3 --wandb_project SSCL_6_test_01  --dataset cifar100 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0
bash sbatch_job_semi.sh 'SimCLR_conv_acc' 3 --wandb_project SSCL_6_test_02_full --dataset cifar100 --simclr.coefficient 1 --double_augmentation 0 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0 --semi_setup_full 1


#SimCLR + EWC
#bash sbatch_job_semi.sh 'simclr_ewc_conv_loss' 3 --wandb_project SSCL_6_test_01  --dataset cifar100 --simclr.coefficient 1 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ewc --use_accuracy_as_metric 0 --random_class_ordering 0
##bash sbatch_job_semi.sh 'simclr_ewc_conv_acc' 3 --wandb_project SSCL_6_test_01  --dataset cifar100 --simclr.coefficient 1 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ewc_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0
#bash sbatch_job_semi.sh 'simclr_ewc_conv_acc' 2 --wandb_project SSCL_6_test_02_full  --dataset cifar100 --simclr.coefficient 1 --double_augmentation 0 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ewc_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0 --semi_setup_full 1


#SimCLR
#bash sbatch_job_semi.sh 'simCLR_detached_conv_loss' 3 --wandb_project SSCL_6_test_01  --detach_classifier --dataset cifar100 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_etached --use_accuracy_as_metric 0 --random_class_ordering 0
##bash sbatch_job_semi.sh 'simCLR_detached_conv_acc' 3 --wandb_project SSCL_6_test_01  --detach_classifier --dataset cifar100 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_etached_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0
#bash sbatch_job_semi.sh 'simCLR_detached_conv_acc' 2 --wandb_project SSCL_6_test_02_full  --detach_classifier --dataset cifar100 --simclr.coefficient 1 --double_augmentation 0 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_etached_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0 --semi_setup_full 1


#SimCLR + EWC
#bash sbatch_job_semi.sh 'simclr_ewc_detached_conv_loss' 3 --wandb_project SSCL_6_test_01  --detach_classifier  --dataset cifar100 --ewc.coefficient 100 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ewc_detached --use_accuracy_as_metric 0 --random_class_ordering 0
##bash sbatch_job_semi.sh 'simclr_ewc_detached_conv_acc' 3 --wandb_project SSCL_6_test_01  --detach_classifier  --dataset cifar100 --ewc.coefficient 100 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ewc_detached_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0
#bash sbatch_job_semi.sh 'simclr_ewc_detached_conv_acc' 3 --wandb_project SSCL_6_test_02_full  --detach_classifier  --dataset cifar100 --ewc.coefficient 100 --simclr.coefficient 1 --double_augmentation 0 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ewc_detached_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0 --semi_setup_full 1


#SimCLR + ICT
#bash sbatch_job_semi.sh 'simclr_ict_conv_loss' 3 --wandb_project SSCL_6_test_01  --dataset cifar100 --mixup.coefficient 1 --simclr.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ict --use_accuracy_as_metric 0 --random_class_ordering 0
##bash sbatch_job_semi.sh 'simclr_ict_conv_acc' 3 --wandb_project SSCL_6_test_01  --dataset cifar100 --mixup.coefficient 1 --simclr.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ict_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0
#bash sbatch_job_semi.sh 'simclr_ict_conv_acc' 3 --wandb_project SSCL_6_test_02_full  --dataset cifar100 --mixup.coefficient 1 --simclr.coefficient 1 --double_augmentation 0 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ict_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0 --semi_setup_full 1


#SimCLR + ICT + EWC
#bash sbatch_job_semi.sh 'simclr_ict_ewc_conv_loss' 3 --wandb_project SSCL_6_test_01  --dataset cifar100 --mixup.coefficient 1 --ewc.coefficient 100 --simclr.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ict_ewc --use_accuracy_as_metric 0 --random_class_ordering 0
##bash sbatch_job_semi.sh 'simclr_ict_ewc_conv_acc' 3 --wandb_project SSCL_6_test_01  --dataset cifar100 --mixup.coefficient 1 --ewc.coefficient 100 --simclr.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ict_ewc_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0
#bash sbatch_job_semi.sh 'simclr_ict_ewc_conv_acc' 3 --wandb_project SSCL_6_test_02_full  --dataset cifar100 --mixup.coefficient 1 --ewc.coefficient 100 --simclr.coefficient 1 --double_augmentation 0 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ict_ewc_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0 --semi_setup_full 1





#0.2
#sup
#bash sbatch_job_semi.sh 'sup_loss' 3 --wandb_project SSCL_6_test_01 --dataset cifar100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group sup_loss --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'sup_acc' 3 --wandb_project SSCL_6_test_02 --dataset cifar100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group sup_acc --use_accuracy_as_metric 1 --random_class_ordering 0

#sup ewc
#bash sbatch_job_semi.sh 'sup_ewc_loss' 3 --wandb_project SSCL_6_test_01 --dataset cifar100 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group sup_ewc_loss --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'sup_ewc_acc' 3 --wandb_project SSCL_6_test_02 --dataset cifar100 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group sup_ewc_acc --use_accuracy_as_metric 1 --random_class_ordering 0

#ICT
#bash sbatch_job_semi.sh 'ict_loss' 3 --wandb_project SSCL_6_test_01 --dataset cifar100 --mixup.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.2 --n_classes_per_task 20 --supervised_epochs_per_task 200  --batch_size 128 --run_group ict_loss_metric_3 --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'ict_acc' 3 --wandb_project SSCL_6_test_02 --dataset cifar100 --mixup.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.2 --n_classes_per_task 20 --supervised_epochs_per_task 200  --batch_size 128 --run_group ict_acc_metric_3 --use_accuracy_as_metric 1 --random_class_ordering 0

##ICT EWC
#bash sbatch_job_semi.sh 'ict_ewc_loss' 3 --wandb_project SSCL_6_test_01  --dataset cifar100 --mixup.coefficient 1 --ewc.coefficient 100  --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.2 --n_classes_per_task 20  --supervised_epochs_per_task 200 --batch_size 128 --run_group ict_ewc_loss_metric --use_accuracy_as_metric 0  --random_class_ordering 0
#bash sbatch_job_semi.sh 'ict_ewc_acc' 3 --wandb_project SSCL_6_test_02  --dataset cifar100 --mixup.coefficient 1 --ewc.coefficient 100  --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.2 --n_classes_per_task 20  --supervised_epochs_per_task 200 --batch_size 128 --run_group ict_ewc_acc_metric --use_accuracy_as_metric 1  --random_class_ordering 0

#SimCLR
#bash sbatch_job_semi.sh 'SimCLR_conv_loss' 3 --wandb_project SSCL_6_test_01  --dataset cifar100 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_loss_metric --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'SimCLR_conv_acc' 3 --wandb_project SSCL_6_test_02  --dataset cifar100 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0

#SimCLR + EWC
#bash sbatch_job_semi.sh 'simclr_ewc_conv_loss' 3 --wandb_project SSCL_6_test_01  --dataset cifar100 --simclr.coefficient 1 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ewc --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'simclr_ewc_conv_acc' 3 --wandb_project SSCL_6_test_02  --dataset cifar100 --simclr.coefficient 1 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ewc_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0

#SimCLR
#bash sbatch_job_semi.sh 'simCLR_detached_conv_loss' 3 --wandb_project SSCL_6_test_01  --detach_classifier --dataset cifar100 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_etached --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'simCLR_detached_conv_acc' 3 --wandb_project SSCL_6_test_02  --detach_classifier --dataset cifar100 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_etached_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0

#SimCLR + EWC
#bash sbatch_job_semi.sh 'simclr_ewc_detached_conv_loss' 3 --wandb_project SSCL_6_test_01  --detach_classifier  --dataset cifar100 --ewc.coefficient 100 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ewc_detached --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'simclr_ewc_detached_conv_acc' 3 --wandb_project SSCL_6_test_02  --detach_classifier  --dataset cifar100 --ewc.coefficient 100 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ewc_detached_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0

#SimCLR + ICT
#bash sbatch_job_semi.sh 'simclr_ict_conv_loss' 3 --wandb_project SSCL_6_test_01  --dataset cifar100 --mixup.coefficient 1 --simclr.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ict --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'simclr_ict_conv_acc' 3 --wandb_project SSCL_6_test_02  --dataset cifar100 --mixup.coefficient 1 --simclr.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ict_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0

#SimCLR + ICT + EWC
#bash sbatch_job_semi.sh 'simclr_ict_ewc_conv_loss' 3 --wandb_project SSCL_6_test_01  --dataset cifar100 --mixup.coefficient 1 --ewc.coefficient 100 --simclr.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ict_ewc --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'simclr_ict_ewc_conv_acc' 3 --wandb_project SSCL_6_test_02  --dataset cifar100 --mixup.coefficient 1 --ewc.coefficient 100 --simclr.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 20 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group simclr_ict_ewc_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0
