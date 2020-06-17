#!/bin/bash        
module load anaconda/3
conda-activate
export SCRATCH=/network/tmp1/ostapeno/SSCl_$USER

#cp -r /home/ostapeno/dev/SSCL/* /home/ostapeno/scratch/repos/SSCL/
#cd /home/ostapeno/scratch/repos/SSCL/scripts/alex

#0.05
#sup
##bash sbatch_job_semi.sh 'sup_loss' 3 --project_name SSCL_6_test_005 --dataset cifar10 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_sup_loss --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'sup_acc' 3 --project_name SSCL_6_test_005 --dataset cifar10 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_sup_acc --use_accuracy_as_metric 1 --random_class_ordering 0
bash sbatch_job_semi.sh 'sup_acc' 3 --project_name SSCL_6_test_005_cifar10_full_2 --semi_setup_full 1 --simclr_augment 0 --dataset cifar10 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_sup_acc --run_name sup_acc --use_accuracy_as_metric 1 --random_class_ordering 0


#sup ewc
#bash sbatch_job_semi.sh 'sup_ewc_loss' 3 --project_name SSCL_6_test_005 --dataset cifar10 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_sup_ewc_loss --use_accuracy_as_metric 0 --random_class_ordering 0
##bash sbatch_job_semi.sh 'sup_ewc_acc' 3 --project_name SSCL_6_test_005 --dataset cifar10 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_sup_ewc_acc --use_accuracy_as_metric 1 --random_class_ordering 0
bash sbatch_job_semi.sh 'sup_ewc_acc' 3 --project_name SSCL_6_test_005_cifar10_full_2 --semi_setup_full 1 --simclr_augment 0 --dataset cifar10 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_sup_ewc_acc --run_name sup_ewc_acc --use_accuracy_as_metric 1 --random_class_ordering 0

#ICT
#bash sbatch_job_semi.sh 'ict_loss' 3 --project_name SSCL_6_test_005 --dataset cifar10 --mixup.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.05 --n_classes_per_task 2 --supervised_epochs_per_task 200  --batch_size 128 --run_group cifar10_ict_loss_metric_3 --use_accuracy_as_metric 0 --random_class_ordering 0
##bash sbatch_job_semi.sh 'ict_acc' 3 --project_name SSCL_6_test_005 --dataset cifar10 --mixup.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.05 --n_classes_per_task 2 --supervised_epochs_per_task 200  --batch_size 128 --run_group cifar10_ict_acc_metric_3 --use_accuracy_as_metric 1 --random_class_ordering 0
bash sbatch_job_semi.sh 'ict_acc' 3 --project_name SSCL_6_test_005_cifar10_full_2 --semi_setup_full 1 --simclr_augment 0 --dataset cifar10 --mixup.coefficient 1 --learning_rate 0.1 --lr_rampdown_epochs 300 --consistency_rampup_starts 1 --mixup_usup_alpha 1.0 --mixup_sup_alpha 0.1 --mixup_consistency 1 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.05 --n_classes_per_task 2 --supervised_epochs_per_task 200  --batch_size 128 --run_group cifar10_ict_acc_metric_3 --run_name ict_acc_metric_3 --use_accuracy_as_metric 1 --random_class_ordering 0


##ICT EWC
#bash sbatch_job_semi.sh 'ict_ewc_loss' 3 --project_name SSCL_6_test_005  --dataset cifar10 --mixup.coefficient 1 --ewc.coefficient 100  --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.05 --n_classes_per_task 2  --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_ict_ewc_loss_metric --use_accuracy_as_metric 0  --random_class_ordering 0
##bash sbatch_job_semi.sh 'ict_ewc_acc' 3 --project_name SSCL_6_test_005  --dataset cifar10 --mixup.coefficient 1 --ewc.coefficient 100  --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.05 --n_classes_per_task 2  --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_ict_ewc_acc_metric --use_accuracy_as_metric 1  --random_class_ordering 0
bash sbatch_job_semi.sh 'ict_ewc_acc' 3 --project_name SSCL_6_test_005_cifar10_full_2 --semi_setup_full 1 --simclr_augment 0 --dataset cifar10 --mixup.coefficient 1 --ewc.coefficient 100  --learning_rate 0.1 --lr_rampdown_epochs 300 --consistency_rampup_starts 1 --mixup_usup_alpha 1.0 --mixup_sup_alpha 0.1 --mixup_consistency 1 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.05 --n_classes_per_task 2  --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_ict_ewc_acc_metric --run_name ict_ewc_acc_metric --use_accuracy_as_metric 1  --random_class_ordering 0

#SimCLR
#bash sbatch_job_semi.sh 'SimCLR_conv_loss' 3 --project_name SSCL_6_test_005  --dataset cifar10 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_loss_metric --use_accuracy_as_metric 0 --random_class_ordering 0
##bash sbatch_job_semi.sh 'SimCLR_conv_acc' 3 --project_name SSCL_6_test_005  --dataset cifar10 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0
bash sbatch_job_semi.sh 'SimCLR_conv_acc' 3 --project_name SSCL_6_test_005_cifar10_full_2 --semi_setup_full 1 --dataset cifar10 --simclr.coefficient 1 --double_augmentation 0 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_accmetr --run_name simclr_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0


#SimCLR + EWC
#bash sbatch_job_semi.sh 'simclr_ewc_conv_loss' 3 --project_name SSCL_6_test_005  --dataset cifar10 --simclr.coefficient 1 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ewc --use_accuracy_as_metric 0 --random_class_ordering 0
##bash sbatch_job_semi.sh 'simclr_ewc_conv_acc' 3 --project_name SSCL_6_test_005  --dataset cifar10 --simclr.coefficient 1 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ewc_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0
bash sbatch_job_semi.sh 'simclr_ewc_conv_acc' 3 --project_name SSCL_6_test_005_cifar10_full_2 --semi_setup_full 1  --dataset cifar10 --simclr.coefficient 1 --double_augmentation 0 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ewc_accmetr --run_name simclr_ewc_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0


#SimCLR
#bash sbatch_job_semi.sh 'simCLR_detached_conv_loss' 3 --project_name SSCL_6_test_005  --detach_classifier --dataset cifar10 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_etached --use_accuracy_as_metric 0 --random_class_ordering 0
##bash sbatch_job_semi.sh 'simCLR_detached_conv_acc' 3 --project_name SSCL_6_test_005  --detach_classifier --dataset cifar10 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_etached_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0
bash sbatch_job_semi.sh 'simCLR_detached_conv_acc' 3 --project_name SSCL_6_test_005_cifar10_full_2 --semi_setup_full 1  --detach_classifier --dataset cifar10 --simclr.coefficient 1 --double_augmentation 0 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_etached_accmetr --run_name simclr_etached_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0


#SimCLR + EWC
#bash sbatch_job_semi.sh 'simclr_ewc_detached_conv_loss' 3 --project_name SSCL_6_test_005  --detach_classifier  --dataset cifar10 --ewc.coefficient 100 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ewc_detached --use_accuracy_as_metric 0 --random_class_ordering 0
##bash sbatch_job_semi.sh 'simclr_ewc_detached_conv_acc' 3 --project_name SSCL_6_test_005  --detach_classifier  --dataset cifar10 --ewc.coefficient 100 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ewc_detached_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0
bash sbatch_job_semi.sh 'simclr_ewc_detached_conv_acc' 3 --project_name SSCL_6_test_005_cifar10_full_2 --semi_setup_full 1  --detach_classifier  --dataset cifar10 --ewc.coefficient 100 --simclr.coefficient 1 --double_augmentation 0 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ewc_detached_accmetr --run_name simclr_ewc_detached_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0


#SimCLR + ICT
#bash sbatch_job_semi.sh 'simclr_ict_conv_loss' 3 --project_name SSCL_6_test_005  --dataset cifar10 --mixup.coefficient 1 --simclr.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ict --use_accuracy_as_metric 0 --random_class_ordering 0
##bash sbatch_job_semi.sh 'simclr_ict_conv_acc' 3 --project_name SSCL_6_test_005  --dataset cifar10 --mixup.coefficient 1 --simclr.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ict_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0
bash sbatch_job_semi.sh 'simclr_ict_conv_acc' 3 --project_name SSCL_6_test_005_cifar10_full_2 --semi_setup_full 1  --dataset cifar10 --mixup.coefficient 1 --simclr.coefficient 1 --double_augmentation 0 --learning_rate 0.1 --lr_rampdown_epochs 300 --consistency_rampup_starts 1 --mixup_usup_alpha 1.0 --mixup_sup_alpha 0.1 --mixup_consistency 1 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ict_accmetr --run_name simclr_ict_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0


#SimCLR + ICT + EWC
#bash sbatch_job_semi.sh 'simclr_ict_ewc_conv_loss' 3 --project_name SSCL_6_test_005  --dataset cifar10 --mixup.coefficient 1 --ewc.coefficient 100 --simclr.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ict_ewc --use_accuracy_as_metric 0 --random_class_ordering 0
##bash sbatch_job_semi.sh 'simclr_ict_ewc_conv_acc' 3 --project_name SSCL_6_test_005  --dataset cifar10 --mixup.coefficient 1 --ewc.coefficient 100 --simclr.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ict_ewc_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0
bash sbatch_job_semi.sh 'simclr_ict_ewc_conv_acc' 3 --project_name SSCL_6_test_005_cifar10_full_2 --semi_setup_full 1   --dataset cifar10 --mixup.coefficient 1 --ewc.coefficient 100 --simclr.coefficient 1 --double_augmentation 0 --learning_rate 0.1 --lr_rampdown_epochs 300 --consistency_rampup_starts 1 --mixup_usup_alpha 1.0 --mixup_sup_alpha 0.1 --mixup_consistency 1 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ict_ewc_accmetr --run_name simclr_ict_ewc_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0





#0.2
#sup
#bash sbatch_job_semi.sh 'sup_loss' 3 --project_name SSCL_6_test_005 --dataset cifar10 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_sup_loss --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'sup_acc' 3 --project_name SSCL_6_test_02 --dataset cifar10 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_sup_acc --use_accuracy_as_metric 1 --random_class_ordering 0

#sup ewc
#bash sbatch_job_semi.sh 'sup_ewc_loss' 3 --project_name SSCL_6_test_005 --dataset cifar10 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_sup_ewc_loss --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'sup_ewc_acc' 3 --project_name SSCL_6_test_02 --dataset cifar10 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_sup_ewc_acc --use_accuracy_as_metric 1 --random_class_ordering 0

#ICT
#bash sbatch_job_semi.sh 'ict_loss' 3 --project_name SSCL_6_test_005 --dataset cifar10 --mixup.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.05 --n_classes_per_task 2 --supervised_epochs_per_task 200  --batch_size 128 --run_group cifar10_ict_loss_metric_3 --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'ict_acc' 3 --project_name SSCL_6_test_02 --dataset cifar10 --mixup.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.2 --n_classes_per_task 2 --supervised_epochs_per_task 200  --batch_size 128 --run_group cifar10_ict_acc_metric_3 --use_accuracy_as_metric 1 --random_class_ordering 0

##ICT EWC
#bash sbatch_job_semi.sh 'ict_ewc_loss' 3 --project_name SSCL_6_test_005  --dataset cifar10 --mixup.coefficient 1 --ewc.coefficient 100  --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.05 --n_classes_per_task 2  --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_ict_ewc_loss_metric --use_accuracy_as_metric 0  --random_class_ordering 0
#bash sbatch_job_semi.sh 'ict_ewc_acc' 3 --project_name SSCL_6_test_02  --dataset cifar10 --mixup.coefficient 1 --ewc.coefficient 100  --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --ratio_labelled 0.2 --n_classes_per_task 2  --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_ict_ewc_acc_metric --use_accuracy_as_metric 1  --random_class_ordering 0

#SimCLR
#bash sbatch_job_semi.sh 'SimCLR_conv_loss' 3 --project_name SSCL_6_test_005  --dataset cifar10 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_loss_metric --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'SimCLR_conv_acc' 3 --project_name SSCL_6_test_02  --dataset cifar10 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0

#SimCLR + EWC
#bash sbatch_job_semi.sh 'simclr_ewc_conv_loss' 3 --project_name SSCL_6_test_005  --dataset cifar10 --simclr.coefficient 1 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ewc --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'simclr_ewc_conv_acc' 3 --project_name SSCL_6_test_02  --dataset cifar10 --simclr.coefficient 1 --ewc.coefficient 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ewc_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0

#SimCLR
#bash sbatch_job_semi.sh 'simCLR_detached_conv_loss' 3 --project_name SSCL_6_test_005  --detach_classifier --dataset cifar10 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_etached --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'simCLR_detached_conv_acc' 3 --project_name SSCL_6_test_02  --detach_classifier --dataset cifar10 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_etached_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0

#SimCLR + EWC
#bash sbatch_job_semi.sh 'simclr_ewc_detached_conv_loss' 3 --project_name SSCL_6_test_005  --detach_classifier  --dataset cifar10 --ewc.coefficient 100 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ewc_detached --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'simclr_ewc_detached_conv_acc' 3 --project_name SSCL_6_test_02  --detach_classifier  --dataset cifar10 --ewc.coefficient 100 --simclr.coefficient 1 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ewc_detached_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0

#SimCLR + ICT
#bash sbatch_job_semi.sh 'simclr_ict_conv_loss' 3 --project_name SSCL_6_test_005  --dataset cifar10 --mixup.coefficient 1 --simclr.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ict --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'simclr_ict_conv_acc' 3 --project_name SSCL_6_test_02  --dataset cifar10 --mixup.coefficient 1 --simclr.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ict_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0

#SimCLR + ICT + EWC
#bash sbatch_job_semi.sh 'simclr_ict_ewc_conv_loss' 3 --project_name SSCL_6_test_005  --dataset cifar10 --mixup.coefficient 1 --ewc.coefficient 100 --simclr.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.05 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ict_ewc --use_accuracy_as_metric 0 --random_class_ordering 0
#bash sbatch_job_semi.sh 'simclr_ict_ewc_conv_acc' 3 --project_name SSCL_6_test_02  --dataset cifar10 --mixup.coefficient 1 --ewc.coefficient 100 --simclr.coefficient 1 --learning_rate 0.001 --lr_rampdown_epochs 350 --consistency_rampup_starts 1 --mixup_usup_alpha 1. --mixup_sup_alpha 0.01 --mixup_consistency 10 --consistency_rampup_ends 100 --multihead 1 --n_classes_per_task 2 --ratio_labelled 0.2 --supervised_epochs_per_task 200 --batch_size 128 --run_group cifar10_simclr_ict_ewc_accmetr --use_accuracy_as_metric 1 --random_class_ordering 0
