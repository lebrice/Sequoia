#!/bin/bash
export SCRATCH=$HOME/scratch

bash sbatch_job_semi.sh 'simclr_sup' 1 --project_name SSCL --label_incremental 0 --simclr_augment 0 --dataset cifar100 --multihead 1 --n_classes_per_task 10 --ratio_labelled 0.05 --supervised_epochs_per_task 100 --batch_size 256 --run_group simclr_sup  --run_name simclr_sup --use_accuracy_as_metric 1 --random_class_ordering 0 #--encoder_model resnet18

bash sbatch_job_semi.sh 'simclr_sup_pretrain' 1 --project_name SSCL --label_incremental 0 --simclr_augment 0 --dataset cifar100 --multihead 1 --n_classes_per_task 10 --unsupervised_epochs_pretraining 200 --pretraining_dataset mini_imagenet --use_full_unlabeled 1 --ratio_labelled 0.05 --supervised_epochs_per_task 100 --batch_size 256 --run_group simclr_sup_petr  --run_name simclr_sup_petr --use_accuracy_as_metric 1 --random_class_ordering 0 #--encoder_model resnet18

bash sbatch_job_semi.sh 'simclr_sup_da' 1 --project_name SSCL --label_incremental 0 --simclr_augment 0 --dataset cifar100 --multihead 1 --n_classes_per_task 10 --dataset_unlabeled mini_imagenet  --use_full_unlabeled 1 --ratio_labelled 0.05 --supervised_epochs_per_task 100 --batch_size 256 --run_group simclr_sup_da  --run_name simclr_sup_da --use_accuracy_as_metric 1 --random_class_ordering 0 #--encoder_model resnet18
