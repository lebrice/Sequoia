import numpy as np
import pdb
import os
import time
import sys
from numpy.random import choice

runs = 1000
run_counter = 0
while run_counter < runs:
    lr = choice([0.1,0.001,0.0001])
    mixup_usup_alpha = choice([0.1,1,0.05,0.01])
    mixup_sup_alpha = choice([0.1, 1, 0.05, 0.01])
    mixup_consistency = choice([1,10,100,1000])
    consistency_rampup_ends = choice([10, 50, 100, 150])
    lr_rampdown_epochs = choice([150,200,300])

    cwd = os.getcwd()
    command = "python ../../task_incremental_sem_sup.py \
    --wandb_project SSCL_hp_4_resnet \
    --dataset cifar10 \
    --mixup.coefficient 1 \
    --ratio_labelled 0.05 \
    --learning_rate %(lr)s \
    --lr_rampdown_epochs %(lr_rampdown_epochs)s \
    --consistency_rampup_starts 1 \
    --mixup_usup_alpha %(mixup_usup_alpha)s \
    --mixup_sup_alpha %(mixup_sup_alpha)s \
    --mixup_consistency %(mixup_consistency)s \
    --consistency_rampup_ends %(consistency_rampup_ends)s \
    --multihead 1 \
    --n_classes_per_task 2 \
    --supervised_epochs_per_task 100 \
    --batch_size 128 \
    --run_name ict_hp \
    --random_class_ordering 0 \
    --encoder_model resnet18 \
              " % locals()
    print(command)

    os.system(command)
    #break
    time.sleep(2)
    run_counter += 1

