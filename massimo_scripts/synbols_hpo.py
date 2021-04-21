import numpy as np
import os

# CHANGE ME #
LAUNCH_COMMAND = 'bash -i launch_job_toolkit.sh '
SEQUOIA_SWEEP_PATH = '/mnt/home/.conda/envs/sequoia/bin/sequoia_sweep' 
WANDB_API_KEY = "799759daba56493c8f8c0bd47660901e9efef01e"
#############

project = 'synbols_sweep2'
# project = 'synbols_debug'
max_runs = 100000
n_jobs = 200

# SETTINGS = ['iid', 'task_incremental', 'multi_task', 'class_incremental']
SETTINGS = ['iid', 'multi_task', 'class_incremental']
CL_SETTINGS = ['task_incremental', 'class_incremental']
NB_TASKS = [12]
METHODS = ['experience_replay']# 'experience_replay']
# METHODS = ['experience_replay']
CL_METHODS = ['ewc', 'experience_replay']
DATASETS = ['synbols']
ENCODERS = ['resnet18']

dataset = DATASETS[0]
encoder = ENCODERS[0]
nb_task = NB_TASKS[0]

for setting in SETTINGS:
    for method in METHODS:
        
        if method in CL_METHODS and setting not in CL_SETTINGS:
            continue    
        
        orion_db = f'orion/{setting}_{dataset}_{method}.pkl'
        
        
        command = f'--setting {setting} ' \
                    f' --max_runs {max_runs} ' \
                    f' --dataset {dataset} ' \
                    f' --nb_tasks {nb_task} ' \
                    f' --method {method} ' \
                    f' --encoder {encoder} ' \
                    f' --database_path {orion_db} ' \
                    f' --project {project} ' \
                    f' --monitor_training_performance True ' \
                    f' --project {project} ' \
                    f' --wandb_api_key {WANDB_API_KEY} ' \

        
        print(f'{SEQUOIA_SWEEP_PATH} {command}')
        for _ in range(n_jobs):
            os.system(f'{LAUNCH_COMMAND} {SEQUOIA_SWEEP_PATH} {command}')

        # exit()