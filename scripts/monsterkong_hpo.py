import numpy as np
import os

# CHANGE ME #
LAUNCH_COMMAND = 'bash launch_job_toolkit.sh'   
SEQUOIA_SWEEP_PATH = '/home/mila/o/ostapeno/dev/Sequoia/scripts/sequoia_sweep' 
#############

project = 'monsterkong_sweep'
max_runs = 2
n_jobs = 1


# SETTINGS = ['rl', 'incremental_rl', 'multi_task_rl', 'task_incremental_rl']
SETTINGS = ['incremental_rl', 'task_incremental_rl', 'rl']
NB_TASKS = [10, 20]
METHODS = ['baseline', 'ewc', 'a2c', 'ppo', 'dqn'] #'experience_replay']
DATASETS = ['monsterkong']
# STEPS_PER_TASKS = [100_000, 1_000_000]
STEPS_PER_TASKS = [1_000_000]

dataset = DATASETS[0]

for setting in SETTINGS:
    for nb_task in NB_TASKS:
        for method in METHODS:
            for steps_per_task in STEPS_PER_TASKS:
                
                command = f'--setting {setting} ' \
                        f' --max_runs {max_runs} ' \
                            f' --dataset {dataset} ' \
                            f' --nb_tasks {nb_task} ' \
                            f' --method {method} ' \
                            f' --project {project} ' \
                            f' --steps_per_task {steps_per_task} '
                print(f'{SEQUOIA_SWEEP_PATH} {command}')
                for _ in range(n_jobs):
                    os.system(f'{LAUNCH_COMMAND} {SEQUOIA_SWEEP_PATH} {command}')

                exit()
