import numpy as np
import os

# CHANGE ME #
LAUNCH_COMMAND = 'bash -i launch_job_toolkit.sh '
SEQUOIA_SWEEP_PATH = '/mnt/home/.conda/envs/sequoia/bin/sequoia_sweep' 
WANDB_API_KEY = "799759daba56493c8f8c0bd47660901e9efef01e"
#############

project = 'monsterkong_sweep3'
max_runs = 1000
n_jobs = 1


# SETTINGS = ['incremental_rl', 'task_incremental_rl', 'rl']
# SETTINGS = ['incremental_rl', 'rl']
SETTINGS = ['incremental_rl', 'rl']
# METHODS = ['baseline', 'ewc', 'a2c', 'ppo', 'dqn'] #'experience_replay']
# METHODS = ['baseline', 'ewc', 'a2c', 'ppo', 'dqn'] #'experience_replay']
METHODS = ['ewc', ] #'experience_replay']
# BENCHMARKS = ['monsterkong_2j_2l_4jl_100kSteps', 'monsterkong_2j_2l_4jl_200kSteps', 'monsterkong_5j_5l_100kSteps', 'monsterkong_5j_5l_200kSteps']
# BENCHMARKS = ['monsterkong_2j_2l_4jl_100kSteps', 'monsterkong_2j_2l_4jl_200kSteps', ]
BENCHMARKS = ['monsterkong_2j_2l_4jl_100kSteps', ]
# BENCHMARKS = ['monsterkong_debug']


for setting in SETTINGS:
    for benchmark in BENCHMARKS:
        for method in METHODS:
            
            if method == 'ewc' and setting=='rl':
                continue

            orion_db = f'orion/{setting}_{benchmark}_{method}.pkl'
            
            command = f'--setting {setting} ' \
                    f' --max_runs {max_runs} ' \
                        f' --method {method} ' \
                        f' --monitor_training_performance' \
                        f' --database_path {orion_db} ' \
                        f' --benchmark {benchmark} ' \
                        # f' --project {project} ' \
                        # f' --wandb_api_key {WANDB_API_KEY} ' \
                        # f' --dataset {dataset} ' \
                        # f' --steps_per_task {steps_per_task} ' \
                        # f' --nb_task {nb_task} ' \
            print(f'{SEQUOIA_SWEEP_PATH} {command}')
            for _ in range(n_jobs):
                os.system(f'{LAUNCH_COMMAND} {SEQUOIA_SWEEP_PATH} {command}')

            # exit()
