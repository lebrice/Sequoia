#!/bin/bash

module load anaconda/3
conda activate sequoia

cd ~/Sequoia
pip install -e .[hpo,monsterkong]

sbatch scripts/sweep.sh --setting rl --dataset monsterkong --nb_tasks 10 --method baseline
sbatch scripts/sweep.sh --setting rl --dataset monsterkong --nb_tasks 10 --method ewc
# sequoia_sweep --setting iid --dataset synbols --method baseline
# sequoia_sweep --setting task_incremental  --dataset synbols --nb_tasks 4   --method baseline
# sequoia_sweep --setting task_incremental  --dataset synbols --nb_tasks 8   --method baseline
# sequoia_sweep --setting task_incremental  --dataset synbols --nb_tasks 12  --method baseline
# sequoia_sweep --setting class_incremental --dataset synbols --nb_tasks 4   --method baseline
# sequoia_sweep --setting class_incremental --dataset synbols --nb_tasks 8   --method baseline
# sequoia_sweep --setting class_incremental --dataset synbols --nb_tasks 12  --method baseline
# sequoia_sweep --setting task_incremental  --dataset synbols --nb_tasks 4   --method baseline --ewc.coefficient 1
# sequoia_sweep --setting task_incremental  --dataset synbols --nb_tasks 8   --method baseline --ewc.coefficient 1
# sequoia_sweep --setting task_incremental  --dataset synbols --nb_tasks 12  --method baseline --ewc.coefficient 1
# sequoia_sweep --setting class_incremental --dataset synbols --nb_tasks 4   --method baseline --ewc.coefficient 1
# sequoia_sweep --setting class_incremental --dataset synbols --nb_tasks 8   --method baseline --ewc.coefficient 1
# sequoia_sweep --setting class_incremental --dataset synbols --nb_tasks 12  --method baseline --ewc.coefficient 1
# sequoia_sweep --setting task_incremental  --dataset synbols --nb_tasks 4   --method hat
# sequoia_sweep --setting task_incremental  --dataset synbols --nb_tasks 8   --method hat
# sequoia_sweep --setting task_incremental  --dataset synbols --nb_tasks 12  --method hat
# sequoia_sweep --setting task_incremental  --dataset synbols --nb_tasks 4   --method pnn
# sequoia_sweep --setting task_incremental  --dataset synbols --nb_tasks 8   --method pnn
# sequoia_sweep --setting task_incremental  --dataset synbols --nb_tasks 12  --method pnn
sequoia_sweep --setting rl --benchmark monsterkong_jumps  --method a2c
# sequoia_sweep --setting rl --benchmark monsterkong_jumps  --method ppo
# sequoia_sweep --setting rl --benchmark monsterkong_jumps  --method dqn
# sequoia_sweep --setting rl --benchmark monsterkong_jumps  --method baseline
# sequoia_sweep --setting rl --benchmark monsterkong_ladders --method a2c
# sequoia_sweep --setting rl --benchmark monsterkong_ladders --method ppo
# sequoia_sweep --setting rl --benchmark monsterkong_ladders --method dqn
# sequoia_sweep --setting rl --benchmark monsterkong_ladders --method baseline
# sequoia_sweep --setting rl --benchmark monsterkong_jumps_and_ladders --method a2c
# sequoia_sweep --setting rl --benchmark monsterkong_jumps_and_ladders --method ppo
# sequoia_sweep --setting rl --benchmark monsterkong_jumps_and_ladders --method dqn
# sequoia_sweep --setting rl --benchmark monsterkong_jumps_and_ladders --method baseline
# sequoia_sweep --setting rl --benchmark monsterkong_5_of_each --method a2c
# sequoia_sweep --setting rl --benchmark monsterkong_5_of_each --method ppo
# sequoia_sweep --setting rl --benchmark monsterkong_5_of_each --method dqn
# sequoia_sweep --setting rl --benchmark monsterkong_5_of_each --method baseline
# sequoia_sweep --setting rl --benchmark monsterkong_all --method a2c
# sequoia_sweep --setting rl --benchmark monsterkong_all --method ppo
# sequoia_sweep --setting rl --benchmark monsterkong_all --method dqn
# sequoia_sweep --setting rl --benchmark monsterkong_all --method baseline
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_jumps  --method a2c
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_jumps  --method ppo
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_jumps  --method dqn
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_jumps  --method baseline
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_ladders --method a2c
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_ladders --method ppo
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_ladders --method dqn
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_ladders --method baseline
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_jumps_and_ladders --method a2c
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_jumps_and_ladders --method ppo
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_jumps_and_ladders --method dqn
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_jumps_and_ladders --method baseline
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_5_of_each --method a2c
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_5_of_each --method ppo
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_5_of_each --method dqn
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_5_of_each --method baseline
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_all --method a2c
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_all --method ppo
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_all --method dqn
# sequoia_sweep --setting task_incremental_rl --benchmark monsterkong_all --method baseline
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_jumps  --method a2c
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_jumps  --method ppo
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_jumps  --method dqn
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_jumps  --method baseline
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_ladders --method a2c
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_ladders --method ppo
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_ladders --method dqn
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_ladders --method baseline
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_jumps_and_ladders --method a2c
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_jumps_and_ladders --method ppo
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_jumps_and_ladders --method dqn
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_jumps_and_ladders --method baseline
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_5_of_each --method a2c
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_5_of_each --method ppo
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_5_of_each --method dqn
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_5_of_each --method baseline
# sequoia_sweep --setting incremental_rl --benchmark monsterkong_all --method a2c