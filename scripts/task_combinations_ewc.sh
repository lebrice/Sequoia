#!/bin/bash

NAME="${1:?'Name must be set'}"
N_JOBS="${2:?'N_JOBS must be set'}"
ARGS="${@:3}"

echo "Sweep with name '$NAME' and with args '$ARGS'"
echo "Number of jobs per task: $N_JOBS"

EWC_ARGS="--ewc.coef 100"

./scripts/task_combinations.sh ${NAME}       $N_JOBS $ARGS
./scripts/task_combinations.sh ${NAME}_ewc   $N_JOBS $ARGS $EWC_ARGS
./scripts/task_combinations.sh ${NAME}_d     $N_JOBS $ARGS --detach_classifier
./scripts/task_combinations.sh ${NAME}_d_ewc $N_JOBS $ARGS --detach_classifier $EWC_ARGS
