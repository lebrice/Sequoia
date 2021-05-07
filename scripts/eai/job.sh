#!/bin/bash
set -o errexit    # Used to exit upon error, avoiding cascading errors
set -o errtrace    # Show error trace
set -o pipefail   # Unveils hidden failures
set -o nounset    # Exposes unset variables

# Get organization name
ORG_NAME=$(eai organization get --field name)
# Get account name
ACCOUNT_NAME=$(eai account get --field name)
ACCOUNT_ID=$ORG_NAME.$ACCOUNT_NAME

EAI_Registry=registry.console.elementai.com/$ACCOUNT_ID

CURRENT_BRANCH="`git branch --show-current`"
BRANCH=${BRANCH:-$CURRENT_BRANCH}
echo "Building container for branch $BRANCH"

source dockers/branch/build.sh

eai job submit \
    --non-preemptable \
    --data $ACCOUNT_ID.home:/mnt/home \
    --data $ACCOUNT_ID.data:/mnt/data \
    --data $ACCOUNT_ID.results:/mnt/results \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --image $EAI_Registry/sequoia:$BRANCH \
    --gpu 1 --cpu 8 --mem 12 \
    -- "$@"
