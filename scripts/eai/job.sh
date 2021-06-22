#!/bin/bash
set -o errexit    # Used to exit upon error, avoiding cascading errors
set -o errtrace    # Show error trace
set -o pipefail   # Unveils hidden failures
# set -o nounset    # Exposes unset variables

# Get organization name
ORG_NAME=$(eai organization get --field name)
# Get account name
ACCOUNT_NAME=$(eai account get --field name)
ACCOUNT_ID=$ORG_NAME.$ACCOUNT_NAME

# EAI_Registry=registry.console.elementai.com/$ACCOUNT_ID
EAI_Registry=registry.console.elementai.com/snow.massimo/ssh

CURRENT_BRANCH="`git branch --show-current`"
BRANCH=${BRANCH:-$CURRENT_BRANCH}
export WANDB_API_KEY=${WANDB_API_KEY?"Need to pass the wandb api key or have it set in the environment variables."}

echo "Building eai-specific container for branch $BRANCH"

if [ "$NO_BUILD" ]; then
    echo "skipping build."
else
    echo "building"
    # TODO: There is something wrong here: How can they possibly build their job, if
    # they don't have the eai dockerfile?
    source dockers/eai/build.sh
fi

# The image we're using is going to be called sequoai_eai:$BRANCH, and will have been
# pushed to the user's eai registry.

eai job submit \
    --restartable \
    --data $ACCOUNT_ID.home:/mnt/home \
    --data $ACCOUNT_ID.data:/mnt/data \
    --data $ACCOUNT_ID.results:/mnt/results \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    --env HOME=/home/toolkit \
    --image $EAI_Registry/sequoia_eai:$BRANCH \
    --gpu 1 --cpu 8 --mem 12 --gpu-model-filter 12gb \
    -- "$@"
