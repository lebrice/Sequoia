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

EAI_Registry=registry.console.elementai.com/$ACCOUNT_ID

CURRENT_BRANCH="`git branch --show-current`"
BRANCH=${BRANCH:-$CURRENT_BRANCH}

existing_interactive_job_id=`eai job ls  --state alive --fields id,interactive | grep true | awk '{print $1}'`
if [ $existing_interactive_job_id ]; then
    echo "Found existing interactive job, with id $existing_interactive_job_id"
    eai job kill $existing_interactive_job_id
    echo "Sleeping for 5 seconds, just to give the job a chance to change its status."
    sleep 5
fi;

echo "building"
source dockers/eai/build.sh

# The image we're using is going to be called sequoai_eai:$BRANCH, and will have been
# pushed to the user's eai registry.

eai job submit \
    --interactive \
    --data $ACCOUNT_ID.home:/mnt/home \
    --data $ACCOUNT_ID.data:/mnt/data \
    --data $ACCOUNT_ID.results:/mnt/results \
    --env WANDB_API_KEY="$WANDB_API_KEY" \
    --image $EAI_Registry/sequoia_eai:$BRANCH \
    --gpu 1 --cpu 8 --mem 12 --gpu-mem 12
