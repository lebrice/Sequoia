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

existing_interactive_job_id=`eai job ls  --state alive --fields id,interactive | grep true | awk '{print $1}'`
if [ $existing_interactive_job_id ]; then
    echo "Found existing interactive job, with id $existing_interactive_job_id"
    eai job kill $existing_interactive_job_id
fi;

eai job submit \
    --interactive \
    --data $ACCOUNT_ID.home:/mnt/home \
    --data $ACCOUNT_ID.data:/mnt/data \
    --data $ACCOUNT_ID.results:/mnt/results \
    --image registry.console.elementai.com/$ACCOUNT_ID/sequoia:base \
    --gpu 1 --cpu 8 --mem 12
