#!/bin/bash
set -o errexit    # Used to exit upon error, avoiding cascading errors
set -o errtrace    # Show error trace
set -o pipefail   # Unveils hidden failures
set -o nounset    # Exposes unset variables

CURRENT_BRANCH="`git branch --show-current`"
BRANCH=${BRANCH:-$CURRENT_BRANCH}
echo "Using branch $BRANCH"
git push

if git diff-index --quiet HEAD --; then
    # No changes
    echo "all good"
else
    # Changes
    echo "Can't build dockers when you have uncommited changes!"
    exit 1
fi

# Get organization name
export ORG_NAME=$(eai organization get --field name)
# Get account name
export ACCOUNT_NAME=$(eai account get --field name)
export ACCOUNT_ID=$ORG_NAME.$ACCOUNT_NAME
EAI_Registry=registry.console.elementai.com/$ACCOUNT_ID
DockerHub_Registry=registry.console.elementai.com/$ACCOUNT_ID


echo "Building the 'base' dockerfile for elementai cluster."
docker build . --file dockers/eai/base.dockerfile \
    --tag $DockerHub_Registry/sequoia:eai_base --tag $EAI_Registry/sequoia:eai_base


echo "Building the container for branch $BRANCH"
docker build . --file dockers/eai/branch.dockerfile \
    --no-cache \
    --build-arg BRANCH=$BRANCH \
    --tag $DockerHub_Registry/sequoia:eai_$BRANCH --tag $EAI_Registry/sequoia:eai_$BRANCH

docker push $DockerHub_Registry/sequoia:eai_base
docker push $DockerHub_Registry/sequoia:eai_$BRANCH
docker push $EAI_Registry/sequoia:eai_base
docker push $EAI_Registry/sequoia:eai_$BRANCH