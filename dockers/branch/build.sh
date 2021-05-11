#!/bin/bash
set -o errexit    # Used to exit upon error, avoiding cascading errors
set -o errtrace    # Show error trace
set -o pipefail   # Unveils hidden failures
set -o nounset    # Exposes unset variables

export CURRENT_BRANCH="`git branch --show-current`"
export BRANCH=${BRANCH:-$CURRENT_BRANCH}
echo "Using branch $BRANCH"

export REGISTRY=${REGISTRY:-`docker info | sed '/Username:/!d;s/.* //'`}
echo "Using registry $REGISTRY"


if git diff-index --quiet HEAD --; then
    # No changes
    echo "all good."
else
    # Changes
    echo "Can't build dockers when you have uncommited changes!"
    exit 1
fi
git push

echo "Building the container for branch $BRANCH (no cache)"
docker build . --file dockers/branch/Dockerfile \
    --no-cache \
    --build-arg BRANCH=$BRANCH \
    --tag sequoia:$BRANCH

docker tag sequoia:$BRANCH $REGISTRY/sequoia:$BRANCH
docker push $REGISTRY/sequoia:$BRANCH
