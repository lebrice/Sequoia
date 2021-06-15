#!/bin/bash
set -o errexit    # Used to exit upon error, avoiding cascading errors
set -o errtrace    # Show error trace
set -o pipefail   # Unveils hidden failures
set -o nounset    # Exposes unset variables

if git diff-index --quiet HEAD --; then
    # No changes
    echo "All good, no uncommitted changes."
else
    # Changes
    echo "Can't build dockers when there are uncommited changes!"
    exit 1
fi


echo "Building the 'base' dockerfile"
docker build . --file dockers/base/Dockerfile --tag sequoia:base

REGISTRY=${REGISTRY:-`docker info | sed '/Username:/!d;s/.* //'`}
echo "Using registry $REGISTRY"

docker tag sequoia:base $REGISTRY/sequoia:base
docker push $REGISTRY/sequoia:base
