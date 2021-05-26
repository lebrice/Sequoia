#!/bin/bash
set -o errexit    # Used to exit upon error, avoiding cascading errors
set -o errtrace    # Show error trace
set -o pipefail   # Unveils hidden failures
# set -o nounset    # Exposes unset variables

source scripts/eai/setup.sh
sequoia --help
