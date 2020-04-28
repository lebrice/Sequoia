#!/bin/bash
BELUGA_DIR="./results/beluga/TaskIncremental"
RESULTS_DIR="./results/TaskIncremental"

mv --update "$BELUGA_DIR/cifar100-10c_mh_d_*" "$RESULTS_DIR/cifar100/multihead_detached/"
