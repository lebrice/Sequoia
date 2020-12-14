#!/usr/bin/env bash

mkdir -p $SLURM_TMPDIR/data/
cp -r /network/data1/cifar/cifar-10-batches-py $SLURM_TMPDIR/data/
cp -r /network/data1/svhn $SLURM_TMPDIR/data/
