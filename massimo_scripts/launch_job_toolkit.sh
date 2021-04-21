#!/bin/bash  
/mnt/home/.local/bin/eai job submit --image registry.console.elementai.com/snow.massimo/ssh \
    --cpu 4 \
    --mem 32 \
    --gpu 1 \
    -d snow.massimo.home:/mnt/home \
    --restartable \
    --workdir /mnt/home/dev/Sequoia \
    -- ${@}
    # -- ./run.sh ${@}