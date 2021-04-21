#!/bin/sh



# THIS SEEMS TO WORK:
# setp1: the trick is to change your HOME from /tmp to /mnt/home before installing stuff
# step2: conda env create --prefix=/mnt/home/.conda/envs/sequoia -f environment.yaml
# step4: /mnt/home/.conda/envs/sequoia/bin/pip install --upgrade --force-reinstall -e .
# !!!! somehow we already have access to monsterkong
# step4: /mnt/home/.conda/envs/sequoia/bin/pip install --upgrade --force-reinstall -e .[monsterkong,hpo]
# bonus: maybe you need export PATH=$PATH:/mnt/home/.local/bin, not sure at which stage
# export PATH=$PATH:/mnt/home/.local/bin # I didnt need this before in the run.sh 
/mnt/home/.conda/envs/sequoia/bin/sequoia_sweep ${@} 

#-------------

# export PATH=$PATH:/mnt/home/.local/bin
# conda activate sequoia
# sequoia_sweep ${@} 
# conda config --append envs_dirs /mnt/home/.conda/envs/
# conda run -n sequoia sequoia_sweep ${@} 

# /mnt/home/.conda/envs/sequoia/bin/pip install -I git+https://github.com/lebrice/MetaMonsterkong.git#egg=meta_monsterkong

# pip install -e .[monsterkong,hpo]

# conda env create -f environment.yaml
# conda env create --prefix=/mnt/home/.conda/envs/sequoia_tmp -f environment.yaml 

# . `conda info --base`/etc/profile.d/conda.sh

# conda config --append envs_dirs /mnt/home/.conda/envs/
# conda run -n sequoia pip install  --upgrade --no-cache --ignore-installed -e .[monsterkong,hpo]

# /mnt/home/miniconda3/bin/conda run -n sequoia sequoia_sweep ${@}

# /opt/conda/bin/activate sequoia
# conda run -n sequoia sequoia_sweep ${@} 