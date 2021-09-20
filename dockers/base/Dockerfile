# syntax=docker/dockerfile:1
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime
USER root
EXPOSE 2222
EXPOSE 6000
EXPOSE 8088
ENV LANG=en_US.UTF-8
RUN apt update && \
    apt install -y \
    git wget zsh unzip rsync build-essential \
        ca-certificates supervisor openssh-server ssh \
        curl wget vim procps htop locales nano man net-tools iputils-ping \
        libosmesa6-dev libgl1-mesa-glx libgl1-mesa-dev libglu1-mesa-dev libglfw3 \
        libglfw3-dev freeglut3 xvfb ffmpeg curl patchelf cmake zlib1g zlib1g-dev \
        swig libopenmpi-dev aptitude screen xz-utils locate && \
    sed -i "s/# en_US.UTF-8/en_US.UTF-8/" /etc/locale.gen && locale-gen && \
    useradd -m -u 13011 -s /bin/zsh toolkit && passwd -d toolkit && \
    useradd -m -u 13011 -s /bin/zsh --non-unique console && passwd -d console && \
    useradd -m -u 13011 -s /bin/zsh --non-unique _toolchain && passwd -d _toolchain && \
    useradd -m -u 13011 -s /bin/bash --non-unique coder && passwd -d coder && \
    chown -R toolkit:toolkit /run /etc/shadow /etc/profile && \
    apt autoremove --purge && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* && \
    echo ssh >> /etc/securetty && \
    rm -f /etc/legal /etc/motd

# RUN conda install -c conda-forge opencv
RUN conda install matplotlib numpy scipy hdf5 h5py cython
# RUN pip install \ 
#     # Needed to build atari_py: (WHY don't they put it in a build_requires?)
#     lockfile 
    # fasteners \ 
    # pybullet \
    # wandb \
    # tqdm \
    # # tensorflow \
    # bs4 \
    # pandas notebook plotly tqdm pyamg lxml numba pyyaml torchmeta

# Removing this `torchtext` package, seems to be causing an import issue in pytorch!
RUN pip uninstall -y torchtext
RUN chown -R toolkit:root /workspace
RUN chmod -R 777 /workspace
# this doesn't do anything
RUN adduser toolkit sudo
RUN chown -R toolkit:root /mnt/
# RUN mkdir -p /mnt/home
RUN chmod 777 /opt/conda
RUN chmod 777 /mnt
RUN chmod -R 777 /workspace
SHELL [ "conda", "run", "-n", "base", "/bin/bash", "-c"]

## Unused zshell and oh-my-zsh stuff:
# RUN sh -c "$(wget -O- https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
# RUN sed -i 's/robbyrussell/clean/' ~/.zshrc
# RUN sed -i 's/plugins=(git)/plugins=(git debian history-substring-search)/' ~/.zshrc


# MuJoCo-related stuff:
# RUN curl -o ~/mujoco200_linux.zip -L -C - https://www.roboti.us/download/mujoco200_linux.zip
# RUN curl -o ~/mjpro150_linux.zip -L -C -  https://www.roboti.us/download/mjpro150_linux.zip
# RUN cd ~ && unzip mujoco200_linux.zip && rm mujoco200_linux.zip
# RUN cd ~ && unzip mjpro150_linux.zip && rm mjpro150_linux.zip
# RUN mkdir ~/.mujoco
# RUN mv ~/mujoco200_linux ~/.mujoco/mujoco200
# RUN mv ~/mjpro150 ~/.mujoco
# RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:~/.mujoco/mujoco200/bin" >> ~/.bashrc
# RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:~/.mujoco/mjpro150/bin" >> ~/.bashrc
# COPY mjkey.txt /home/toolkit/.mujoco/
# ENV LD_LIBRARY_PATH /home/toolkit/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}
# ENV LD_LIBRARY_PATH /home/toolkit/.mujoco/mjpro150/bin:${LD_LIBRARY_PATH}
# RUN mkdir /workspace/tools
# RUN cd /workspace/tools && git clone https://github.com/openai/mujoco-py.git && pip install -e mujoco-py

# For Wandb (TODO: Doesn't appear to work, using env variable with WANDB_API_KEY
# instead.)
# COPY .netrc /home/toolkit/.netrc
# COPY .netrc /root/.netrc
# COPY .netrc /tmp/.netrc

VOLUME /mnt/data
VOLUME /mnt/results
# USER toolkit

ENV DATA_DIR=/mnt/data
ENV RESULTS_DIR=/mnt/results
ENV WANDB_DIR=/mnt/results

# VOLUME /mnt/home
# WORKDIR /mnt/home
ENV PATH /home/toolkit/.local/bin:${PATH}
# RUN cd /workspace/tools && git clone https://github.com/openai/gym.git && cd gym && pip install -e '.[all]'
# RUN cd /workspace/tools && git clone https://github.com/openai/baselines.git && cd baselines && pip install -e .
RUN cd /workspace/ && git clone --recursive https://github.com/lebrice/Sequoia.git
RUN pip install -e /workspace/Sequoia[all]
RUN pip install -e /workspace/Sequoia/methods/continual_world
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "base", "/bin/bash", "-c"]
