# syntax=docker/dockerfile:1
FROM lebrice/sequoia:base
USER root
SHELL [ "conda", "run", "-n", "base", "/bin/bash", "-c"]
ARG BRANCH=master
RUN conda install -y cudatoolkit
RUN cd /workspace/Sequoia && git fetch -p && git checkout ${BRANCH} && pip install -e .[no_mujoco]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "base", "/bin/bash", "-c"]
