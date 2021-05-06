FROM lebrice/sequoia:eai_base
USER root
SHELL [ "conda", "run", "-n", "base", "/bin/bash", "-c"]
ARG BRANCH=master
RUN git fetch -p
RUN cd /workspace/Sequoia && git checkout ${BRANCH} && pip install -e .[monsterkong,hpo,avalanche]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "base", "/bin/bash", "-c"]
