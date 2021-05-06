FROM lebrice/sequoia:eai_base
USER root
SHELL [ "conda", "run", "-n", "base", "/bin/bash", "-c"]
ARG BRANCH=master
RUN cd /workspace/Sequoia && git fetch -p && git checkout ${BRANCH} && pip install -e .[monsterkong,hpo,avalanche]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "base", "/bin/bash", "-c"]
