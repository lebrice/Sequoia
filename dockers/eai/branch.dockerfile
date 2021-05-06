FROM lebrice/sequoia:eai_base
ARG BRANCH=master
RUN git fetch -p
RUN git checkout ${BRANCH} && pip install -e .[monsterkong,hpo,avalanche]
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "base", "/bin/bash", "-c"]
