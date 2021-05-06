FROM lebrice/sequoia:eai_base
ARG BRANCH=master
RUN git fetch -p
RUN git checkout ${BRANCH} && pip install -e .[monsterkong,hpo,avalanche]
ENV DATA_DIR=/mnt/data
ENV RESULTS_DIR=/mnt/results
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "base", "/bin/bash", "-c"]
