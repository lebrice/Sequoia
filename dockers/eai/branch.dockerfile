FROM lebrice/sequoia:eai_base
ARG BRANCH=master
RUN git fetch -p
RUN git checkout ${BRANCH} && pip install -e .[monsterkong,hpo,avalanche]
CMD ["/tk/bin/start.sh"]