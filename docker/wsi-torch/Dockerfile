FROM mambaorg/micromamba

WORKDIR /var/task

ENV CONDA_OVERRIDE_CUDA=12.4.1

COPY --chown=$MAMBA_USER:$MAMBA_USER environment.yml environment.yml

RUN  micromamba install -n base --yes --file environment.yml && \
  micromamba clean -ayf 


COPY --chown=$MAMBA_USER:$MAMBA_USER requirements.txt requirements.txt


ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN  uv pip install -r requirements.txt --no-cache --system  &&\
    uv cache clean

CMD ["/bin/bash"]