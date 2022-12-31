FROM continuumio/miniconda3:4.12.0 as builder

WORKDIR /var/task

COPY environment.yml environment.yml
COPY requirements.txt requirements.txt

RUN conda install -n base mamba -c conda-forge && \
  mamba env update -n base --file environment.yml && \
  mamba clean -ayf && \
  pip install -r requirements.txt --no-cache && \
  pip uninstall jaxlib Pillow -y --no-cache && \
  pip install Pillow jaxlib --no-cache  \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html  && \
  pip cache purge && \
  chmod -R a+w /opt/conda

WORKDIR /var/task

COPY test* .

RUN pytest 

ENV TF_FORCE_GPU_ALLOW_GROWTH true
ENV XLA_PYTHON_CLIENT_PREALLOCATE false

ENTRYPOINT ["/opt/conda/bin/conda", "run", "-n", "base", "--no-capture-output"]
CMD ["/bin/bash"]