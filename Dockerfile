FROM continuumio/miniconda3:4.12.0 as builder

WORKDIR /var/task

COPY environment.yml environment.yml
COPY requirements.txt requirements.txt

# Necessary to enable building image with up2date tensorflow
ENV CONDA_OVERRIDE_CUDA 11.2.0  

RUN conda install -n base mamba conda-forge::ncurses -c conda-forge && \
  chmod -R a+w /opt/conda

RUN  mamba env update -n base --file environment.yml && \
  mamba clean -ayf  && \
  pip install -r requirements.txt --no-cache \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
  pip cache purge && \
  chmod -R a+w /opt/conda

# RUN pip install -U https://s3-us-west-2.amazonaws.com/ray-wheels/latest/ray-3.0.0.dev0-cp39-cp39-manylinux2014_x86_64.whl && pip cache purge && chmod -R a+w /opt/conda

COPY test* .

ENV LD_LIBRARY_PATH /opt/conda/lib

ENV TF_FORCE_GPU_ALLOW_GROWTH true
ENV TF_XLA_FLAGS --tf_xla_enable_xla_devices
ENV TF_CPP_MIN_LOG_LEVEL 3
ENV XLA_PYTHON_CLIENT_PREALLOCATE false
ENV CUDA_DIR /opt/conda

RUN mkdir -p /usr/local/cuda/nvvm/libdevice && \
  cp /opt/conda/nvvm/libdevice/* /usr/local/cuda/nvvm/libdevice

RUN pytest -v

ENTRYPOINT ["/opt/conda/bin/conda", "run", "-n", "base", "--no-capture-output"]
CMD ["/bin/bash"]