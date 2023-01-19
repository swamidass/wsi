FROM continuumio/miniconda3:4.12.0 as builder

WORKDIR /var/task

COPY environment.yml environment.yml
COPY requirements.txt requirements.txt

# Necessary to enable building image with up2date tensorflow
ENV CONDA_OVERRIDE_CUDA 11.2.0  

RUN conda install -n base mamba -c conda-forge && \
  mamba env update -n base --file environment.yml && \
  mamba clean -ayf  && \
  pip install -r requirements.txt --no-cache \
    --find-links https://storage.googleapis.com/jax-releases/jax_cuda_releases.html && \
  pip cache purge && \
  chmod -R a+w /opt/conda

# RUN pip install -U "ray" && pip cache purge && chmod -R a+w /opt/conda

COPY test* .

ENV LD_LIBRARY_PATH /opt/conda/lib

ENV TF_FORCE_GPU_ALLOW_GROWTH true
ENV TF_XLA_FLAGS --tf_xla_enable_xla_devices
ENV TF_CPP_MIN_LOG_LEVEL 3
ENV XLA_PYTHON_CLIENT_PREALLOCATE false

#RUN pytest 

ENTRYPOINT ["/opt/conda/bin/conda", "run", "-n", "base", "--no-capture-output"]
CMD ["/bin/bash"]