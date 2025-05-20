FROM continuumio/miniconda3

WORKDIR /var/task



# Necessary to enable building image with up2date tensorflow
ENV CONDA_OVERRIDE_CUDA=12.4.1

RUN conda install -n base mamba -c conda-forge && \
  chmod -R a+w /opt/conda

COPY environment.yml environment.yml

RUN  mamba env update -n base --file environment.yml && \
  mamba clean -ayf && \
  chmod -R a+w /opt/conda 

COPY requirements.txt requirements.txt

ENV LD_LIBRARY_PATH=/opt/conda/lib

RUN  uv pip install -r requirements.txt --no-cache --system && \
  # uv pip cache purge && \
  chmod -R a+w /opt/conda

# RUN mamba install -n base conda-forge::openjdk && \
#   mamba clean -ayf && \
#   chmod -R a+w /opt/conda

# RUN uv pip install hydra-core@git+https://github.com/swamidass/hydra.git --system&& \
# # RUN uv pip install -U jupyter jupyterlab jupyterlab_code_formatter  --system && \
# # # pip cache purge && \
#   chmod -R a+w /opt/conda


ENTRYPOINT ["/opt/conda/bin/conda", "run", "-n", "base", "--no-capture-output"]
CMD ["/bin/bash"]
