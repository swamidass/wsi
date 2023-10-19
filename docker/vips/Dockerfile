FROM continuumio/miniconda3:4.12.0 as builder

WORKDIR /var/task

# COPY environment.yml environment.yml
# COPY requirements.txt requirements.txt

RUN conda update -n base -c defaults conda && \
  conda install -n base libvips -c conda-forge && \
  chmod -R a+w /opt/conda

# RUN  mamba env update -n base --file environment.yml && \
#   mamba clean -ayf  && \
#   # pip install -r requirements.txt --no-cache \
#   # pip cache purge && \
#   chmod -R a+w /opt/conda

ENV LD_LIBRARY_PATH /opt/conda/lib

ENTRYPOINT ["/opt/conda/bin/conda", "run", "-n", "base", "--no-capture-output"]
CMD ["/bin/bash"]
