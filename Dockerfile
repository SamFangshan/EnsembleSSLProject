FROM rappdw/docker-java-python

COPY . /EnsembleSSLProject

WORKDIR /EnsembleSSLProject

RUN apt update \
    && apt -y install r-base \
    && apt -y install libgdal-dev \
    && apt -y install libudunits2-dev \
    && echo 'a' | Rscript install.dependencies.R \
    && rm install.dependencies.R

