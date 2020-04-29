FROM rappdw/docker-java-python

COPY . /EnsembleSSLProject

WORKDIR /EnsembleSSLProject

RUN pip3 install -r requirements.txt \
    && python3 cheat_scikit.py \
    && rm cheat_scikit.py \
    && apt update \
    && apt -y install r-base \
    && apt -y install libgdal-dev \
    && apt -y install libudunits2-dev \
    && echo 'a' | Rscript install.dependencies.R \
    && rm install.dependencies.R
    && ./fetch_datasets.sh \
    && rm fetch_datasets.sh