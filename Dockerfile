FROM rappdw/docker-java-python

COPY . /EnsembleSSLProject

WORKDIR /EnsembleSSLProject

RUN pip3 install -r requirements.txt \
    && apt update \
    && apt -y install r-base \
    && ./fetch_datasets.sh
