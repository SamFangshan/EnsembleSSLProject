#!/bin/bash

# download datasets
curl -O https://sci2s.ugr.es/keel/dataset/data/semisupervised//full/ssl_10.zip
curl -O https://sci2s.ugr.es/keel/dataset/data/semisupervised//full/ssl_20.zip
curl -O https://sci2s.ugr.es/keel/dataset/data/semisupervised//full/ssl_30.zip
curl -O https://sci2s.ugr.es/keel/dataset/data/semisupervised//full/ssl_40.zip

# unzip all files
for p in "10" "20" "30" "40"
do
    unzip ssl_$p.zip
    unzip "ssl_$p/*.zip" -d ssl_$p
    rm ssl_$p/*.zip
    rm ssl_$p.zip
done
