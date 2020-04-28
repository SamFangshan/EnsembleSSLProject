# CE4041/CZ4041 Machine Learning Research Project
This is a research project on semi-supervised learning (SSL) for the course CE4041/CZ4041 Machine Learning at Nanyang Technological University, Singapore
### This project runs on [Docker](http://www.docker.com)
[<img src="https://www.docker.com/sites/default/files/social/docker_facebook_share.png">](http://www.docker.com/)<br/>
Please make sure Docker is installed before you proceed. To download Docker for Mac or Windows, please click [here](https://www.docker.com/products/docker-desktop).
## 1. Build Docker Image
Run the following command to build the Docker image for this project.
```bash
docker build -t ensemblesslproject .
```
## 2. Run Docker Container
Run the following command to enter the Docker container
```bash
docker run -it ensemblesslproject /bin/bash
```
## 2. Data Preprocessing
The original data sets were retrieved from https://sci2s.ugr.es/keel/semisupervised.php. All retrived data sets are in `.dat` format. In order to make the data sets Python friendly, they are all converted to `.csv` format. Afterwards, labeled training data and unlabeled training data are separated into two files. Data sets with categorical features are all one-hot encoded, and all numerical features are minmax normalized. `data_preprocess.py` is used to perform the aboved mentioned jobs. Run the following command:
```bash
python3 data_preprocess.py
```
## 3. Primitive Classifier SSL
Run the following commands to test SSL algorithms using primitive classifiers (Refer to project report for details on the primitive classifiers used)
```bash
python3 SSL_using_primitive_classifier.py -m self # self-training
python3 SSL_using_primitive_classifier.py -m co # co-training
python3 SSL_using_primitive_classifier.py -m tri # tri-training
```

## 4. Voting Classifier SSL
Run the following commands to test SSL algorithms using voting classifiers (both weighted voting and majority voting)
```bash
python3 SSL_using_voting_classifier.py -m self # self-training
python3 SSL_using_voting_classifier.py -m co # co-training
python3 SSL_using_voting_classifier.py -m tri # tri-training
```
## 5. Comparison SSL Using SC3-MC
Run the following command to perform the comparison experiment using [SC3-MC](https://link.springer.com/article/10.1007/s11063-020-10191-1) algorithm
```bash
python3 SSL_SC3_MC.py
```
## 6. Friedman Aligned Ranks Test with Bergmann-Hommel Post Hoc Procedure
The output CSV files generated from previous sections are expected to be manually organized using tools such as Excel. The input CSV file to the R script file is expected to be in the following format, where each cell represents a prediction accuracy value:
```CSV
algorithm1,algorithm2,algorithm3,algorithm4
0.9,0.8,0.7,0.9 # dataset 1
1.0,0.9,0.9,0.7 # dataset 2
0.9,1.0,0.8.0.9 # dataset 3
```
Run the following command to perform the statistical test:
```bash
Rscript statistics/FAR.test.script.R input_filename.csv
```
