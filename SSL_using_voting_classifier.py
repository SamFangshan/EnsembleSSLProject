import argparse
import weka.core.jvm as jvm
from ML_functions.functions import *
import numpy as np
import pandas as pd

def test(mode):
    clf_weights = [[2,1.5,1], [1,1,1]]
    datasets = ['banana', 'glass', 'lymphography', 'breast', 'flare', 'titanic', 'led7digit', 'zoo', 'wisconsin', 'iris']
    percentages = [10, 20, 30, 40]
    # prepare file to store experiment result
    result_file = "{}_voting.csv".format(mode)
    if not os.path.exists(result_file):
        f = open(result_file,"w+")
        f.write('base_classifier,dataset,percentage,transductive,inductive\n')
        already_finished_iterations = 0
        f.close()
    else:
        f = open(result_file, 'r')
        lines = f.readlines()
        already_finished_iterations = len(lines) - 1
        f.close()
    i = 0
    for dataset in datasets:
        for percentage in percentages:
            for weights in clf_weights:
                i += 1
                # skip previously finished iterations
                if i <= already_finished_iterations:
                    continue
                # cross validation
                clf = get_ensemble(weights)
                if weights == [1,1,1]:
                    clf_name = "Majority"
                else:
                    clf_name = "Weighted"
                tra, ind = cross_validation(clf, dataset, percentage, mode=mode, clf_name=clf_name)
                # write results
                f = open(result_file,"a+")
                to_be_written = '{},{},{},{},{}\n'.format(clf_name, dataset, percentage, tra, ind)
                f.write(to_be_written)
                f.close()
                
if __name__ == "__main__":
    jvm.start()
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('-m', type=str)
    args = parser.parse_args()
    test(args.m)