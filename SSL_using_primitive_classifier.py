import argparse
import weka.core.jvm as jvm
from ML_functions.functions import *
import numpy as np
import pandas as pd

def test(mode):
    clf_names = ["NB", "MLP", "SMO", "LMT", "PART", "3NN", "C4.5", "RIPPER"]
    datasets = ['hepatitis', 'lymphography', 'iris', 'automobile', 'wine', 'sonar', 'glass', 'tae', 'spectfheart', 'zoo', 'heart', 'cleveland', 'breast', 'ecoli']
    percentages = [10, 20, 30, 40]
    # prepare file to store experiment result
    result_file = "{}.csv".format(mode)
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
            for clf_name in clf_names:
                i += 1
                # skip previously finished iterations
                if i <= already_finished_iterations:
                    continue
                # cross validation
                clf = get_base_classifier(clf_name)
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