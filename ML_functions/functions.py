import pandas as pd
from sklearn.base import clone
from scikit_learn_weka.wrapper import WekaWrapper, ScikitLearnWekaWrapper
from weka.classifiers import Classifier
from sklearn.metrics import accuracy_score
from SSL_Classifiers import *

DATASETS = ['satimage', 'glass', 'hepatitis', 'magic', 'bupa', 'wisconsin', 'lymphography', 'phoneme', 'titanic', 'abalone', 'spambase', 'sonar', 'wine', 'vowel', 'australian', 'vehicle', 'spectfheart', 'contraceptive', 'twonorm', 'breast', 'crx', 'ecoli', 'saheart', 'automobile', 'segment', 'penbased', 'flare', 'movement_libras', 'banana', 'coil2000', 'tic-tac-toe', 'ring', 'dermatology', 'page-blocks', 'thyroid', 'mammographic', 'iris', 'german', 'cleveland', 'haberman', 'monk-2', 'nursery', 'appendicitis', 'zoo', 'pima', 'chess', 'splice', 'mushroom', 'heart', 'tae', 'led7digit', 'yeast', 'marketing', 'housevotes', 'texture']

# Use a factory pattern obtain a primitive classifier
def get_base_classifier(clf_name):
    if clf_name == "NB":
        classname = "weka.classifiers.bayes.NaiveBayes"
    elif clf_name == "MLP":
        classname = "weka.classifiers.functions.MultilayerPerceptron"
    elif clf_name == "SMO":
        classname = "weka.classifiers.functions.SMO"
    elif clf_name == "LMT":
        classname = "weka.classifiers.trees.LMT"
    elif clf_name == "PART":
        classname = "weka.classifiers.rules.PART"
    elif clf_name == "3NN":
        classname = "weka.classifiers.lazy.IBk"
    elif clf_name == "C4.5":
        classname = "weka.classifiers.trees.J48"
    elif clf_name == "RIPPER":
        classname = "weka.classifiers.rules.JRip"
        
    return ScikitLearnWekaWrapper(WekaWrapper(Classifier(classname=classname), classname))

# Train and Validate on a specific dataset
def train_and_validate(clf, L, U, X_test, y_test, mode="self"):
    if mode == "self":
        ssl_clf, tra_acc = SelfTraining(clf).fit(L, U)
    elif mode == "co":
        ssl_clf, tra_acc = CoTraining(clf).fit(L, U)
    elif mode == "tri":
        ssl_clf, tra_acc = TriTraining(clf).fit(L, U)

    y_pred = ssl_clf.predict(X_test)
    ind_acc = accuracy_score(y_test, y_pred)
    return tra_acc, ind_acc

# Conduct cross validation across all partitions of a specific dataset
def cross_validation(clf, dataset, percentage, mode="self"):
    df_u_list = []
    df_l_list = []
    df_t_list = []
    for p in range(1, 11):
        filename_u = 'ssl_{}/{}-ssl{}/{}-ssl{}-10-{}{}.csv'.format(percentage, dataset,
                                                             percentage, dataset,
                                                             percentage, p, 'tra-u')
        filename_l = 'ssl_{}/{}-ssl{}/{}-ssl{}-10-{}{}.csv'.format(percentage, dataset,
                                                             percentage, dataset,
                                                             percentage, p, 'tra-l')
        filename_t = 'ssl_{}/{}-ssl{}/{}-ssl{}-10-{}{}.csv'.format(percentage, dataset,
                                                             percentage, dataset,
                                                             percentage, p, 'tst')
        
        df_u = pd.read_csv(filename_u)
        df_l = pd.read_csv(filename_l)
        df_t = pd.read_csv(filename_t)
        
        df_u_list.append(df_u)
        df_l_list.append(df_l)
        df_t_list.append(df_t)
    
    tra_acc_avg = 0
    ind_acc_avg = 0
    for p in range(0, 10):
        X_test = df_t_list[p].iloc[:,0:-1]
        y_test = df_t_list[p].iloc[:,-1]
        
        U = None
        L = None
        for i in range(0, 10):
            if i == p:
                continue
            if U is None:
                U = df_u_list[i]
            else:
                U = U.append(df_u_list[i], ignore_index = True) 
            if L is None:
                L = df_l_list[i]
            else:
                L = L.append(df_l_list[i], ignore_index = True)
        clf_copy = clone(clf)
        tra_acc, ind_acc = train_and_validate(clf_copy, L, U, X_test, y_test, mode=mode)
        tra_acc_avg += tra_acc
        ind_acc_avg += ind_acc
        
    tra_acc_avg /= 10
    ind_acc_avg /= 10
        
    print("Average transductive accuracy: {}".format(tra_acc_avg))
    print("Average inductive accuracy: {}".format(ind_acc_avg))
    return tra_acc_avg, ind_acc_avg

# Average performance of the classifier on datasets with a specific labeled percentage
def avg_by_percentage(clf, percentage, mode="self"):
    tra_avg = 0
    ind_avg = 0
    for dataset in DATASETS:
        tra, ind = cross_validation(clf, dataset, percentage, mode="self")
        tra_avg += tra
        ind_avg += ind
    tra_avg /= len(DATASETS)
    ind_avg /= len(DATASETS)
    return tra_avg, ind_avg
