import os
import pandas as pd
from sklearn.base import clone
from scikit_learn_weka.wrapper import WekaWrapper, ScikitLearnWekaWrapper
from weka.classifiers import Classifier
from sklearn.metrics import accuracy_score
from SSL_Classifiers.SC3MC import SC3MCClassifier
from SSL_Classifiers.CoTraining import CoTrainingClassifier
from SSL_Classifiers.TriTraining import TriTrainingClassifier
from SSL_Classifiers.SelfTraining import SelfTrainingClassifier

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
        ssl_clf, tra_acc = SelfTrainingClassifier(clf).fit(L, U)
    elif mode == "co":
        ssl_clf, tra_acc = CoTrainingClassifier(clf).fit(L, U)
    elif mode == "tri":
        ssl_clf, tra_acc = TriTrainingClassifier(clf).fit(L, U)
    elif mode == "sc3mc":
        ssl_clf, tra_acc = SC3MCClassifier(clf).fit(L, U)

    y_pred = ssl_clf.predict(X_test)
    ind_acc = accuracy_score(y_test, y_pred)
    return tra_acc, ind_acc

# Conduct cross validation across all partitions of a specific dataset
def cross_validation(clf, dataset, percentage, mode="self", clf_name="unknown"):
    print("Cross Validation: {} {}%".format(dataset, percentage))
    cv_result_file = "crossvalidation.csv"
    if not os.path.exists(cv_result_file):
        f = open(cv_result_file,"w+")
        f.write('base_classifier,dataset,percentage,mode,iteration,transductive,inductive\n')
        f.close()
        
    already_done = -1
    df_result = pd.read_csv(cv_result_file)
    df_result = df_result[(df_result["base_classifier"] == clf_name) & (df_result["dataset"] == dataset) & 
              (df_result["percentage"] == percentage) & (df_result["mode"] == mode)]
    if not df_result.empty:
        already_done = df_result['iteration'].max()
    
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
        if p <= already_done:
            print("Already done: ({}/{})".format(p, already_done))
            tra_acc = list(df_result[df_result["iteration"] == p]["transductive"])[0]
            ind_acc = list(df_result[df_result["iteration"] == p]["inductive"])[0]
        else:
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
            f = open(cv_result_file,"a+")
            to_be_written = '{},{},{},{},{},{},{}\n'.format(clf_name, dataset, percentage, mode, p, tra_acc, ind_acc)
            print(to_be_written)
            f.write(to_be_written)
            f.close()
        
        tra_acc_avg += tra_acc
        ind_acc_avg += ind_acc
        
        
    tra_acc_avg /= 10
    ind_acc_avg /= 10
        
    print("Average transductive accuracy: {}".format(tra_acc_avg))
    print("Average inductive accuracy: {}".format(ind_acc_avg))
    return tra_acc_avg, ind_acc_avg
