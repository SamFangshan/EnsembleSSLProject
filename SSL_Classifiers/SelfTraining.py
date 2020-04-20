import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from copy import deepcopy


class SelfTrainingClassifier(object):
    def __init__(self, clf):
        """
        :param clf: base classifier
        """
        self.clf = clf

    def fit(self, L: pd.DataFrame, U: pd.DataFrame, ConLev = 0.8, early_stop=True):
        """
        :param L: Labeled Data
        :param U: Unlabled Data
        :param ConLev: Confidence level
        :param early_stop: stop early
        :return:self, transductive accuracy
        """
        original_length = U.shape[0]
        print("Unlabeled:", original_length)
        clf = clone(self.clf)
        X_L = L.iloc[:,:-1]
        y_L = L.iloc[:,-1]
        acc = 0
        valid_iter = 0
        last_U_shape = original_length
        while not U.empty and valid_iter <= 50:
            # Train C on L
            clf.fit(X_L, y_L)
            # Apply C on U
            X_U = U.iloc[:,:-2]
            y_U_answer = U.iloc[:,-1]
            preds = clf.predict(X_U)
            probas = clf.predict_proba(X_U)
            
            # Select instances based on ConLev
            to_add = list()
            for i in range(0, len(probas)):
                if max(probas[i]) > ConLev:
                    to_add.append(i)
            index = y_U_answer.iloc[to_add].index
            preds_to_add = deepcopy(preds[to_add])
            y_U_answer_to_add = deepcopy(y_U_answer.iloc[to_add])
            
            # Add to L
            U.iloc[:,-2] = preds
            L = L.append(U.iloc[to_add,:-1], ignore_index=True)
            
            # Remove from U
            U.drop(index, inplace=True)
            
            if last_U_shape == U.shape[0]:
                ConLev -= 0.05
            else:
                acc += accuracy_score(preds_to_add, y_U_answer_to_add)
                valid_iter += 1
            
            last_U_shape = U.shape[0]
            print("Labeled:", original_length - U.shape[0], "/", original_length)

        self.clf = clf
        return self, acc/valid_iter

    def predict(self, X: pd.DataFrame):
        return self.clf.predict(X)