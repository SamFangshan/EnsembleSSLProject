from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
import numpy as np
import pandas as pd
import statistics

class SimpleVotingClassifier:
    """
    A weighted voting classifier that uses already trained base classifiers

    Attributes:
    _estimators: a list of base classifiers
    _weights: a list of weights of base classifiers
    """
    def __init__(self, estimators, weights):
        self._estimators = estimators
        self._weights = weights
    
    def predict(self, X):
        ind_preds = []
        for e in self._estimators:
            ind_preds.append(e[1].predict(X))
        preds = []
        for i in range(0, len(ind_preds[0])):
            stat = {}
            max_l = ind_preds[0][i]
            acc_w = 0
            for ind_pred, w in zip(ind_preds, self._weights):
                try:
                    stat[ind_pred[i]] += w
                except KeyError:
                    stat[ind_pred[i]] = w
                if stat[ind_pred[i]] > acc_w:
                    max_l = ind_pred[i]
                    acc_w = stat[ind_pred[i]]
            
            preds.append(max_l)
        preds = np.array(preds)
        return preds
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

class SC3MCClassifier:
    """
    An SSL classifier based on https://link.springer.com/article/10.1007/s11063-020-10191-1

    Attributes:
    clf: supervised learning classifier object
    sec_clfs: classifiers used for security verification
    """
    def __init__(self, clf, sec_clfs=None):
        self.clf = clf
        if sec_clfs == None:
            self.sec_clfs = [clone(self.clf), clone(self.clf), clone(self.clf)]
        else:
            self.sec_clfs = sec_clfs
        self.vc = None

    def _calc_dist(self, data, s):
        d = 0
        count = 0
        for r in s.iterrows():
            s_data = list(r[1])[0:len(r[1])-1]
            d += distance.euclidean(data, s_data)
            count += 1
        return d/count

    def _get_vc(self, clfs):
        estimators = []
        weights = []
        i = 1
        for c in clfs:
            estimators.append(("clf{}".format(i), c))
            weights.append(clfs[c])
            i += 1
        vc = SimpleVotingClassifier(estimators=estimators, weights=weights)
        return vc

    def _train_clfs(self, clfs, L, sample_frac):
        for clf in clfs:
            sample = L.sample(frac = sample_frac)
            X_s = sample.iloc[:,0:sample.shape[1]-1]
            y_s = sample.iloc[:,-1]
            clf.fit(X_s, y_s)

    def fit(self, L, U, test_frac=0.1, k=3, sample_frac=0.4, v=0.01, a=0.1, early_stop=True):
        # initialization
        X_L = L.iloc[:,0:L.shape[1]-1]
        y_L = L.iloc[:,-1]
        X_L, X_T, y_L, y_T = train_test_split(X_L, y_L, test_size=test_frac)
        clf0 = clone(self.clf)
        clf0.fit(X_L, y_L)
        error1 = 1 - clf0.score(X_T, y_T)
        X_L_copy = X_L.copy()
        X_L_copy[L.columns[-1]] = y_L
        L = X_L_copy
        X_T_copy = X_T.copy()
        X_T_copy[L.columns[-1]] = y_T
        T = X_T_copy
        acc = 0
        predicted = 0
        half_total_unlabeled = U.shape[0]//2
        stop_count = 0
        end_now = False
        print("Unlabeled:", U.shape[0])

        clfs = {}
        for i in range(0, k):
            sample = L.sample(frac = sample_frac)
            X_s = sample.iloc[:,0:sample.shape[1]-1]
            y_s = sample.iloc[:,-1]
            clf = clone(self.clf)
            clf.fit(X_s, y_s)
            clfs[clf] = 1/k

        while not U.empty and not end_now:
            # filtering unlabeled samples
            labels = L.iloc[:,-1].unique()
            subsets = []
            for l in labels:
                label_name = L.columns[-1]
                subsets.append(L[L[label_name] == l])
            for r in U.iterrows():
                index = r[0]
                data = list(r[1])[0:len(r[1])-2]
                distances = []
                for s in subsets:
                    d = self._calc_dist(data, s)
                    distances.append(d)
                dist = statistics.stdev(distances)
                U.drop(index, inplace=True)
                print("Unlabeled:", U.shape[0])
                # classifier prediction
                if dist >= v:
                    # weighted voting mechanism
                    vc = self._get_vc(clfs)
                    pl = vc.predict([data])[0]
                    for c in clfs:
                        cl = c.predict([data])[0]
                        # weight updating mechanism
                        if cl == pl:
                            clfs[c] += a
                        else:
                            clfs[c] -= a
                    # decision judgement
                    data_copy = data.copy()
                    data_copy.append(pl)
                    L_prime = L.append(pd.Series(data_copy, index=L.columns), ignore_index=True)
                    X_L = L_prime.iloc[:,0:L_prime.shape[1]-1]
                    y_L = L_prime.iloc[:,-1]
                    clf0 = clone(self.clf)
                    clf0.fit(X_L, y_L)
                    error2 = 1 - clf0.score(X_T, y_T)
                    if error1 - error2 < 0.005 and half_total_unlabeled > U.shape[0] and early_stop:
                        stop_count += 1
                        if stop_count >= 3:
                            end_now = True
                            break
                    else:
                        if stop_count > 0:
                            stop_count = 0
                    # security verification
                    if error2 <= error1:
                        X_L = L.iloc[:,0:L.shape[1]-1]
                        y_L = L.iloc[:,-1]
                        X_S, X_1, y_S, y_1 = train_test_split(X_L, y_L, test_size=0.33)
                        X_2, X_3, y_2, y_3 = train_test_split(X_S, y_S, test_size=0.50)
                        datasets = [(X_1, y_1), (X_2, y_2), (X_3, y_3)]
                        for i in range(0, 3):
                            dataset = datasets[i]
                            clf = self.sec_clfs[i]
                            try:
                                clf.fit(dataset[0], dataset[1])
                            except Exception:
                                continue
                            if clf.predict([data])[0] != pl:
                                continue
                        # finalize
                        L = L_prime
                        error1 = error2
                        self._train_clfs(clfs, L, sample_frac)
                        predicted += 1
                        if pl == list(r[1])[-1]:
                            acc += 1
        # end
        L = L.append(T, ignore_index=True)
        self._train_clfs(clfs, L, sample_frac)
        self.vc = self._get_vc(clfs)
        return self, acc/predicted

    def predict(self, X):
        return self.vc.predict(X)

    def predict_proba(self, X):
        return self.vc.predict(X)

    def score(self, X, y):
        return self.vc.score(X, y)



