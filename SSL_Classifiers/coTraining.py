from typing import Tuple

import numpy as np
import random
import copy

import pandas as pd
from sklearn import preprocessing, linear_model


class CoTrainingClassifier(object):
    def __init__(self, clf1, clf2=None, V1: list = None, V2: list = None, iteration=40, unsample=20):
        """
        :param clf1:
        :param clf2:
        :param iteration: number of iterations
        :param unsample: choose a number of unsampled data from U to form U'
        """
        self.clf1 = clf1
        # clf2 is a copy of clf1 if not specifies
        if clf2 is None:
            self.clf2 = copy.copy(clf1)
        else:
            self.clf2 = clf2

        assert V1 is not None
        assert V2 is not None

        self.V1 = V1
        self.V2 = V2

        self.iteration_ = iteration
        self.unsample = unsample

        random.seed(a=None, version=2)

    def fit(self, L: np.ndarray, U: np.ndarray):
        """
        :param L: Labeled Data
        :param U: Unlabled Data
        :param V1:  two feature conditionally independent views of instances.
        :param V2:  two feature conditionally independent views of instances.
        :return:self, transductive accuracy
        """

        # randomize here, and then just take from the back
        # so we don't have to sample every time
        random.shuffle(U)

        # Split U into two
        u_np_fract, U = np.split(U, [min(len(U), self.unsample)])

        it = 0
        # while U_np is not empty or we reach maximum iterations
        while it != self.iteration_ and U.size:
            data_l = L[:, :-1]
            labels_l = L[:, -1]

            it += 1

            # Train C1 and C2 on L(V1), L(V2) respectively
            self.clf1.fit(data_l[:, self.V1], labels_l)
            self.clf2.fit(data_l[:, self.V2], labels_l)

            prob = self.predict_proba(u_np_fract)
            classes = np.argmax(prob, axis=1)

            # Find confident prediction
            indices = np.max(prob, axis=1) >= 0.72

            # Remove P and N from U′
            u_with_label = np.column_stack((u_np_fract, classes))

            # Add P and N to L.
            L = np.concatenate((L, u_with_label[indices, :]))
            u_np_fract = u_np_fract[np.invert(indices), :]

            # Refill U' with examples from U to keep U' at constant size of u
            fill_up, U = np.split(U, [self.unsample - len(u_np_fract)])
            u_np_fract = np.concatenate((u_np_fract, fill_up))

    def predict(self, x):
        prob = self.predict_proba(x)

        return np.argmax(prob, axis=1)

    def predict_proba(self, x):
        # obtain the confidence of two classifier
        clf1_probs = self.clf1.predict_proba(x[:, self.V1])
        clf2_probs = self.clf2.predict_proba(x[:, self.V2])

        # choose the class with higher confidence
        clf1_probs_max = np.max(clf1_probs, axis=1)
        clf2_probs_max = np.max(clf2_probs, axis=1)

        prob = np.where((clf1_probs_max > clf2_probs_max).reshape(-1, 1), clf1_probs, clf2_probs)

        return prob


def prepare_data(csv_path: str, encoder: preprocessing.LabelEncoder = None):
    df = pd.read_csv(csv_path)
    example = df.head(1)

    labeled = not (example.Clas == 'unlabeled').iloc[0]

    if labeled:
        data = df.iloc[:, :-1].to_numpy()
        labels: np.ndarray = df.iloc[:, -1].to_numpy()
        labels = labels.astype(np.str)

        if encoder is None:
            encoder = preprocessing.LabelEncoder()
            labels = encoder.fit_transform(labels)
        else:
            labels = encoder.transform(labels)

        return data, labels, encoder
    else:
        data = df.iloc[:, :-2].to_numpy()
        ground_truth = df.iloc[:, -1].to_numpy()
        ground_truth = encoder.transform(ground_truth)
        return data, ground_truth


if __name__ == '__main__':
    clf = linear_model.LogisticRegression()
    co_training_classifier = CoTrainingClassifier(clf, V1=list(range(9)), V2=list(range(9, 18)))
    data_l, label_l, encoder = prepare_data("vehicle-ssl10-10-10tra-l.csv")
    data_u, ground_truth = prepare_data("vehicle-ssl10-10-10tra-u.csv", encoder=encoder)
    L = np.column_stack((data_l, label_l))
    co_training_classifier.fit(L, data_u)
    preds = co_training_classifier.predict(data_u)
    print(np.mean(preds == ground_truth))
