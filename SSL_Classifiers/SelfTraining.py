import numpy as np
import random
import pandas as pd
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score


class SelfTrainingClassifier(object):
    def __init__(self, clf, iteration=None, unsample=250, to_predict=25, view_ratio=0.5):
        """
        :param clf:
        :param iteration: number of iterations
        :param unsample: choose a number of unsampled data from U to form U'
        :param to_predict: number of instances from U' to predict for each class
        :param view_ratio: view split ratio
        """
        self.clf = clf
        self.iteration_ = iteration
        self.unsample = unsample
        self.to_predict = to_predict
        self.view_ratio = view_ratio

        self._encoder = None

        random.seed(a=None, version=2)

    def fit(self, L: pd.DataFrame, U: pd.DataFrame, early_stop=True):
        """
        :param L: Labeled Data
        :param U: Unlabled Data
        :return:self, transductive accuracy
        """

        # randomize here, and then just take from the back
        # so we don't have to sample every time
        U = shuffle(U)
        half_total_unlabeled = U.shape[0] // 2
        print("Unlabeled size:", U.shape[0])

        # preprocess
        # extract the label
        labels_to_fit = np.append(L.iloc[:, -1].values, U.iloc[:, -1].values)
        # seperate data and label, encode the label
        data_l, label_l = self._prepare_data(L, labels_to_fit)
        U, ground_truth = self._prepare_data(U)
        # stack two arrays into one 2-D array
        L = np.column_stack((data_l, label_l))

        # Split U into two
        u_np_fract, U = np.split(U, [min(len(U), self.unsample)])
        ground_truth_np_fract, ground_truth = np.split(ground_truth, [min(len(ground_truth), self.unsample)])

        iterations = 0
        accuracy = 0
        stop_count = 0
        prev_trans = 0
        predicted = 0

        # while U_np is not empty or we reach maximum iterations
        while (self.iteration_ is None or iterations != self.iteration_) and (U.size or iterations == 0):
            # seperate data and labels (as two arrays)
            data_l = L[:, :-1]
            labels_l = L[:, -1]

            iterations += 1

            # Train C1 and C2 on L(V1), L(V2) respectively
            self.clf.fit(data_l, labels_l)

            prob = self.predict_proba(u_np_fract)
            classes = np.argmax(prob, axis=1)

            # Find confident prediction
            output_prob = np.max(prob, axis=1)
            to_predict_indices = np.array([])
            for cls in np.unique(labels_l):
                cls_indices = np.where(classes == cls)[0]
                output_prob_cls = output_prob[cls_indices]
                indices_conf = (-output_prob_cls).argsort()[:self.to_predict]
                indices = cls_indices[indices_conf]
                to_predict_indices = np.append(to_predict_indices, indices)
            indices = np.zeros(len(output_prob), dtype=bool)
            to_predict_indices = to_predict_indices.astype(np.int64)
            np.put(indices, to_predict_indices, True)

            # Remove P and N from U′
            u_with_label = np.column_stack((u_np_fract, classes))

            # compile transductive accuracy
            trans = accuracy_score(classes[indices], ground_truth_np_fract[indices])
            accuracy += trans
            predicted += len(to_predict_indices)
            print("Predicted:", predicted)

            # Add P and N to L.
            L = np.concatenate((L, u_with_label[indices, :]))
            u_np_fract = u_np_fract[np.invert(indices), :]
            ground_truth_np_fract = ground_truth_np_fract[np.invert(indices)]

            # early stop
            if (trans - prev_trans) < 0.001 and early_stop and half_total_unlabeled <= predicted:
                stop_count += 1
                print("stop_count", stop_count)
                if stop_count >= 5:
                    break
            else:
                if stop_count > 0:
                    stop_count = 0

            # Refill U' with examples from U to keep U' at constant size of u
            fill_up, U = np.split(U, [self.unsample - len(u_np_fract)])
            u_np_fract = np.concatenate((u_np_fract, fill_up))
            gt_fill_up, ground_truth = np.split(ground_truth, [self.unsample - len(ground_truth_np_fract)])
            ground_truth_np_fract = np.concatenate((ground_truth_np_fract, gt_fill_up))

            prev_trans = trans

        return self, accuracy/iterations

    def predict(self, X: pd.DataFrame):
        X = X.values
        prob = self.predict_proba(X)

        return self._encoder.inverse_transform(np.argmax(prob, axis=1))

    def predict_proba(self, X):
        # obtain the confidence and return prob
        probs = self.clf.predict_proba(X)
        return probs

    def _prepare_data(self, df: pd.DataFrame, labels_to_fit=None):
        example = df.head(1)

        labeled = example.iloc[0, -2] != 'unlabeled'

        if labeled:
            data = df.iloc[:, :-1].values
            labels: np.ndarray = df.iloc[:, -1].values

            if self._encoder is None:
                self._encoder = preprocessing.LabelEncoder()
                if labels_to_fit is not None:
                    self._encoder.fit(labels_to_fit)
                    labels = self._encoder.transform(labels)
                else:
                    labels = self._encoder.fit_transform(labels)
            else:
                labels = self._encoder.transform(labels)

            return data, labels
        else:
            data = df.iloc[:, :-2].values
            ground_truth = df.iloc[:, -1].values
            ground_truth = self._encoder.transform(ground_truth)
            return data, ground_truth