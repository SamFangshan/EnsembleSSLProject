from sklearn.base import MultiOutputMixin, BaseEstimator
from sklearn.metrics import accuracy_score
from abc import ABCMeta
import numpy as np
import pandas as pd
from weka.core.dataset import create_instances_from_matrices
from weka.filters import Filter

class ScikitLearnWekaWrapper(MultiOutputMixin, BaseEstimator, metaclass=ABCMeta):
    """
    A class to wrap Weka classifiers to be compatible with Scikit-Learn classifiers

    Attributes:
    _clf: Weka classifier object
    """

    def __init__(self, clf=None):
        self._clf = clf

    def _get_prediction_dataset(self, X):
        # convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            raise Exception("Incompatible data type")
        dataset = create_instances_from_matrices(X, name="generated from matrices")
        dataset.insert_attribute(self._class_att, dataset.num_attributes)
        dataset.class_is_last()
        return dataset

    def _get_training_dataset(self, X, y):
        # convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            raise Exception("Incompatible data type")
        if isinstance(y, pd.Series):
            y = y.values
        elif not isinstance(y, np.ndarray):
            raise Exception("Incompatible data type")

        if y.dtype == "O":
            for i in range(0, len(y)):
                y[i] = y[i].encode()
        dataset = create_instances_from_matrices(X, y, name="generated from matrices") # generate dataset

        # convert label to nominal
        try:
            y.astype(float)
            self._label_type = np.float64
            nominal = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal", options=["-R", "last"])
        except ValueError:
            self._label_type = str
            nominal = Filter(classname="weka.filters.unsupervised.attribute.StringToNominal", options=["-R", "last"])
        nominal.inputformat(dataset)
        dataset = nominal.filter(dataset)

        # sort labels
        sorter = Filter(classname="weka.filters.unsupervised.attribute.SortLabels")
        sorter.inputformat(dataset)
        dataset = sorter.filter(dataset)

        dataset.class_is_last() # indicate class label

        return dataset

    def _store_class_labels(self, dataset):
        self._class_att = dataset.class_attribute # store label attribute
        labels = str(self._class_att).split()[2][1:-1].split(',')
        if self._label_type == np.float64:
            labels = [np.float64(n) for n in labels]
        self.classes_ = np.array(labels)

    def predict(self, X):
        dataset = self._get_prediction_dataset(X)
        preds = []
        for index, inst in enumerate(dataset):
            preds.append(self.classes_[int(self._clf.classify_instance(inst))])
        preds = np.array(preds)
        return preds

    def predict_proba(self, X):
        dataset = self._get_prediction_dataset(X)
        dists = []
        for index, inst in enumerate(dataset):
            dists.append(self._clf.distribution_for_instance(inst))
        dists = np.array(dists)
        return dists

    def fit(self, X, y):
        # transform dataset
        dataset = self._get_training_dataset(X, y)
        # store labels
        self._store_class_labels(dataset)
        # start training
        self._clf.build_classifier(dataset)

    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def get_params(self, deep=False):
        return {'clf' : self._clf}
