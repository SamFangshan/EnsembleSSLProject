from sklearn.base import MultiOutputMixin, BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score
from abc import ABCMeta
import numpy as np
import pandas as pd
from weka.classifiers import Classifier
from weka.core.dataset import create_instances_from_matrices
from weka.filters import Filter

class ScikitLearnWekaWrapper(ClassifierMixin, BaseEstimator, metaclass=ABCMeta):
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
        elif isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray):
            raise Exception("Incompatible data type: {}".format(type(X)))
        dataset = create_instances_from_matrices(X, name="generated from matrices")
        dataset.insert_attribute(self._class_att, dataset.num_attributes)
        dataset.class_is_last()
        return dataset
        
    def _get_training_dataset(self, X, y):
        # convert to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray):
            raise Exception("Incompatible data type: {}".format(type(X)))
        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, list):
            y = np.array(y)
        elif not isinstance(y, np.ndarray):
            raise Exception("Incompatible data type: {}".format(type(y)))
            
        if y.dtype == "O":
            for i in range(0, len(y)):
                try:
                    y[i] = y[i].encode()
                except:
                    pass
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
            preds.append(self.classes_[int(self._clf.get_clf().classify_instance(inst))])
        preds = np.array(preds)
        if preds.dtype == np.float64:
            preds = preds.astype(np.int64)
        return preds
        
    def predict_proba(self, X):
        dataset = self._get_prediction_dataset(X)
        dists = []
        for index, inst in enumerate(dataset):
            dists.append(self._clf.get_clf().distribution_for_instance(inst))
        dists = np.array(dists)
        return dists
        
    def fit(self, X, y):
        # transform dataset
        dataset = self._get_training_dataset(X, y)
        # store labels
        self._store_class_labels(dataset)
        # start training
        self._clf.get_clf().build_classifier(dataset)
        
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)
        
    def get_params(self, deep=False):
        return {'clf' : self._clf}

class WekaWrapper:
    def __init__(self, clf, classname):
        self._clf = clf
        self._classname = classname
        
    def get_clf(self):
        return self._clf
        
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        clf = Classifier(classname=self._classname)
        result.__dict__["_clf"] = clf
        return result
        
    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.__dict__.update(self.__dict__)
        clf = Classifier(classname=self._classname)
        result.__dict__["_clf"] = clf
        return result
    
