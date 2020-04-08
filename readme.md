# CE4041/CZ4041 Research
### This research is to be conducted on Google Drive through Colaboratory
### Note: Save this folder to your Drive first before you proceed!
## 1. Move to project directory
Run this piece of code every time to move to the project directory
```python
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/My Drive/4041 Code+Data')
```
## 2. Data Preprocessing
The original data sets were retrieved from https://sci2s.ugr.es/keel/semisupervised.php.  All retrived data sets are in `.dat` format. In order to make the data sets Python friendly, they are all converted to `.csv` format. Afterwards, labeled training data and unlabeled training data are separated into two files. Data sets with categorical features are all one-hot encoded, and all numerical features are minmax normalized.
## 3. File access template
The following template can be used when you need to load a specific CSV file

```python
# The percentage of labeled records
percentage = 10 # can be 10, 20, 30 or 40
# Dataset name
dataset = "abalone"
# Partition ID, used for cross validation
partition = 3 # can be 1, 2, 3, 4, 5, 6, 7, 8, 9 or 10
# Type of data
type_of_data = "tra-l" # can be "tra-l", "tra-u" or "tst"
# "tra-l" is labeled training data, "tra-u" is unlabeled training data, "tst" is test data

csv_filename = 'ssl_{}/{}-ssl{}/{}-ssl{}-10-{}{}.csv'.format(percentage, dataset,
                                                             percentage, dataset,
                                                             percentage, partition, type_of_data)
```
Labeled training data example:
```csv
attribute1,attribute2,attribute3,attribute4,attribute5,label
```
Unlabeled training data example (already provided with answer to check):
```csv
attribute1,attribute2,attribute3,attribute4,attribute5,label,label_answer
```
Test data example:
```csv
attribute1,attribute2,attribute3,attribute4,attribute5,label
```

## 4. WEKA
Run the following two chunks of code when you need to use the WEKA package, to install dependencies
#### What is WEKA? Refer to https://www.cs.waikato.ac.nz/ml/weka/ and https://weka.sourceforge.io/doc.stable-3-8/weka/classifiers/Classifier.html
```python
import os
import sys
sys.path
sys.path.append("/usr/lib/jvm/java-11-openjdk-amd64/bin/")
os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-11-openjdk-amd64/"
!apt-get install build-essential python3-dev
!apt-get install python3-pil python3-pygraphviz
!apt install openjdk-11-jdk
!pip3 install javabridge --no-cache-dir
!pip3 install python-weka-wrapper3 --no-cache-dir
```
```python
import weka.core.jvm as jvm
jvm.start()
```
Also remember to import the WEKA Scikit-Learn wrapper
```python
from scikit_learn_weka.wrapper import ScikitLearnWekaWrapper
```
Initialisation example
```python
from weka.classifiers import Classifier
cls = Classifier(classname="weka.classifiers.functions.SMO", options=["-N", "0"]) # WEKA classifier
cls = ScikitLearnWekaWrapper(cls) # wrap WEKA classifier
# use cls as if it were a Scikit-Learn classifier
```
For a full example, please refer to weka-example.ipynb
## 5. Functions to be implemented
```python
def generate_base_classifier(clf_name):
    return clf

# ensemble of three classifiers
def get_ensemble(ensemble_name):
    return clf_ensemble # A VotingClassifier object 
# refer to https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier

def train_and_validate(clf, L, U, X_test, y_test, mode="self"):
    # invoke self_training/co_training/tri_training
    print(transductive accuracy)
    print(inductive accuracy)
    return transductive accuracy, inductive accuracy

def cross_validation(clf, dataset_name, percentage, mode="self"):
    # invoke train_and_validate
    print(average transductive accuracy)
    print(average inductive accuracy)
    return average transductive accuracy, average inductive accuracy

def avg_by_percentage(clf, percentage, mode="self"):
    return tra_avg, ind_avg
```
```python
class Classifier:
    def __init__(self, clf1, clf2=None, clf3=None):
        pass

    def fit(self, L, U):
    """
    @param
    L: labeled data
    U: unlabeled data

    Entries in U will be gradually added into L, until U becomes empty
    or some stopping condition is met.
    """
        return self, transductive_accuracy

    def predict(self, X):
        return y_pred

    def predict_proba(self, X):
        # (optional)
        return y_pred_proba

    def score(self, X, y):
        # (optional)
        return score
```
## 6. Research Paper Components
+ 1. Introduction
+ 2. Literature Review
+ 3. Review on Semi-supervised Self-labeled Classification (??)
+ 4. Methodology
+ 5. Experimental Results
+ 6. Comparison Analysis
+ 6. Conclusion
