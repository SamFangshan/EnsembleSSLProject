import sklearn
import pandas as pd
import copy

class TriTrainClassifier:
    clfs = []
    def __init__(self, clf1, clf2=None, clf3=None):

        for n in range(3):
            clfs.append(copy.deepcopy(clf))




    def fit(self, L, U):
   
      size = len(L)
      sample_size = size//3
      ####-------bootstrapping
      LS = []
      for n in range(3):
          #https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
          #L1 = sklearn.utils.resample(L,n_samples=sample_size,replace=True, random_state = n)
          L1 = sklearn.utils.resample(L,n_samples=sample_size,replace=True)
          LS.append(L1)

      x_train = []
      y_train = []
      #####-----seperate into x and y
      for n in range(3):
          x_train_1 = LS[n].iloc [:, [n for n in range(len(LS[n].columns)-1)]] 
          x_train.append(x_train_1)

          y_train_1 = LS[n].iloc [:, [len(LS[0].columns)-1]].to_numpy()
          y_train.append(y_train_1)
      #######-----------transductive learning
      for n in range(3):
          clfs[n].fit(x_train[n],y_train[n])
      #how to get accuracy?? Compare with Unlabled answer   


      #####----------inductive learning

      #get unlabled data
      x_unlabled = U.iloc[:, [n for n in range(len(U.columns)-2)]] 
      x_unlabled_copy = copy.deepcopy(x_unlabled)

      print(len(x_unlabled_copy.index))


      u_size_previous = 0
      u_size = len(x_unlabled.index)
      U_size = len(x_unlabled.index)

      iteration = 0

      while True:
        #predict
        result=[]
        for n in range(3):
          result.append(clfs[n].predict(x_unlabled_copy))
        
        #collate answer
        Unlabled = copy.deepcopy(x_unlabled_copy)
        for n in range(3):
          Unlabled["result{}".format(n)]=result[n]
        
        #majority vote
        Teaching = []
        answerName = L.columns[-1]
        #print(answerName)
        for n in range(3):
          #find what the other two classifer agree on
          temp = Unlabled.loc[Unlabled["result{}".format((n+1)%3)]==Unlabled["result{}".format((n+2)%3)]]
          #format it into training dataset
          temp = temp.drop(["result{}".format((n+1)%3),"result{}".format(n)], axis=1)
          temp = temp.rename(columns={"result{}".format((n+2)%3):answerName})
          Teaching.append(temp)

        
        for n in range(3):
          x_unlabled_copy = x_unlabled_copy.drop([m for m in Teaching[n].index],axis=0, errors = "ignore")

        ###train again
        for n in range(3):
          LS[n]=LS[n].append(Teaching[n])
        
        x_train = []
        y_train = []
        for n in range(3):
          x_train_1 = LS[n].iloc [:, [n for n in range(len(LS[n].columns)-1)]] 
          x_train.append(x_train_1)

          y_train_1 = LS[n].iloc [:, [len(LS[0].columns)-1]].to_numpy()
          y_train.append(y_train_1)

        
        for n in range(3):
          clfs[n].fit(x_train[n],y_train[n])
        
        iteration += 1
        print(iteration)

        u_size_previous = U_size
        u_size = len(x_unlabled_copy.index)
        print(len(x_unlabled_copy.index))
        if len(x_unlabled_copy.index) < 0.01*U_size or iteration >15:
          break
      
      classifier_result=[] #result of each classifer
      final_result = [] #
      for n in range(3):
        classifier_result.append(clfs[n].predict(x_unlabled))
        
      #collate answer
      Unlabled_result = copy.deepcopy(x_unlabled)
      #for n in range(3):
      #  Unlabled_result["result{}".format(n)]=result[n]
      answer = 0
      for n in range(len(classifier_result[0])):
        answer = 0
        #Unlabled_result.iloc[n.]
        if classifier_result[0][n]==classifier_result[1][n] or classifier_result[0][n]==classifier_result[2][n]:
          answer = classifier_result[0][n]
        elif classifier_result[1][n]==classifier_result[2][n] :
          answer = classifier_result[0][n]
        else:
          #print("cannot agree")
          answer = classifier_result[0][n]
        final_result.append(answer)
          
      #Unlabled_result[answerName]=final_result
      #predicted = Unlabled_result[answerName]
      predicted = final_result
      reference = U.iloc [:, [len(U.columns)-1]].to_numpy()
      transductive_accuracy = sklearn.metrics.accuracy_score(reference,predicted)
      ##use sklearn.metrics.accuracy_score to produce transductive accuracy?
      return self, transductive_accuracy

    def predict(self, X):
        ##predict base on majority vote?(what if no one agrees?)
      result=[] #result of each classifer
      y_pred = [] #
      for n in range(3):
        result.append(clfs[n].predict(X))
        
      #collate answer
      Unlabled_result = copy.deepcopy(X)
      #for n in range(3):
      #  Unlabled_result["result{}".format(n)]=result[n]
      answer = 0
      disagree_count = 0
      for n in range(len(result[0])):
        answer = 0
        #Unlabled_result.iloc[n.]
        if result[0][n]==result[1][n] or result[0][n]==result[2][n]:
          answer = result[0][n]
        elif classifier_result[1][n]==classifier_result[2][n] :
          answer = result[1][n]
        else:
          #print("cannot agree")
          disagree_count += 1 
          #print(disagree_count," cannot agree on")
          answer = result[0][n]
        y_pred.append(answer)
      print(disagree_count," really cannot agree on")
      return y_pred

    #def predict_proba(self, X):

    #    return y_pred_proba

    def score(self, X, y):
        predicted = self.predict(X)
        reference = y
        score = sklearn.metrics.accuracy_score(reference,predicted)

        return score