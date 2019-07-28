from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import time
import pandas as pd
from models.utils import Utils
import matplotlib.pyplot as plt
import pickle

class BaselineModel():
    """Naive Bayes Model"""
    def __init__(self, data, debug=False):
        self.data = data
        self.debug = debug
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.training_data = None
        self.testing_data = None
        self.naive_bayes = None
        self.predictions = None
        self.vector = None

    def splitData(self):
        if self.debug: print('In splitData')
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data['Text'], self.data['Labels'], random_state=1)

        if self.debug:
            print(' ')
            print('Number of rows in the total set: {}'.format(self.data.shape[0]))
            print('Number of rows in the training set: {}'.format(self.X_train.shape[0]))
            print('Number of rows in the test set: {}'.format(self.X_test.shape[0]))
            print(' ')
            print(self.data.head())
            print(' ')
            
    def cvDataset(self):
        if self.debug: print('In cvDataset')
        
        self.vector = CountVectorizer(stop_words='english')
        self.training_data = self.vector.fit_transform(self.X_train)
        self.testing_data = self.vector.transform(self.X_test)

        if self.debug:
            data = pd.DataFrame(self.training_data.toarray(), columns=self.vector.get_feature_names())
            print(data.head())
            print(' ')

    def fit(self):
        if self.debug: print('In fitNaiveBayes')
        
        self.naive_bayes = MultinomialNB()
        t0 = time.time()
        self.naive_bayes.fit(self.training_data, self.y_train)
        t1 = time.time()
        
        if self.debug:
            time_linear_train = t1-t0
            print("Training time: %fs" % (time_linear_train))
            print(' ')
            
    def predict(self):
        if self.debug: print('In predict')
        
        self.predictions = self.naive_bayes.predict(self.testing_data)

    def printScores(self, posLabel=4):
        if self.debug: print('In printScores')
        
        print(' ')
        print('Accuracy score: ', format(accuracy_score(self.y_test, self.predictions), '.4f'))
        print('Precision score: ', format(precision_score(self.y_test, self.predictions, pos_label=posLabel), '.4f'))
        print('Recall score: ', format(recall_score(self.y_test, self.predictions, pos_label=posLabel), '.4f'))
        print('F1 score: ', format(f1_score(self.y_test, self.predictions, pos_label=posLabel), '.4f'))
        print(' ')
        
    def plotConfusionMatrix(self):
        if self.debug: print('In plotConfusionMatrix')
        
        utils = Utils(debug=True)
        
        # Compute confusion matrix
        cnf_matrix = confusion_matrix(self.y_test, self.predictions)
        plt.figure()
        utils.plot_confusion_matrix(cnf_matrix, classes=['Pos Sendiment','Neg Sendiment'], title='Confusion matrix, without normalization')
        plt.figure()
        utils.plot_confusion_matrix(cnf_matrix, classes=['Pos Sendiment','Neg Sendiment'], normalize=True, title='Confusion matrix, with normalization')
        
        if self.debug: utils.printTP_FP_TN_FN(cnf_matrix)

    def testTwit(self, twit):
        review_vector = self.vector.transform([twit]) # vectorizing
        predict = self.naive_bayes.predict(review_vector)
        
        if predict == 4:
            print("This twit is positive (4)")
        else:
            print("This twit is negative (0)")
          
        return predict

    def saveModelAndVector(self):
        # pickling the vectorizer
        pickle.dump(self.vector, open('baseline_vectorizer.sav', 'wb'))
        # pickling the model
        pickle.dump(self.naive_bayes, open('baseline_classifier.sav', 'wb'))