from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import time
import pandas as pd

class Model():
    """SVM Model"""
    def __init__(self, data, debug=False):
        self.data = data
        self.debug = debug
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.training_data = None
        self.testing_data = None
        self.svm = None
        self.predictions = None

    def splitData(self, p=False):
        if self.debug: print('In splitData')
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data['Text'], self.data['Labels'], random_state=1)

        if p:
            print(' ')
            print('Number of rows in the total set: {}'.format(self.data.shape[0]))
            print('Number of rows in the training set: {}'.format(self.X_train.shape[0]))
            print('Number of rows in the test set: {}'.format(self.X_test.shape[0]))
            print(' ')
            print(self.data.head())

    def cvDataset(self):
        if self.debug: print('In cvDataset')
        
        count_vector = CountVectorizer(stop_words='english')
        self.training_data = count_vector.fit_transform(self.X_train)
        self.testing_data = count_vector.transform(self.X_test)
        
        if self.debug:
            data = pd.DataFrame(self.training_data.toarray(), columns=count_vector.get_feature_names())
            print(data.head())

    def tfidfDataset(self):
        if self.debug: print('In tfidfDataset')
            
        #tfidf_vector = TfidfVectorizer(min_df = 5, max_df = 0.8, sublinear_tf = True, use_idf = True)
        tfidf_vector = TfidfVectorizer(stop_words='english')
        self.training_data = tfidf_vector.fit_transform(self.X_train)
        self.testing_data = tfidf_vector.transform(self.X_test)

        if self.debug:
            data = pd.DataFrame(self.training_data.toarray(), columns=tfidf_vector.get_feature_names())
            print(data.head())

    def fitSVM(self):
        if self.debug: print('In fitSVM')
            
        self.svn = svm.SVC(kernel='linear')
        t0 = time.time()
        self.svn.fit(self.training_data, self.y_train)
        t1 = time.time()
        
        if self.debug:
            time_linear_train = t1-t0
            print("Training time: %fs" % (time_linear_train))

    def predict(self):
        if self.debug: print('In predict')
        
        self.predictions = self.svn.predict(self.testing_data)

    def printScores(self, posLabel=4):
        if self.debug: print('In printScores')
        
        print(' ')
        print('Accuracy score: ', format(accuracy_score(self.y_test, self.predictions), '.4f'))
        print('Precision score: ', format(precision_score(self.y_test, self.predictions, pos_label=posLabel), '.4f'))
        print('Recall score: ', format(recall_score(self.y_test, self.predictions, pos_label=posLabel), '.4f'))
        print('F1 score: ', format(f1_score(self.y_test, self.predictions, pos_label=posLabel), '.4f'))