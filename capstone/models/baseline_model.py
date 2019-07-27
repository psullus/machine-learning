from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import time
import pandas as pd

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

    def fitNaiveBayes(self):
        if self.debug: print('In fitNaiveBayes')
        
        self.naive_bayes = MultinomialNB()
        t0 = time.time()
        self.naive_bayes.fit(self.training_data, self.y_train)
        t1 = time.time()
        
        if self.debug:
            time_linear_train = t1-t0
            print("Training time: %fs" % (time_linear_train))

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

    def printConfusionMatrix():
        mat = confusion_matrix(tself.testing_data, self.y_test)
        sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=self.training_data.target_names, yticklabels=self.training_data.target_names)
        plt.xlabel('true label')
        plt.ylabel('predicted label');