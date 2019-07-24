from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class BaselineModel():
    """Naive Bayes Model"""
    def __init__(self, data, debug=False):
        self.data = data
        self.debug = debug
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.count_vector = None
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
        
        self.count_vector = CountVectorizer()
        self.training_data = self.count_vector.fit_transform(self.X_train)
        self.testing_data = self.count_vector.transform(self.X_test)

    def fitNaiveBayes(self):
        if self.debug: print('In fitNaiveBayes')
        
        self.naive_bayes = MultinomialNB()
        self.naive_bayes.fit(self.training_data, self.y_train)

    def predict(self):
        if self.debug: print('In predict')
        
        self.predictions = self.naive_bayes.predict(self.testing_data)

    def printScores(self, posLabel=4):
        if self.debug: print('In printScores')
        
        print(' ')
        print('Accuracy score: ', format(accuracy_score(self.y_test, self.predictions)))
        print('Precision score: ', format(precision_score(self.y_test, self.predictions, pos_label=posLabel)))
        print('Recall score: ', format(recall_score(self.y_test, self.predictions, pos_label=posLabel)))
        print('F1 score: ', format(f1_score(self.y_test, self.predictions, pos_label=posLabel)))
