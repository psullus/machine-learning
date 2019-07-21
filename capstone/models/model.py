from sklearn.model_selection import train_test_split
from sklearn import svm
#from sklearn.feature_extraction.text import CountVectorizer
#from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Model():
    """SVM Model"""
    def __init__(self, data):
        self.data = data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.count_vector = None
        self.training_data = None
        self.testing_data = None
        self.svm = None
        self.predictions = None

    def splitData(self, p=False):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data['Text'], self.data['Labels'], random_state=1)

        if p:
            print(' ')
            print('Number of rows in the total set: {}'.format(self.data.shape[0]))
            print('Number of rows in the training set: {}'.format(self.X_train.shape[0]))
            print('Number of rows in the test set: {}'.format(self.X_test.shape[0]))
            print(' ')
            print(self.data.head())

    def fitSVM(self):
        self.svn = svm.SVC(kernel='linear')
        self.svn.fit(train_vectors, trainData['Label'])
        self.naive_bayes.fit(self.training_data, self.y_train)