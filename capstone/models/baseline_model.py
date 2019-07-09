from sklearn.model_selection import train_test_split

class BaselineModel():
    """Naive Bayes Model"""
    def __init__(self, data):
        self.data = data
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def splitData(self, p=False):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data['text'], self.data['labels'], random_state=1)

        if p:
            print('Number of rows in the total set: {}'.format(self.data.shape[0]))
            print('Number of rows in the training set: {}'.format(self.X_train.shape[0]))
            print('Number of rows in the test set: {}'.format(self.X_test.shape[0]))
            print(self.data.head())