
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

models = dict()
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()


class TrainingData:
    def __init__(self, x=[], y=[]):
        '''Provides training data to the classifier
        x: array-like traning data
            shape [n_samples, n_features]
            e.g. [ [a, b, c] [a, b, c] ]
        y: array-like target values for x
            shape: [n_samples]
            e.g. [ 'good', 'bad' ]'''
        self.x = x
        self.y = y


    def add_x(self, x):
        '''Adds training X data'''
        self.x.extend(x)


    def add_y(self, y):
        '''Adds training X data'''
        self.y.extend(y)


    def replace_x(self, x):
        '''replaces training X data'''
        self.x = x


    def replace_y(self, y):
        '''replaces training X data'''
        self.y = y


    def serialize(self):
        return self.__dict__


class Classifier:
    def __init__(self, modelname=None):
        '''Classifies data with supervised machine learning
        Determines whether or not child routes will be 
        "good" members of the population
        
        model: must be a callable classifier model. 
        If None, defaults to KNN'''
        if modelname is None:
            self.modelname = 'KNN'
            self.model = KNeighborsClassifier()
        else:
            self.modelname = modelname
            self.model = models[modelname]
        self.training_data = TrainingData()


    def re_train(self):
        '''Trains the classifier with the training data previously supplied'''
        self.model.fit(self.training_data.x, self.training_data.y)


    def predict(self, x):
        '''predicts the classification of the input x data'''
        return self.model.predict(x)

    def serialize(self):
        return self.modelname


def spot_check():
    '''Spot checks classification methods'''

    # given a population
    pop = None

    # 