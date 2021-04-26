# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
import warnings
warnings.simplefilter("ignore", UserWarning)
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# DEPENDENCIES (Local)
# ----------------------------------------------------------------------------------------------------
import utils.const as const

# MAIN CLASS
# ----------------------------------------------------------------------------------------------------
class BaseModel():

    # Aux Function - Save model in pickle file
    def save(self, path):
        pickle.dump(self.predictor, open(path, 'wb'))

    # Aux Function - Load model from pickle file
    def load(self, path):
        self.classifier = pickle.load(open(path, 'rb'))

    # Constructor
    def __init__(self, model='mlp_classifier', params={}):

        # Generic configuration
        self.model = params['model'] if 'model' in params else model
        self.params = params['params'] if 'params' in params else None

        # Interfaces
        self.predictor = None

    # Train predictor for dataset X and tags Y
    def train(self, X, Y):

        # If params exist, use them to train predictor
        if self.params is not None:
            if self.model == 'svm':
                self.predictor = SVC(**self.params)
            elif self.model == 'tree':
                self.predictor = DecisionTreeClassifier(**self.params)
            elif self.model == 'knn':
                self.predictor = KNeighborsClassifier(**self.params)
            elif self.model == 'mlp_classifier':
                self.predictor = MLPClassifier(**self.params)

        # If params don't exist, use generic params to train predictor
        else:
            if self.model == 'svm':
                self.predictor = SVC(kernel='linear', C=10)
            elif self.model == 'tree':
                self.predictor = DecisionTreeClassifier(criterion='entropy', max_depth=8)
            elif self.model == 'knn':
                self.predictor = KNeighborsClassifier(3)
            elif self.model == 'mlp_classifier':
                self.predictor = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', alpha=0.05, solver='adam',  max_iter=2000)

        # Train using dataset X and tags Y
        self.predictor.fit(X, Y)

    # Predict tags for X
    def predict(self, X):
        return self.predictor.predict(X)

    # Evaluate predictor for testset X and tags Y 
    def evaluate(self, X, Y):

        # Get prediction to compare
        prediction = self.predict(X)

        # Get metrics
        accuracy = accuracy_score(Y, prediction)
        results = classification_report(Y, prediction, output_dict=True)
        report_string = classification_report(Y, prediction)
        matrix = confusion_matrix(Y, prediction)

        report = {
            'f1_score': results['macro avg']['f1-score'],
            'precision': results['macro avg']['precision'],
            'recall': results['macro avg']['recall'],
        }
        return accuracy, report, report_string, matrix
