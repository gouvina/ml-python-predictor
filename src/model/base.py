# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
import warnings
warnings.simplefilter("ignore", UserWarning)
import pickle
import numpy as np
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# DEPENDENCIES (Local)
# ----------------------------------------------------------------------------------------------------
from constants.enums import BaseModels
from constants.params import BaseParams

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
    def __init__(self, model=BaseModels.MLP, params=BaseParams[BaseModels.MLP][0]):

        # Generic configuration
        self.model = model
        self.params = params

        # Interfaces
        self.predictor = None

    # Train predictor for dataset X and tags Y
    def train(self, X, Y):

        # Create model's instance based on params
        if self.model == BaseModels.TREE:
            self.predictor = DecisionTreeClassifier(**self.params)
        elif self.model == BaseModels.KNN:
            self.predictor = KNeighborsClassifier(**self.params)
        elif self.model == BaseModels.SVM:
            self.predictor = SVC(**self.params)
        elif self.model == BaseModels.MLP:
            self.predictor = MLPClassifier(**self.params)

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
        matrix = confusion_matrix(Y, prediction)

        # Fill report
        report = {
            'accuracy': accuracy,
            'f1_score': results['macro avg']['f1-score'],
            'precision': results['macro avg']['precision'],
            'recall': results['macro avg']['recall'],
        }
        return report, matrix
