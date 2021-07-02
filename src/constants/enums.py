# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
from enum import Enum

# CONSTANTS
# ----------------------------------------------------------------------------------------------------

# Model classes
class Models(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2
    NEURAL = 3

# Classification model types
class ClassificationModels(Enum):
    TREE = '(DBT) Decision Binary Tree'
    GNB = '(GNB) Gaussian Naive Bayes'
    KNN = '(KNN) K-Nearest Neighbors'
    SVM = '(SVM) Support Vector Machines'
    LOR = '(LoR) Logistic Regression'
    MLP = '(MLP) Multi-Layered Perceptron'

# Regression model types
class RegressionModels(Enum):
    KNN = '(KNN) K-Nearest Neighbors'
    SVM = '(SVM) Support Vector Machines'
    LIR = '(LiR) Linear Regression'
    MLP = '(MLP) Multi-Layered Perceptron'

# Neural model types
class NeuralModels(Enum):
    MLP = '(MLP) Multi-Layered Perceptron'
