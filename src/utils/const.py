# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
from enum import Enum

# CONSTANTS
# ----------------------------------------------------------------------------------------------------

# Folder paths
MODEL_FOLDER = 'models/'
DATA_FOLDER = 'data/'

# File paths
IRIS_DATASET = 'iris.csv'
IRIS_PREDICTOR = 'iris.pkl'

# Model classes
class Models(Enum):
    BASE = 1
    NEURAL = 2

# Base model types
class BaseModels(Enum):
    TREE = 1
    NB = 2    
    KNN = 3
    SVM = 4
    MLP = 5

# Base model parameter's spaces
class BaseParams():
    TREE = [
        {
            'criterion': 'gini',
            'max_depth': 4
        },
        {
            'criterion': 'gini',
            'max_depth': 8
        },
        {
            'criterion': 'gini',
            'max_depth': 12
        },
        {
            'criterion': 'entropy',
            'max_depth': 4
        },
        {
            'criterion': 'entropy',
            'max_depth': 8
        },
        {
            'criterion': 'entropy',
            'max_depth': 12
        },
    ]
    KNN = [
        {
            'n_neighbors': 3,
            'weights': 'uniform',
            'metric': 'euclidean'
        },
        {
            'n_neighbors': 5,
            'weights': 'uniform',
            'metric': 'euclidean'
        },
        {
            'n_neighbors': 3,
            'weights': 'distance',
            'metric': 'euclidean'
        },
        {
            'n_neighbors': 5,
            'weights': 'distance',
            'metric': 'euclidean'
        },
        {
            'n_neighbors': 3,
            'weights': 'uniform',
            'metric': 'manhattan'
        },
        {
            'n_neighbors': 5,
            'weights': 'uniform',
            'metric': 'manhattan'
        },
        {
            'n_neighbors': 3,
            'weights': 'distance',
            'metric': 'manhattan'
        },
        {
            'n_neighbors': 5,
            'weights': 'distance',
            'metric': 'manhattan'
        },
    ]
    SVM = [
        {
            'kernel': 'rbf',
            'gamma': 'auto',
            'C': 1
        },
        {
            'kernel': 'rbf',
            'gamma': 'auto',
            'C': 10
        },
        {
            'kernel': 'linear',
            'C': 1
        },
        {
            'kernel': 'linear',
            'C': 10
        }
    ]
    MLP = [
        {
            'hidden_layer_sizes': (100,),
            'max_iter': 2000,
            'activation': 'relu',
            'solver': 'sgd',
            'alpha': 0.0001,
            'learning_rate': 'constant'
        },
        {
            'hidden_layer_sizes': (100,100),
            'max_iter': 2000,
            'activation': 'relu',
            'solver': 'sgd',
            'alpha': 0.0001,
            'learning_rate': 'constant'
        },
        {
            'hidden_layer_sizes': (100,100,100),
            'max_iter': 2000,
            'activation': 'relu',
            'solver': 'sgd',
            'alpha': 0.0001,
            'learning_rate': 'constant'
        },
        {
            'hidden_layer_sizes': (100,),
            'max_iter': 2000,
            'activation': 'logistic',
            'solver': 'sgd',
            'alpha': 0.0001,
            'learning_rate': 'constant'
        },
        {
            'hidden_layer_sizes': (100,100),
            'max_iter': 2000,
            'activation': 'logistic',
            'solver': 'sgd',
            'alpha': 0.0001,
            'learning_rate': 'constant'
        },
        {
            'hidden_layer_sizes': (100,100,100),
            'max_iter': 2000,
            'activation': 'logistic',
            'solver': 'sgd',
            'alpha': 0.0001,
            'learning_rate': 'constant'
        },
        {
            'hidden_layer_sizes': (100,),
            'max_iter': 2000,
            'activation': 'logistic',
            'solver': 'sgd',
            'alpha': 0.05,
            'learning_rate': 'constant'
        },
        {
            'hidden_layer_sizes': (100,100),
            'max_iter': 2000,
            'activation': 'logistic',
            'solver': 'sgd',
            'alpha': 0.05,
            'learning_rate': 'constant'
        },
        {
            'hidden_layer_sizes': (100,100,100),
            'max_iter': 2000,
            'activation': 'logistic',
            'solver': 'sgd',
            'alpha': 0.05,
            'learning_rate': 'constant'
        },
        {
            'hidden_layer_sizes': (100,),
            'max_iter': 2000,
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.05,
            'learning_rate': 'constant'
        },
        {
            'hidden_layer_sizes': (100,100),
            'max_iter': 2000,
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.05,
            'learning_rate': 'constant'
        },
        {
            'hidden_layer_sizes': (100,100,100),
            'max_iter': 2000,
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.05,
            'learning_rate': 'constant'
        },
        {
            'hidden_layer_sizes': (100,),
            'max_iter': 2000,
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.05,
            'learning_rate': 'adaptive'
        },
        {
            'hidden_layer_sizes': (100,100),
            'max_iter': 2000,
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.05,
            'learning_rate': 'adaptive'
        },
        {
            'hidden_layer_sizes': (100,100,100),
            'max_iter': 2000,
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.05,
            'learning_rate': 'adaptive'
        },
        {
            'hidden_layer_sizes': (100,),
            'max_iter': 2000,
            'activation': 'logistic',
            'solver': 'adam',
            'alpha': 0.05,
            'learning_rate': 'adaptive'
        },
        {
            'hidden_layer_sizes': (100,100),
            'max_iter': 2000,
            'activation': 'logistic',
            'solver': 'adam',
            'alpha': 0.05,
            'learning_rate': 'adaptive'
        },
        {
            'hidden_layer_sizes': (100,100,100),
            'max_iter': 2000,
            'activation': 'logistic',
            'solver': 'adam',
            'alpha': 0.05,
            'learning_rate': 'adaptive'
        },
    ]

# Neural model types
class NeuralModels(Enum):
    MLP = 1