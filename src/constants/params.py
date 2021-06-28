# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
from constants.enums import BaseModels

# PARAMETERS
# ----------------------------------------------------------------------------------------------------

# Base model parameter's spaces
BaseParams = {
    BaseModels.TREE:[
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
    ],
    BaseModels.KNN:[
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
    ],
    BaseModels.SVM:[
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
    ],
    BaseModels.MLP:[
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
    ],
}
