# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
from constants.enums import ClassificationModels, RegressionModels

# PARAMETERS
# ----------------------------------------------------------------------------------------------------

# Classification model parameter's spaces
ClassificationParams = {
    ClassificationModels.TREE:[
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
    ClassificationModels.GNB:[
        {
            'var_smoothing': 0.00000001,
        },
        {
            'var_smoothing': 0.00001,
        },
        {
            'var_smoothing': 0.001,
        },
        {
            'var_smoothing': 0.1,
        },
    ],
    ClassificationModels.KNN:[
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
    ClassificationModels.SVM:[
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
    ClassificationModels.LOR:[
        {
            'solver': 'lbfgs',
            'penalty': 'l2',
            'max_iter': 200,
        },
        {
            'solver': 'newton-cg',
            'penalty': 'l2',
            'max_iter': 200,
        },
        {
            'solver': 'sag',
            'penalty': 'l2',
            'max_iter': 200,
        },
        {
            'solver': 'saga',
            'penalty': 'l1',
            'max_iter': 200,
        },
        {
            'solver': 'saga',
            'penalty': 'l2',
            'max_iter': 200,
        },
        {
            'solver': 'liblinear',
            'penalty': 'l1',
            'max_iter': 200,
        },
        {
            'solver': 'liblinear',
            'penalty': 'l2',
            'max_iter': 200,
        },
    ],
    ClassificationModels.MLP:[
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

# Regression model parameter's spaces
RegressionParams = {
    RegressionModels.KNN:[
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
    RegressionModels.SVM:[
        {
            'kernel': 'rbf',
            'gamma': 'auto',
            'tol': 1e-3,
            'C': 1
        },
        {
            'kernel': 'rbf',
            'gamma': 'scale',
            'tol': 1e-3,
            'C': 1
        },
        {
            'kernel': 'rbf',
            'gamma': 'auto',
            'tol': 1e-3,
            'C': 10
        },
        {
            'kernel': 'rbf',
            'gamma': 'scale',
            'tol': 1e-3,
            'C': 10
        },
        {
            'kernel': 'rbf',
            'gamma': 'auto',
            'tol': 1e-3,
            'C': 0.1
        },
        {
            'kernel': 'rbf',
            'gamma': 'scale',
            'tol': 1e-3,
            'C': 0.1
        },
        {
            'kernel': 'rbf',
            'gamma': 'auto',
            'tol': 1e-6,
            'C': 1
        },
        {
            'kernel': 'rbf',
            'gamma': 'scale',
            'tol': 1e-6,
            'C': 1
        },
        {
            'kernel': 'rbf',
            'gamma': 'auto',
            'tol': 1e-6,
            'C': 10
        },
        {
            'kernel': 'rbf',
            'gamma': 'scale',
            'tol': 1e-6,
            'C': 10
        },
        {
            'kernel': 'rbf',
            'gamma': 'auto',
            'tol': 1e-6,
            'C': 0.1
        },
        {
            'kernel': 'rbf',
            'gamma': 'scale',
            'tol': 1e-6,
            'C': 0.1
        },

        {
            'kernel': 'linear',
            'tol': 1e-3,
            'C': 1
        },
        {
            'kernel': 'linear',
            'tol': 1e-3,
            'C': 10
        },
        {
            'kernel': 'linear',
            'tol': 1e-3,
            'C': 0.1
        },
        {
            'kernel': 'linear',
            'tol': 1e-6,
            'C': 1
        },
        {
            'kernel': 'linear',
            'tol': 1e-6,
            'C': 10
        },
        {
            'kernel': 'linear',
            'tol': 1e-6,
            'C': 0.1
        },
    ],
    RegressionModels.LIR:[
        {}
    ],
    RegressionModels.MLP:[
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
