# DEPENDENCIES (Local)
# ----------------------------------------------------------------------------------------------------
from constants.enums import Models, ClassificationModels, RegressionModels, NeuralModels
from constants.params import ClassificationParams, RegressionParams

# AUX METHODS
# ----------------------------------------------------------------------------------------------------

# Parse model class into enum with types:
# -> Base = Models based on common ML methods
# -> Neural = Models based on neural networks
def parse_model_class(model_class):
    
    model = None
    model_type = None

    if model_class == 'classification':
        model = Models.CLASSIFICATION
        model_type = ClassificationModels.MLP
    if model_class == 'regression':
        model = Models.REGRESSION  
        model_type = RegressionModels.MLP
    if model_class == 'neural':
        model = Models.NEURAL
        model_type = NeuralModels.MLP

    return model, model_type

# Parse model type depending of model class, into enum with types:
# Base:
# -> Tree = Binary decision tree
# -> KNN = K Nearest Neighbors
# -> SVM = Support Vector Machines
# -> MLP = Multi Layered Perceptron
def parse_model_type(model_class, model_type):

    model = None
    model_params = None

    if model_class == Models.CLASSIFICATION:
        if model_type == 'tree':
            model = ClassificationModels.TREE
        if model_type == 'gnb':
            model = ClassificationModels.GNB
        if model_type == 'knn':
            model = ClassificationModels.KNN
        if model_type == 'svm':
            model = ClassificationModels.SVM
        if model_type == 'lor':
            model = ClassificationModels.LOR
        if model_type == 'mlp':
            model = ClassificationModels.MLP
        model_params = ClassificationParams[model][0]
    elif model_class == Models.REGRESSION:
        if model_type == 'knn':
            model = RegressionModels.KNN
        if model_type == 'svm':
            model = RegressionModels.SVM
        if model_type == 'lir':
            model = RegressionModels.LIR
        if model_type == 'mlp':
            model = RegressionModels.MLP
        model_params = RegressionParams[model][0]
    else:
        if model_type == 'mlp':
            model = NeuralModels.MLP

    return model, model_params

# Retrieve model training params depending on class and type
def parse_model_params(model_class, model_type, model_params):

    params = None

    if model_class == Models.CLASSIFICATION:
        params = ClassificationParams[model_type][model_params]
    elif model_class == Models.REGRESSION:
        params = RegressionParams[model_type][model_params]
        
    return params
