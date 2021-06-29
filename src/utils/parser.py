# DEPENDENCIES (Local)
# ----------------------------------------------------------------------------------------------------
from constants.enums import Models, BaseModels, NeuralModels
from constants.params import BaseParams

# AUX METHODS
# ----------------------------------------------------------------------------------------------------

# Parse model class into enum with types:
# -> Base = Models based on common ML methods
# -> Neural = Models based on neural networks
def parse_model_class(model_class):
    
    model = None
    model_type = None

    if model_class == 'base':
        model = Models.BASE
        model_type = BaseModels.MLP
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

    if model_class == Models.BASE:
        if model_type == 'tree':
            model = BaseModels.TREE
        if model_type == 'gnb':
            model = BaseModels.GNB
        if model_type == 'knn':
            model = BaseModels.KNN
        if model_type == 'svm':
            model = BaseModels.SVM
        if model_type == 'lor':
            model = BaseModels.LOR
        if model_type == 'mlp':
            model = BaseModels.MLP
        model_params = BaseParams[model][0]
    else:
        if model_type == 'mlp':
            model = NeuralModels.MLP

    return model, model_params

# Retrieve model training params depending on class and type
def parse_model_params(model_class, model_type, model_params):

    params = None

    if model_class == Models.BASE:
        params = BaseParams[model_type][model_params]
        
    return params
