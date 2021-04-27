# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
from utils.const import Models, BaseModels, NeuralModels, BaseParams

# AUX METHODS
# ----------------------------------------------------------------------------------------------------

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

def parse_model_type(model_class, model_type):

    model = None
    model_params = None

    if model_class == Models.BASE:
        if model_type == 'tree':
            model = BaseModels.TREE
            model_params = BaseParams.TREE[0]
        if model_type == 'knn':
            model = BaseModels.KNN
            model_params = BaseParams.KNN[0]
        if model_type == 'svm':
            model = BaseModels.SVM
            model_params = BaseParams.SVM[0]
        if model_type == 'mlp':
            model = BaseModels.MLP
            model_params = BaseParams.MLP[0]
    else:
        if model_type == 'mlp':
            model = NeuralModels.MLP

    return model, model_params

def parse_model_params(model_class, model_type, model_params):

    params = None

    if model_class == Models.BASE:
        if model_type == BaseModels.TREE:
            params = BaseParams.TREE[model_params]
        if model_type == BaseModels.KNN:
            params = BaseParams.KNN[model_params]
        if model_type == BaseModels.SVM:
            params = BaseParams.SVM[model_params]
        if model_type == BaseModels.MLP:
            params = BaseParams.MLP[model_params]
        
    return params