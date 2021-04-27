# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
from utils.const import BaseModels, BaseParams

# AUX METHODS
# ----------------------------------------------------------------------------------------------------

# Print predictor's model data
def print_model(model):
    params_code = 0
    print("Model:")
    if model.model == BaseModels.TREE:
        print("-> Type - (BT) Binary Tree")
        params_code = BaseParams.TREE.index(model.params)
    if model.model == BaseModels.KNN:
        print("-> Type - (KNN) K-Nearest Neighbors")
        params_code = BaseParams.KNN.index(model.params)
    if model.model == BaseModels.SVM:
        print("-> Type - (SVM) Support Vector Machines")
        params_code = BaseParams.SVM.index(model.params)
    if model.model == BaseModels.MLP:
        print("-> Type - (MLP) Multi-Layered Perceptron")
        params_code = BaseParams.MLP.index(model.params)
    print("Params:")
    print("-> Params code: " + str(params_code))
    for param in model.params:
        print("-> " + param + ": " + str(model.params[param]))

# Print predictor's evaluation
def print_evaluation(report, matrix):
    print("Metrics:")
    print("-> F1 Score - ", "{0:.2f}".format(report['f1_score']))
    print("-> Precision - ", "{0:.2f}".format(report['precision']))
    print("-> Recall - ", "{0:.2f}".format(report['recall']))
    print("-> Accuracy - ", "{0:.2f}".format(report['accuracy']))
    print("Confusion Matrix:")
    print(matrix)