# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
from utils.const import BaseModels

# AUX METHODS
# ----------------------------------------------------------------------------------------------------

# Print predictor's model data
def print_model(model):
    print("Model:")
    if model.model == BaseModels.TREE:
        print("-> Type - (BT) Binary Tree")
    if model.model == BaseModels.KNN:
        print("-> Type - (KNN) K-Nearest Neighbors")
    if model.model == BaseModels.SVM:
        print("-> Type - (SVM) Support Vector Machines")
    if model.model == BaseModels.MLP:
        print("-> Type - (MLP) Multi-Layered Perceptron")
    print("Params:")
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