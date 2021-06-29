# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
from constants.enums import Models, ClassificationModels, RegressionModels, NeuralModels
from constants.params import ClassificationParams, RegressionParams

# AUX METHODS
# ----------------------------------------------------------------------------------------------------

# Print predictor's model data
def print_model(model_class, model):

    params_code = 0
    if model_class == Models.CLASSIFICATION:
        params_code = ClassificationParams[model.model].index(model.params)
    
    print("Model:")
    if model.model == ClassificationModels.TREE:
        print("-> Type - (BT) Binary Tree")
    if model.model == ClassificationModels.GNB:
        print("-> Type - (GNB) Gaussian Naive Bayes")
    if model.model == ClassificationModels.KNN or model.model == RegressionModels.KNN:
        print("-> Type - (KNN) K-Nearest Neighbors")
    if model.model == ClassificationModels.SVM or model.model == RegressionModels.SVM:
        print("-> Type - (SVM) Support Vector Machines")
    if model.model == RegressionModels.LIR:
        print("-> Type - (LiR) Linear Regression")
    if model.model == ClassificationModels.LOR:
        print("-> Type - (LoR) Logistic Regression")
    if model.model == ClassificationModels.MLP or model.model == RegressionModels.MLP:
        print("-> Type - (MLP) Multi-Layered Perceptron")
    print("Params:")
    print("-> Params code: " + str(params_code))
    for param in model.params:
        print("-> " + param + ": " + str(model.params[param]))

# Print predictor's evaluation
def print_evaluation(model_class, report):
    if model_class == Models.CLASSIFICATION:
        print("Metrics:")
        print("-> F1 Score - ", "{0:.2f}".format(report['f1_score']))
        print("-> Precision - ", "{0:.2f}".format(report['precision']))
        print("-> Recall - ", "{0:.2f}".format(report['recall']))
        print("-> Accuracy - ", "{0:.2f}".format(report['accuracy']))
    elif model_class == Models.REGRESSION:
        print("Metrics:")
        print("-> R2 Score - ", "{0:.2f}".format(report['scores']['r2_score']))
        print("-> Explained variance - ", "{0:.2f}".format(report['scores']['explained_variance']))
        print("Errors:")
        print("-> Mean Squared Error - ", "{0:.2f}".format(report['errors']['mean_squared_error']))
        print("-> Mean Absolute Error - ", "{0:.2f}".format(report['errors']['mean_absolute_error']))
        print("-> Median Absolute Error - ", "{0:.2f}".format(report['errors']['median_absolute_error']))
        print("-> Max Error - ", "{0:.2f}".format(report['errors']['max_error']))

# Print predictor evaluation's confusion matrix
def print_matrix(matrix):
    if matrix is not None:
        print("Confusion Matrix:")
        print(matrix)