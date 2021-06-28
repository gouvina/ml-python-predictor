# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
from enum import Enum

# CONSTANTS
# ----------------------------------------------------------------------------------------------------

# Model classes
class Models(Enum):
    BASE = 1
    NEURAL = 2

# Base model types
class BaseModels(Enum):
    TREE = '(BT) Binary Tree'
    KNN = '(KNN) K-Nearest Neighbors'
    SVM = '(SVM) Support Vector Machines'
    MLP = '(MLP) Multi-Layered Perceptron'

# Neural model types
class NeuralModels(Enum):
    MLP = 1