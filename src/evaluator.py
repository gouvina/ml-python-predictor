# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
import sys
import time

# DEPENDENCIES (Local)
# ----------------------------------------------------------------------------------------------------
import utils.ui as ui
import utils.parser as parser
import processing.reader as reader
from model.base import BaseModel
from utils.const import DATA_FOLDER, MODEL_FOLDER, IRIS_DATASET, IRIS_PREDICTOR, Models, BaseModels, NeuralModels, BaseParams

# MAIN FUNCTIONS
# ----------------------------------------------------------------------------------------------------

# Evaluate a predictor
def evaluate(dataset, model_class, model_type, model_params):

    # 1. Read dataset
    print('(EVALUATOR) Reading dataset...')  
    X,Y,X_test,Y_test = reader.read_split_dataset(DATA_FOLDER + dataset)
    
    # 2. Create model
    print('(EVALUATOR) Creating model...')    
    model = None
    if model_class == Models.BASE:
        model = BaseModel(model=model_type, params=model_params)

    # 3. Train model
    print('(EVALUATOR) Training predictor...')
    model.train(X,Y)

    # 4. Evaluate predictor
    print('(EVALUATOR) Evaluating predictor...')
    report, matrix = model.evaluate(X_test, Y_test)

    # 5. Show results
    print('(EVALUATOR) Results:')
    ui.print_model(model)
    ui.print_evaluation(report, matrix)   

if __name__ == "__main__":

    # Preset arguments values
    exploration = 0
    dataset = IRIS_DATASET
    model_class = Models.BASE
    model_type = BaseModels.MLP
    model_params = BaseParams.MLP[0]

    # Get length of provided arguments
    args = len(sys.argv)

    # Check if arguments are provided and set them
    if args > 1:
        exploration = int(sys.argv[1])
    if args > 2:
        dataset = sys.argv[2]
    if args > 3:
        model_class, model_type = parser.parse_model_class(sys.argv[3])
    if args > 4:
        model_type, model_params = parser.parse_model_type(model_class, sys.argv[4])
    if args > 5:
        model_params = parser.parse_model_params(model_class, model_type, int(sys.argv[5]))

    if exploration == 0:
        evaluate(dataset, model_class, model_type, model_params)

