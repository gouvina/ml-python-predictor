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
    ui.print_model(model_class, model)
    ui.print_evaluation(report, matrix)

# Search parameters' space and evaluate best for a predictor
def search(dataset, model_class):

    # 0. Aux structures
    models = {}

    # 1. Read dataset
    print('(EVALUATOR) Reading dataset...')  
    X,Y,X_test,Y_test = reader.read_split_dataset(DATA_FOLDER + dataset)

    # 2. Iterate over model types
    if model_class == Models.BASE:
        for model_type in BaseModels:

            # 2.1. Empty list for every parameter configuration from same model type
            models[model_type] = []
            print()

            # 2.2. Iterate over model params
            for model_params in BaseParams[model_type]:

                print('(EVALUATOR) Evaluating model: Type - ' + str(model_type.value) + ', Params - ' + str(BaseParams[model_type].index(model_params)))

                # 2.2.1. Create model
                model = BaseModel(model=model_type, params=model_params)

                # 2.2.2. Train model
                model.train(X,Y)

                # 2.2.3. Evaluate predictor
                report, _ = model.evaluate(X_test, Y_test)

                # 2.2.4. Load predictor's evaluation in evaluation list
                models[model_type].append((model, report['f1_score'], report))

            # 2.3. Sort list based on f1_score
            models[model_type] = sorted(models[model_type], key=lambda x: x[1], reverse=True)

    # 3. Show best results for each model type
    if model_class == Models.BASE:
        for model_type in BaseModels:

            # 3.1. Pick best model
            best_model, _, best_report = models[model_type][0]
            
            # 3.2. Show results
            print()
            print('(EVALUATOR) Results:')
            ui.print_model(Models.BASE, best_model)
            ui.print_evaluation(best_report, None)
            
        
if __name__ == "__main__":

    # Preset arguments values
    exploration = 0
    dataset = IRIS_DATASET
    model_class = Models.BASE
    model_type = BaseModels.MLP
    model_params = BaseParams[model_type][0]

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
    elif exploration == 1:
        search(dataset, model_class)

