# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
import sys

# DEPENDENCIES (Local)
# ----------------------------------------------------------------------------------------------------
import utils.ui as ui
import utils.parser as parser
import processing.reader as reader
from model.classification import ClassificationModel
from model.regression import RegressionModel
from constants.routes import DATA_FOLDER, IRIS_DATASET
from constants.enums import Models, ClassificationModels, NeuralModels, RegressionModels
from constants.params import ClassificationParams, RegressionParams

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
    if model_class == Models.CLASSIFICATION:
        model = ClassificationModel(model=model_type, params=model_params)
    elif model_class == Models.REGRESSION:
        model = RegressionModel(model=model_type, params=model_params)

    # 3. Train model
    print('(EVALUATOR) Training predictor...')
    model.train(X,Y)

    if model_class == Models.CLASSIFICATION:
        # 4. Evaluate predictor
        print('(EVALUATOR) Evaluating predictor...')
        report, matrix = model.evaluate(X_test, Y_test)

        # 5. Show results
        print('(EVALUATOR) Results:')
        ui.print_model(model_class, model)
        ui.print_evaluation(model_class, report)
        ui.print_matrix(matrix)

    elif model_class == Models.REGRESSION:
        # 4. Evaluate predictor
        print('(EVALUATOR) Evaluating predictor...')
        report = model.evaluate(X_test, Y_test)

        # 5. Show results
        print('(EVALUATOR) Results:')
        ui.print_model(model_class, model)
        ui.print_evaluation(model_class, report)

# Search parameters' space and evaluate best for a predictor of specific type
def search(dataset, model_class, model_type):

    # 0. Aux structures
    models = []

    # 1. Read dataset
    print('(EVALUATOR) Reading dataset...')  
    X,Y,X_test,Y_test = reader.read_split_dataset(DATA_FOLDER + dataset)

    # 2. Iterate over model types
    if model_class == Models.CLASSIFICATION:

        # 2.1. Iterate over model params
        for model_params in ClassificationParams[model_type]:

            print('(EVALUATOR) Evaluating model: Type - ' + str(model_type.value) + ', Params - ' + str(ClassificationParams[model_type].index(model_params)))

            # 2.2.1. Create model
            model = ClassificationModel(model=model_type, params=model_params)

            # 2.2.2. Train model
            model.train(X,Y)

            # 2.2.3. Evaluate predictor
            report, _ = model.evaluate(X_test, Y_test)

            # 2.2.4. Load predictor's evaluation in evaluation list
            models.append((model, report['f1_score'], report))

        # 2.2. Sort list based on f1_score
        models = sorted(models, key=lambda x: x[1], reverse=True)
    
    elif model_class == Models.REGRESSION:

        # 2.1. Iterate over model params
        for model_params in RegressionParams[model_type]:

            print('(EVALUATOR) Evaluating model: Type - ' + str(model_type.value) + ', Params - ' + str(RegressionParams[model_type].index(model_params)))

            # 2.2.1. Create model
            model = RegressionModel(model=model_type, params=model_params)

            # 2.2.2. Train model
            model.train(X,Y)

            # 2.2.3. Evaluate predictor
            report = model.evaluate(X_test, Y_test)

            # 2.2.4. Load predictor's evaluation in evaluation list
            models.append((model, report['scores']['r2_score'], report))

        # 2.2. Sort list based on f1_score
        models = sorted(models, key=lambda x: x[1], reverse=True)

    # 3. Show best results for each model type
    best_model, _, best_report = models[0]
    print()
    print('(EVALUATOR) Results:')
    ui.print_model(model_class, best_model)
    ui.print_evaluation(model_class, best_report)
            
# Search parameters' space and evaluate best for a predictor
def search_all(dataset, model_class):

    # 0. Aux structures
    models = {}

    # 1. Read dataset
    print('(EVALUATOR) Reading dataset...')  
    X,Y,X_test,Y_test = reader.read_split_dataset(DATA_FOLDER + dataset)

    # 2. Iterate over model types
    if model_class == Models.CLASSIFICATION:
        for model_type in ClassificationModels:

            # 2.1. Empty list for every parameter configuration from same model type
            models[model_type] = []
            print()

            # 2.2. Iterate over model params
            for model_params in ClassificationParams[model_type]:

                print('(EVALUATOR) Evaluating model: Type - ' + str(model_type.value) + ', Params - ' + str(ClassificationParams[model_type].index(model_params)))

                # 2.2.1. Create model
                model = ClassificationModel(model=model_type, params=model_params)

                # 2.2.2. Train model
                model.train(X,Y)

                # 2.2.3. Evaluate predictor
                report, _ = model.evaluate(X_test, Y_test)

                # 2.2.4. Load predictor's evaluation in evaluation list
                models[model_type].append((model, report['f1_score'], report))

            # 2.3. Sort list based on f1_score
            models[model_type] = sorted(models[model_type], key=lambda x: x[1], reverse=True)
    elif model_class == Models.REGRESSION:
        for model_type in RegressionModels:

            # 2.1. Empty list for every parameter configuration from same model type
            models[model_type] = []
            print()

            # 2.2. Iterate over model params
            for model_params in RegressionParams[model_type]:

                print('(EVALUATOR) Evaluating model: Type - ' + str(model_type.value) + ', Params - ' + str(RegressionParams[model_type].index(model_params)))

                # 2.2.1. Create model
                model = RegressionModel(model=model_type, params=model_params)

                # 2.2.2. Train model
                model.train(X,Y)

                # 2.2.3. Evaluate predictor
                report = model.evaluate(X_test, Y_test)

                # 2.2.4. Load predictor's evaluation in evaluation list
                models[model_type].append((model, report['scores']['r2_score'], report))

            # 2.3. Sort list based on f1_score
            models[model_type] = sorted(models[model_type], key=lambda x: x[1], reverse=True)

    # 3. Show best results for each model type
    if model_class == Models.CLASSIFICATION:
        for model_type in ClassificationModels:

            # 3.1. Pick best model
            best_model, _, best_report = models[model_type][0]
            
            # 3.2. Show results
            print()
            print('(EVALUATOR) Results:')
            ui.print_model(Models.CLASSIFICATION, best_model)
            ui.print_evaluation(model_class, best_report)
    elif model_class == Models.REGRESSION:
        for model_type in RegressionModels:

            # 3.1. Pick best model
            best_model, _, best_report = models[model_type][0]
            
            # 3.2. Show results
            print()
            print('(EVALUATOR) Results:')
            ui.print_model(Models.REGRESSION, best_model)
            ui.print_evaluation(model_class, best_report)

if __name__ == "__main__":

    # Preset arguments values
    exploration = 0
    dataset = IRIS_DATASET
    model_class = Models.CLASSIFICATION
    model_type = ClassificationModels.MLP
    model_params = ClassificationParams[model_type][0]

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
        search(dataset, model_class, model_type)
    elif exploration == 2:
        search_all(dataset, model_class)

