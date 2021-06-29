# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
import sys
import time

# DEPENDENCIES (Local)
# ----------------------------------------------------------------------------------------------------
import utils.parser as parser
import processing.reader as reader
from model.classification import ClassificationModel
from model.regression import RegressionModel
from constants.routes import DATA_FOLDER, MODEL_FOLDER, IRIS_DATASET, IRIS_PREDICTOR
from constants.enums import Models, ClassificationModels, RegressionModels, NeuralModels
from constants.params import ClassificationParams, RegressionParams

# MAIN FUNCTIONS
# ----------------------------------------------------------------------------------------------------

# Train a predictor
def train(dataset, model_class, model_type, model_params, model_file):

    # 1. Read dataset
    print('(TRAINER) Reading dataset...')  
    X,Y = reader.read_dataset(DATA_FOLDER + dataset)
    
    # 2. Create model
    print('(TRAINER) Creating model...')
    model = None
    if model_class == Models.CLASSIFICATION:
        model = ClassificationModel(model=model_type, params=model_params)
    elif model_class == Models.REGRESSION:
        model = RegressionModel(model=model_type, params=model_params)

    # 3. Train model
    print('(TRAINER) Training predictor...')
    tic = time.time()
    model.train(X,Y)
    toc = time.time()
    print('(TRAINER) Predictor trained in ' + '{0:.2f}'.format(toc - tic) + ' seconds')

    # 4. Save model
    print('(TRAINER) Saving predictor...')
    model.save(MODEL_FOLDER +  model_file)

    return model

if __name__ == "__main__":

    # Preset arguments values
    dataset = IRIS_DATASET
    model_class = Models.CLASSIFICATION
    model_type = ClassificationModels.MLP
    model_params = ClassificationParams[model_type][0]
    model_file = IRIS_PREDICTOR

    # Get length of provided arguments
    args = len(sys.argv)

    # Check if arguments are provided and set them   
    if args < 3:
        print('(TRAINER) Dataset and model class must be provided')
    else:
        dataset = sys.argv[1]
        model_class, model_type = parser.parse_model_class(sys.argv[2])
        if args > 3:
            model_type, model_params = parser.parse_model_type(model_class, sys.argv[3])
        if args > 4:
            model_params = parser.parse_model_params(model_class, model_type, int(sys.argv[4]))

        train(dataset, model_class, model_type, model_params, model_file)
