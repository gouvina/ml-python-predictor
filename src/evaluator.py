# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
import sys
import time

# DEPENDENCIES (Local)
# ----------------------------------------------------------------------------------------------------
import utils.ui as ui
import utils.const as const
import processing.reader as reader
from model.base import BaseModel

# MAIN FUNCTIONS
# ----------------------------------------------------------------------------------------------------

# 
def evaluate():

    # 1. Read dataset
    print('(EVALUATOR) Reading dataset...')  
    X,Y,X_test,Y_test = reader.read_split_dataset(const.DATA_FOLDER + const.IRIS_DATASET)
    
    # 2. Create model
    print('(EVALUATOR) Creating model...')    
    model = BaseModel()

    # 3. Train model
    print('(EVALUATOR) Training predictor...')
    tic = time.time()
    model.train(X,Y)
    toc = time.time()
    print('(EVALUATOR) Predictor trained in ' + '{0:.2f}'.format(toc - tic) + ' seconds')

    # 4. Evaluate predictor
    print('(EVALUATOR) Evaluating predictor...')
    report, matrix = model.evaluate(X_test, Y_test)

    # 5. Show results
    print('(EVALUATOR) Results:')
    ui.print_model(model)
    ui.print_evaluation(report, matrix)
    

if __name__ == "__main__":
    evaluate()
