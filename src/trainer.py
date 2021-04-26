# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
import os
import sys
import time

# DEPENDENCIES (Local)
# ----------------------------------------------------------------------------------------------------
import utils.const as const
import processing.reader as reader
from model.base import BaseModel

# MAIN FUNCTIONS
# ----------------------------------------------------------------------------------------------------

# Train a predictor
def train():

    # 1. Read dataset
    print('(TRAINER) Reading dataset...')  
    X,Y = reader.read_dataset(const.DATA_FOLDER + const.IRIS_DATASET)
    
    # 2. Create model
    print('(TRAINER) Creating model...')    
    model = BaseModel()

    # 3. Train model
    print('(TRAINER) Training predictor...')
    tic = time.time()
    model.train(X,Y)
    toc = time.time()
    print('(TRAINER) Predictor trained in ' + '{0:.2f}'.format(toc - tic) + ' seconds')

    # 4. Save model
    print('(TRAINER) Saving predictor...')
    model.save(const.MODEL_FOLDER +  const.IRIS_PREDICTOR)

    return model

if __name__ == "__main__":
    train()
