# Predictor (Python)
Base Project for training and evaluating machine learning predictors based on different models and libraries, and trained for tagged datasets provided as CSV files.

## Branches
Depending on what version of the base project is needed (and which features), different groups of features have been organized in the next branches:
- `main` - Version using standard machine learning models from `sklearn`
- `neural` - Version using standard neural network architectures from `keras`

**Current Branch:** `main`

## Features
This version of the base project supports the next features:
- Training of custom predictor for both classification and regression problems
- Evaluation of those same predictors, using specific parameter configurations or exploring a list of possibilities

It also has integrated the following datasets to help testing:
- `iris` - Classification dataset, 150 samples, 3 classes, 4 continous attributes
- `covertype` - Classification dataset, 10959 samples, 5 classes, +50 continous and discrete attributes
- `boston` - Regression dataset, 506 samples, 13 continous attributes

## Requirements
In order to run the project, it is needed to have downloaded and installed the following software:
- Python - https://www.python.org/downloads/

## Setup
In order to run the project, it is needed to follow the next steps, taking into account they are designed to run the project *locally*. URL's should change otherwise:
1. Clone repository
2. (Optional) Generate virtual environment - `python -m venv env`
3. (Optional) Activate virtual environment:
    - Windows - `env/Scripts/activate.ps1` 
    - Unix - `./env/Scripts/activate`
4. Install requirements - `pip install -r requirements.txt`
5. Add target CSV dataset to `data` folder

## Run
There are two main scripts:
1. Trainer: Used for training a predictor and generating its file
2. Evaluator: Used for evaluating a predictor using a set of metrics

### Trainer
`python trainer.py <path> <class> <type> <params>` <br>

Parameters:
1. `path`: path to a CSV file that contains the dataset.
2. `class`: a string representing the kind of model based on the strategy used and the target dataset
    - `'clasification'` for classification datasets and models
    - `'regression'` for regression datasets and models
3. `type`: an integer representing the type of model based on the model class previously chosen
    - Classification:
        - `'tree'` - Binary Decision Tree
        - `'gnb'` - Gaussian Naive Bayes
        - `'knn'` - K Nearest Neighbors
        - `'svm'` - Support Vector Machines
        - `'lor'` - Logistic Regression
        - `'mlp'` - Multi Layered Perceptron
    - Regression:
        - `'knn'` - K Nearest Neighbors
        - `'svm'` - Support Vector Machines
        - `'lir'` - Linear Regression
        - `'mlp'` - Multi Layered Perceptron
4. `params`: an integer representing the set of params chosen for the model, depending on the class and type. The list of potential params is located in `constants/params.py`.

### Evaluator
`python evaluator.py <exploration> <path> <class> <type> <params>` <br>

 Parameters:
1. `exploration`: an integer representing if evaluation should be done using the provided parameters or trying different combinations and choosing the best one
    - Evaluate current model and params = 0
    - Search best model and params = 1
    - Search best params for model = 2
2. `path`: path to a CSV file that contains the dataset.
2. `class`: a string representing the kind of model based on the strategy used and the target dataset
    - `'clasification'` for classification datasets and models
    - `'regression'` for regression datasets and models
3. `type`: an integer representing the type of model based on the model class previously chosen
    - Classification:
        - `'tree'` - Binary Decision Tree
        - `'gnb'` - Gaussian Naive Bayes
        - `'knn'` - K Nearest Neighbors
        - `'svm'` - Support Vector Machines
        - `'lor'` - Logistic Regression
        - `'mlp'` - Multi Layered Perceptron
    - Regression:
        - `'knn'` - K Nearest Neighbors
        - `'svm'` - Support Vector Machines
        - `'lir'` - Linear Regression
        - `'mlp'` - Multi Layered Perceptron
5. `params`: an integer representing the set of params chosen for the model, depending on the class and type. The list of potential params is located in `constants/params.py`.
