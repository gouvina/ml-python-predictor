# Predictor (Python)
Base Project for training and evaluating machine learning predictors based on different models and libraries, and trained for tagged datasets provided as CSV files.

## Setup
1 - (Optional) Generate virtual environment: <br>
`python -m venv env` <br>

2 - (Optional) Activate virtual environment: <br>
`env/Scripts/activate.bat` (Windows)<br>
`./env/Scripts/activate` (Unix)<br>

3 - Install requirements: <br>
`pip install -r requirements.txt` <br>

4 - Add target CSV dataset to `data` folder

## Usage
There are two main scripts:
1. Trainer: Used for training a predictor and generating its file
2. Evaluator: Used for evaluating a predictor using a set of metrics

### Trainer
`python trainer.py <path> <class> <type> <params>` <br>

Parameters:
1. `path`: path to a CSV file that contains the dataset.
2. `class`: an integer representing the kind of model based on the strategy used
    - Base = 0
    - Neural = 1
3. `type`: an integer representing the type of model based on the model class previously chosen
    - Base:
        - Tree (Binary Decision Tree) = 0
        - KNN (K Nearest Neighbors) = 1
        - SVM (Support Vector Machines) = 2
        - MLP (Multi Layered Perceptron) = 3
    - Neural:
        - MLP (Multi Layered Perceptron) = 0
4. `params`: an integer representing the set of params chosen for the model, depending on the class and type. The list of potential params is located in `constants/params.py`.

### Evaluator
`python evaluator.py <exploration> <path> <class> <type> <params>` <br>

 Parameters:
1. `exploration`: an integer representing if evaluation should be done using the provided parameters or trying different combinations and choosing the best one
    - Single evalution = 0
    - Search exploration = 1
2. `path`: path to a CSV file that contains the dataset.
3. `class`: an integer representing the kind of model based on the strategy used
    - Base = 0
    - Neural = 1
4. `type`: an integer representing the type of model based on the model class previously chosen
    - Base:
        - Tree (Binary Decision Tree) = 0
        - KNN (K Nearest Neighbors) = 1
        - SVM (Support Vector Machines) = 2
        - MLP (Multi Layered Perceptron) = 3
    - Neural:
        - MLP (Multi Layered Perceptron) = 0
5. `params`: an integer representing the set of params chosen for the model, depending on the class and type. The list of potential params is located in `constants/params.py`.
