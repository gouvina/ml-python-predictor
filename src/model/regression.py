# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
import warnings
warnings.simplefilter("ignore", UserWarning)
import pickle
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import explained_variance_score, r2_score, max_error, mean_absolute_error, mean_squared_error, median_absolute_error

# DEPENDENCIES (Local)
# ----------------------------------------------------------------------------------------------------
from constants.enums import RegressionModels
from constants.params import RegressionParams

# MAIN CLASS
# ----------------------------------------------------------------------------------------------------
class RegressionModel():

    # Aux Function - Save model in pickle file
    def save(self, path):
        pickle.dump(self.predictor, open(path, 'wb'))

    # Aux Function - Load model from pickle file
    def load(self, path):
        self.classifier = pickle.load(open(path, 'rb'))

    # Constructor
    def __init__(self, model=RegressionModels.MLP, params=RegressionParams[RegressionModels.MLP][0]):

        # Generic configuration
        self.model = model
        self.params = params

        # Interfaces
        self.predictor = None
        self.scaler = None

    # Train predictor for dataset X and tags Y
    def train(self, X, Y):

        # Create model's instance based on params
        if self.model == RegressionModels.KNN:
            self.predictor = KNeighborsRegressor(**self.params)
        elif self.model == RegressionModels.SVM:
            self.predictor = SVR(**self.params)
        elif self.model == RegressionModels.LIR:
            self.predictor = LinearRegression(**self.params)
        elif self.model == RegressionModels.MLP:
            self.predictor = MLPRegressor(**self.params)
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)

        # Train using dataset X and tags Y
        self.predictor.fit(X, Y)

    # Predict tags for X
    def predict(self, X):
        return self.predictor.predict(X)

    # Evaluate predictor for testset X and tags Y 
    def evaluate(self, X, Y):

        # Normalize values if scaler exists
        if self.scaler is not None:
            X = self.scaler.transform(X)

        # Get prediction to compare
        prediction = self.predict(X)
        
        # Get metrics
        explained_variance = explained_variance_score(Y, prediction)
        r2 = r2_score(Y, prediction)
        max_err = max_error(Y, prediction)
        absolute_err = mean_absolute_error(Y, prediction)
        squared_err = mean_squared_error(Y, prediction)
        median_err = median_absolute_error(Y, prediction)

        # Fill report
        report = {
            'scores': {
                'explained_variance': explained_variance,
                'r2_score': r2,
            },
            'errors': {
                'mean_squared_error': squared_err,
                'mean_absolute_error': absolute_err,
                'median_absolute_error': median_err,
                'max_error': max_err,
            }
        }

        return report
