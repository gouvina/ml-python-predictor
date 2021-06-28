# DEPENDENCIES (Libraries)
# ----------------------------------------------------------------------------------------------------
import pandas as pd

# MAIN FUNCTIONS
# ----------------------------------------------------------------------------------------------------

# Read generic dataset
def read_dataset(path):

    # Read CSV file as dataframe
    df = pd.read_csv(path)

    # Shuffle dataframe
    df = df.sample(frac=1)

    # Separate dataset and tags      
    X = df.iloc[:,:-1]
    Y = df.iloc[:,-1]

    return (X,Y)

# Read generic dataset and split it
def read_split_dataset(path, proportion=0.8):

    # Read CSV file as dataframe
    df = pd.read_csv(path)

    # Shuffle dataframe
    df = df.sample(frac=1)

    # Separate dataset and testset
    data = df.head(int(len(df)*proportion))
    test = df.tail(int(len(df)*(1-proportion)))

    # Separate data and tags
    X = data.iloc[:,:-1]
    Y = data.iloc[:,-1]
    X_test = test.iloc[:,:-1]
    Y_test = test.iloc[:,-1]

    return (X,Y,X_test,Y_test)