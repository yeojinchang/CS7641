import numpy as np


def create_nl_feature(X):
    '''
    TODO - Create additional features and add it to the dataset

    returns:
        X_new - (N, d + num_new_features) array with
                additional features added to X such that it
                can classify the points in the dataset.
    '''
    #  Delete this line when you implement the function
    #X_new = create_nl_feature(X)
    print(X)
    x1 = X[:,0]*X[:,1]
    X_new = np.column_stack((X, x1))

    return X_new

