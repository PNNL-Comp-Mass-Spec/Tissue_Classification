"""
Provides functions to package together common classification steps
"""

import re
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree

#########################
#
# Classifiers
#
#########################

"""
Abstraction of cross-validation score calculation and stat printing
Called from all specific make_x_model functions

Args:
    model: instance of any sklearn classification model
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test
    
Returns:
    The given model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def fit_model(model, data, labels, num_splits):
    scores = cross_val_score(model, data, labels, cv=num_splits)
    print('Scores:',scores)
    print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(), scores.std() * 2))
    return model.fit(data, labels)

"""
Args:
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test

Returns:
    K Nearest Neighbors classification model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def make_knn_model(data, labels, num_splits):
    knn = KNeighborsClassifier()
    return fit_model(knn, data, labels, num_splits)

"""
Args:
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test

Returns:
    Decision Tree classification model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def make_decisiontree_model(data, labels, num_splits):
    dt = tree.DecisionTreeClassifier()
    return fit_model(dt, data, labels, num_splits)


"""
Use a classification model to make label predictions on new data

Args:
    model: classification model
    data (dataframe): new data to be labelled by the model
    labels (list of strings): list of correct labels for the input data
    
Returns:
    List of strings: List of predicted labels
    Prints accuracy score, as well as the predicted and actual labels
"""
def make_prediction(model, data, labels):
    pred = model.predict(data)
    print('score', accuracy_score(pred, labels))
    print('pred', pred)
    print('actual', labels)

    
#########################
#
# Dataframe adjustments
#
#########################

"""
Rename columns so that all instances of "before" are replaced with "after"

Example usage:
my_df.columns = rename_columns(my_df, 'Adult', 'Human')

Args:
    df (dataframe)
    before (string)
    after (string)
    
Returns:
    List of strings: a list of the new column names
"""
def rename_columns(df, before, after):
    columns = df.columns.values.tolist()
    new_columns = []
    for column in columns:
        new_column = re.sub(before, after, column)
        new_columns.append(new_column)
        
    return new_columns


"""
Args: 
    df (dataframe)
    columns (list of strings): list of all column names in df
    organ_to_columns (dict): mapping of each organ to its column names {str: list of str}
    
Returns: 
    List of strings representing the labels for each dataframe column
"""
def get_labels(df, columns, organ_to_columns):
    labels = []

    for column in columns:
        key = next(key for key, value in organ_to_columns.items() if column in value)
        labels.append(key)
        
    return labels