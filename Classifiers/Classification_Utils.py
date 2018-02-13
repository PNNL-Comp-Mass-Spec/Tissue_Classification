"""
Provides functions to package together common classification steps
"""

import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
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
def knn_model_crossval(data, labels, num_splits):
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
def decisiontree_model_crossval(data, labels, num_splits):
    dt = tree.DecisionTreeClassifier()
    return fit_model(dt, data, labels, num_splits)


"""
Args:
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test

Returns:
    Random forest classification model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def randomforest_model_crossval(data, labels, num_splits):
    rf = RandomForestClassifier(n_estimators=10)
    return fit_model(rf, data, labels, num_splits)

"""
Args:
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test

Returns:
    Naive Bayes Gaussian classification model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def bayes_gaussian_model_crossval(data, labels, num_splits):
    gnb = GaussianNB()
    return fit_model(gnb, data, labels, num_splits)


"""
Args:
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test

Returns:
    List of SVC classification models fitted on all inputted data
    Prints mean cross-validation scores and 95% confidence intervals
"""
def SVC_models_crossval(data, labels, num_splits):
    C = 1.0  # SVM regularization parameter
    models = (SVC(kernel='linear', C=C, probability=True),
              LinearSVC(C=C),
              SVC(kernel='rbf', gamma=0.7, C=C, probability=True),
              SVC(kernel='poly', degree=3, C=C, probability=True))

    # Fit all the models
    models = (fit_model(clf, data, labels, num_splits) for clf in models)
    model_list = list(models)

    return model_list

"""
Use a classification model to make label predictions on test data set

Args:
    model: classification model
    data (dataframe): new data to be labelled by the model
    labels (list of strings): list of correct labels for the input data
    print_details (boolean, optional): Determines whether to print prediction information. Defaults to True
    
Returns:
    List of strings: List of predicted labels
    Prints accuracy score, as well as the predicted and actual labels
"""
def make_test_prediction(model, data, labels, print_details=True):
    pred = model.predict(data)
    if(print_details):
        print('score', accuracy_score(pred, labels))
        print('pred', pred)
        print('actual', labels)
    
    return pred

"""
Args:
    model: SKLearn classification model
    data (dataframe): test data
    idx (int): index of sample to show prediction probabilities for

Returns:
    Prints each class's prediction probability for the specified data sample
"""
def show_prediction_probabilities(model, data, idx):
    pred_probabilities = model.predict_proba(data)
    classes = model.classes_

    print('Prediction probabilities for sample:')
    for prob in zip(classes, pred_probabilities[idx]):
        print(prob[0], ':', prob[1])
    
    
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


"""
Fits new data to training features so that it can be classified

Args:
    original_df (dataframe): data used to train classification model
    new_df (dataframe): new data to be classified

Returns:
    dataframe: new_df joined to the features of the training data. This dataframe can now be classified by a model trained with original_df
"""
def fit_new_data(original_df, new_df):
    return original_df.iloc[:,:0].join(new_df)
