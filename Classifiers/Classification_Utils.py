"""
Provides functions to package together common classification steps
"""

import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from sklearn import tree
from sklearn.decomposition import PCA, NMF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import RFECV, SelectFromModel, SelectKBest, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_predict, cross_val_score, GridSearchCV
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.svm import LinearSVC, SVC

#########################
#
# Constants
#
#########################

ESTIMATORS = [RandomForestClassifier(), 
              ExtraTreesClassifier(), 
              LinearSVC(C=0.01, penalty="l1", dual=False)]
MIN_SAMPLES_SPLIT = [2, 3, 4, 5, 10]
N_ESTIMATORS = [25, 50, 100, 200]
N_FEATURES_OPTIONS = [2, 4, 8]
K_FEATURES_OPTIONS = [10, 100, 1000]
PERCENTILE_OPTIONS = [5, 10, 25, 50, 75, 90, 100]

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
    Naive Bayes Multinomial classification model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def bayes_multinomial_model_crossval(data, labels, num_splits):
    mnb = MultinomialNB()
    return fit_model(mnb, data, labels, num_splits)

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
    Logistic Regression classification model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def logistic_regression_model_crossval(data, labels, num_splits):
    lr = LogisticRegression()
    return fit_model(lr, data, labels, num_splits)


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
Args:
    data (dataframe): contains all data that will be used to fit the model
    labels (list of strings): corresponding labels for each row of data
    num_splits (int): number of train-test splits to test

Returns:
    Gradient Boosting Classifier model fitted on all inputted data
    Prints mean cross-validation score and 95% confidence interval
"""
def gradient_boosting_crossval(data, labels, num_splits):
    gbc = GradientBoostingClassifier()
    return fit_model(gbc, data, labels, num_splits)


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
# Grid Searches for Classifier hyperparameter tuning
# 
# Each grid search finds the best combination of dimensionality reduction and classification parameters for a given model
#
#########################

"""
Args:
    cv (int): Number of cross-validation folds
    n_jobs(int): Number of jobs to run in parallel
    
Returns:
    GridSearchCV: sklearn.model_selection.GridSearchCV instance for SVC classification variations; attributes include best_estimator_, best_score_, and best_params_
"""
def svc_grid_search(cv, n_jobs, scoring=None):

    C_OPTIONS = [.01, .1, 1, 10, 100, 1000]
    KERNELS = ['linear', 'rbf', 'poly']
    GAMMAS = [0.001, 0.01, 0.1, 1]

    svc_grid = {
            'classify__C': C_OPTIONS,
            'classify__kernel': KERNELS,
            'classify__gamma': GAMMAS
    }

    return grid_search(cv, n_jobs, SVC(probability=True), svc_grid, scoring=scoring)
    
"""
Args:
    cv (int): Number of cross-validation folds
    n_jobs(int): Number of jobs to run in parallel
    
Returns:
    GridSearchCV: sklearn.model_selection.GridSearchCV instance for KNN variations; attributes include best_estimator_, best_score_, and best_params_
"""
def knn_grid_search(cv, n_jobs, scoring=None):

    N_NEIGHBORS = [1, 3, 5, 10, 20]
 
    knn_grid = {
            'classify__n_neighbors': N_NEIGHBORS
    }

    return grid_search(cv, n_jobs, KNeighborsClassifier(), knn_grid, scoring=scoring)

"""
Args:
    cv (int): Number of cross-validation folds
    n_jobs(int): Number of jobs to run in parallel
    
Returns:
    GridSearchCV: sklearn.model_selection.GridSearchCV instance for Multinomial Naive Bayes variations; attributes include best_estimator_, best_score_, and best_params_
"""
def mnb_grid_search(cv, n_jobs, scoring=None):

    ALPHAS = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
 
    mnb_grid = {
            'classify__alpha': ALPHAS
    }

    return grid_search(cv, n_jobs, MultinomialNB(), mnb_grid, scoring=scoring)

"""
Args:
    cv (int): Number of cross-validation folds
    n_jobs(int): Number of jobs to run in parallel
    
Returns:
    GridSearchCV: sklearn.model_selection.GridSearchCV instance for RandomForest variations; attributes include best_estimator_, best_score_, and best_params_
"""
def rf_grid_search(cv, n_jobs, scoring=None):
    
    MAX_FEATURES = ['auto', 'sqrt', 'log2']
    
    rf_grid = {
            'classify__n_estimators': N_ESTIMATORS,
            'classify__min_samples_split': MIN_SAMPLES_SPLIT,
            'classify__max_features': MAX_FEATURES
    }
    
    return grid_search(cv, n_jobs, RandomForestClassifier(), rf_grid, scoring=scoring)


"""
Args:
    cv (int): Number of cross-validation folds
    n_jobs(int): Number of jobs to run in parallel
    
Returns:
    GridSearchCV: sklearn.model_selection.GridSearchCV instance for GradientBoostingClassifier variations; attributes include best_estimator_, best_score_, and best_params_
"""
def gbc_grid_search(cv, n_jobs, scoring=None):
    
    MAX_DEPTH = range(5,16,2)
    
    gbc_grid = {
            'classify__n_estimators': N_ESTIMATORS,
            'classify__min_samples_split': MIN_SAMPLES_SPLIT,
            'classify__max_depth': MAX_DEPTH 
    }
    
    return grid_search(cv, n_jobs, GradientBoostingClassifier(), gbc_grid, scoring=scoring)


"""
Args:
    cv (int): Number of cross-validation folds
    n_jobs(int): Number of jobs to run in parallel
    
Returns:
    GridSearchCV: sklearn.model_selection.GridSearchCV instance for MLPClassifier variations; attributes include best_estimator_, best_score_, and best_params_
"""
def mlp_grid_search(cv, n_jobs, scoring=None):
    
    mlp_grid = {
        'classify__hidden_layer_sizes': [(10,), (100,), (500,), (1000,)],
        'classify__tol': [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        #'classify__epsilon': [1e-3, 1e-7, 1e-8, 1e-9, 1e-8],
        #'classify__activation': ['identity', 'logistic', 'tanh', 'relu'],
        'classify__max_iter': [300, 500],
        'classify__solver': ['lbfgs', 'sgd', 'adam']
    }
    
    return grid_search(cv, n_jobs, MLPClassifier(), mlp_grid, scoring=scoring)

"""
General grid_search function called by all specific grid searches
"""
def grid_search(cv, n_jobs, model, model_grid, scoring=None):
    pipe = Pipeline([
        ('reduce_dim', PCA()),
        ('classify', model)])
    
    param_grid = [
        {
            'reduce_dim': [PCA(), NMF()],
            'reduce_dim__n_components': N_FEATURES_OPTIONS,
        },
#        {
#            'reduce_dim': [SelectKBest()],
#            'reduce_dim__k': K_FEATURES_OPTIONS,
#        },
        {
            'reduce_dim': [SelectPercentile()],
            'reduce_dim__percentile': PERCENTILE_OPTIONS,
        },
        {
            'reduce_dim': [SelectFromModel(RandomForestClassifier())],
            'reduce_dim__estimator': [*ESTIMATORS],
        },
    ]
    
    for feature_grid in param_grid:
        feature_grid.update(model_grid)

    grid = GridSearchCV(pipe, cv=cv, n_jobs=n_jobs, param_grid=param_grid, scoring=scoring)
    return grid


#########################
#
# Dataframe adjustments
#
#########################

""" Reads in tab separated files containing a 'Peptide' column

Args:
    file_dir (string): path to directory containing files
    file_paths (list of strings): list of file names of csvs to read
    
Returns:
    dataframe containing data from all csvs referenced by file_paths. Dataframe index is 'Peptide'; each column represents a single sample.
"""
def combine_csvs(file_dir, file_names):
    
    dfs = []

    for file in file_names:
        df = pd.read_csv(file_dir + file, sep='\t', lineterminator='\r')
        dfs.append(df)

    combined_df = pd.DataFrame()
    for df in dfs:
        df.set_index('Peptide', inplace=True)
        combined_df = combined_df.join(df, how='outer')
        
    return combined_df


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
    columns (list of strings): list of all column names in df
    organ_to_columns (dict): mapping of each organ to its column names {str: list of str}
    
Returns: 
    List of strings representing the labels for each dataframe column
"""
def get_labels(columns, organ_to_columns):
    labels = []

    for column in columns:
        key = next(key for key, value in organ_to_columns.items() if column in value)
        labels.append(key)
        
    return labels

"""
Args:
    df (dataframe): rows are peptide/proteins, columns are samples, data = abundance values
    
Returns:
    dataframe transformed so that rows represent all pairwise peptide/protein ratios
"""
def pairwise_transform(df):

    index = df.index.values.tolist()
    
    new_indices = []
    new_data = {}
    
    for col in df.columns:                             # For each sample
        for i in index:                                # For each pair of peptides
            for j in index:
                ratio = df.loc[i, col]/df.loc[j, col]  # Calculate ratio
                new_index = i + '/' + j                # Create new index value 'i/j'
                if new_index not in new_indices:
                    new_indices.append(new_index)      # Add new index to list

                data = new_data.get(col, list())       # Add ratio to corresponding data
                data.append(ratio) 

                new_data[col] = data

    transformed_df = pd.DataFrame(new_data, columns=df.columns, index=new_indices)
    return transformed_df
    

"""
Fits new data to training features so that it can be classified

Args:
    original_df (dataframe): data used to train classification model
    new_df (dataframe): new data to be classified

Returns:
    dataframe: new_df joined to the features of the training data. This dataframe can now be classified by a model trained with original_df
"""
def fit_new_data(original_df, new_df):

    #fitted_data = original_df.join(new_df)
    fitted_data = original_df.drop(original_df.columns[:], axis=1).join(new_df)
    
    fitted_data.iloc[:,:] = np.log2(fitted_data.iloc[:,:])
    fitted_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    fitted_data = fitted_data.fillna(fitted_data.min().min()/2)
    
    #median_of_medians = fitted_data.median().median()
    #fitted_data /= fitted_data.median(axis=0) # divide each value by sample median
    #fitted_data *= median_of_medians
    
    fitted_data.iloc[:,:] = RobustScaler().fit_transform(fitted_data)
    
    # fitted_data.drop(original_df.columns, inplace=true)
    
    return fitted_data.T


#########################
#
# Plotting
#
#########################

""" Creates a mapping of each tissue to all corresponding columns in the dataframe. Assumes columns contain the names of the tissues (e.g. column names might look like 'Lung_01', 'Lung_02', 'Brain_01' etc.

Args:
    df (dataframe): columns represent samples, named with the tissue type
    list_of_tissues (list of strings): all tissues represented in the dataframe
    
Returns:
    dict {string: list of strings} where keys are tissues and values are corresponding column names
"""
def map_tissues_to_columns(df, list_of_tissues):
    
    tissues_to_columns = dict([(key, []) for key in list_of_tissues])

    for column_name in df.columns.values.tolist():
        for tissue in list_of_tissues:
            if tissue in column_name:
                tissues_to_columns[tissue].append(column_name)
                continue
                
    return tissues_to_columns

"""
From SKLearn ConfusionMatrix documentation:
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
"""
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

"""
From SKLearn ConfusionMatrix documentation:
http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

Args:
    y_test (list of strings): actual labels
    y_pred (list of strings): predicted labels
    groups (list of strings): list of all unique labels
"""
def show_confusion_matrices(y_test, y_pred, groups):
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=groups,
                          title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=groups, normalize=True,
                          title='Normalized confusion matrix')

    plt.show()
    
    
"""
Args:
    df (dataframe)
    labels (list of strings): List of column labels for each column in df
    
Returns:
    dict({string: list of strings}): key is the name of an organ/tissue, value is a sorted list of the top proteins expressed in that organ/tissue by mean abundance
"""
def get_descending_abundances(df, labels):
    labelled_df = df
    labelled_df.columns = labels
    
    label_to_proteins = {} # {label: list of proteins}
    for label in labels:
        sub_df = labelled_df[label]
        sorted_proteins = sub_df.mean(axis=1).sort_values(ascending=False)
        label_to_proteins[label] = sorted_proteins.index.values
        
    return label_to_proteins
        
"""
Args:
    labels_to_proteins (dict {string : list of strings}): output from get_descending_abundances
    label (string): organ/tissue
    n (int): number of proteins to get

Returns:
    list of strings: top n proteins by abundance for the given organ/tissue
"""
def n_most_abundant(labels_to_proteins, label, n):
    
    top_proteins = labels_to_proteins[label][:n] 
    return top_proteins


#########################
#
# Top distinguishing features
#
#########################

"""Transforms a dataframe to keep only the k rows most significant in terms of group-wise ANOVA-F value

Args:
    df (dataframe): rows are proteins/peptides, columns are samples
    labels (list of strings): list of corresponding labels for df columns
    k (int): number of features to keep
    
Returns:
    transformed df with only the k best features kept
"""
def keep_k_best_features(df, labels, k):

    select_k_best_classifier = SelectKBest(k=k)
    kbest = select_k_best_classifier.fit_transform(df[:].T, labels)

    fit_transformed_features = select_k_best_classifier.get_support()

    kbest_df = pd.DataFrame(df, index = df.T.columns[fit_transformed_features])
    return kbest_df

"""Works on Decision Tree, Random Forest, Gradient Boosting

Args:
    df (dataframe)
    clf (classifier): classifier with feature_importances_ attribute
    n (int): number of features to print
    
Returns:
    Prints out top n features (in descending order of importance) and their corresponding coefficient values
"""
def print_top_n_features(df, clf, n):
    
    importances, indices = top_features(df, clf, n)
    
    print("Feature ranking:")

    features = df.index.values.tolist()
    for n in range(n):
        idx = indices[n]
        feature = features[idx]
        print("%s (%f)" % (feature, importances[idx]), end = "")
    
"""
Args:
    df (dataframe)
    clf (classifier): classifier with feature_importances_ attribute
    n (int): number of features to retrieve
    
Returns:
    importances (list of ints): importance values, sorted in descending order
    indices (list of ints): corresponding indices
    
"""
def top_features(df, clf, n):
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1] # Read backwards to get highest values first
    return importances, indices

"""Works on any algorithm with coef_ feature (SVC, Logistic Regression, Naive Bayes)

Args:
    df (dataframe)
    clf (classifier): linear classifier containing coef_ attribute
    n (int): number of features per class to print
    class_labels (list of strings): list of all unique class names (e.g. ['Brain', 'Heart', 'Liver'])
    
Returns:
    Prints out top n features (in descending order of importance) and their corresponding coefficient values for each class
"""
def print_top_n_features_coef(df, clf, n, class_labels):
    
    class_to_indices = top_features_per_class(df, clf, n, class_labels) # {'Lung': [list of indices]}
    
    print('Feature ranking:')
    features = df.index.values.tolist()
    
    for label, indices in class_to_indices.items():
        top_n_indices = indices[:n]
        print('%s: %s' % (label, " ".join(features[i] for i in top_n_indices)))
        print('\n')

"""     
Args:
    df (dataframe)
    clf (classifier): classifier with feature_importances_ attribute
    n (int): number of features to retrieve
    class_labels (list of strings): list of all unique class names (e.g. ['Brain', 'Heart', 'Liver'])
    
Returns:
    dict ({string: list of ints}): keys are class labels, values are lists of indices representing corresponding to features with the highest importances (sorted descending)
"""
def top_features_per_class(df, clf, n, class_labels):
    
    coefs = clf.coef_
    class_to_importances = dict.fromkeys(class_labels) # {'Lung': [3, 10, 5...]}
    
    for i, class_label in enumerate(class_labels):
        sorted_indices = np.argsort(coefs[i])
        class_to_importances[class_label] = sorted_indices

    return class_to_importances
