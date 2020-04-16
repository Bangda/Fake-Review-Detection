#################################
## credit Geller for this code ##
# https://towardsdatascience.com/normalization-vs-standardization-quantitative-analysis-a91e8a79cebf
#################################

from __future__ import print_function

import itertools
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier



def print_results(names, results, test_scores):
    print()
    print("#" * 30 + "Results" + "#" * 30)
    counter = 0

    class Color:
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        DARKCYAN = '\033[36m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'

    # Get max row
    clf_names = set([name.split("_")[1] for name in names])
    max_mean = {name: 0 for name in clf_names}
    max_mean_counter = {name: 0 for name in clf_names}
    for name, result in zip(names, results):
        counter += 1
        clf_name = name.split("_")[1]
        if result.mean() > max_mean[clf_name]:
            max_mean_counter[clf_name] = counter
            max_mean[clf_name] = result.mean()

    # print max row in BOLD
    counter = 0
    prev_clf_name = names[0].split("_")[1]
    for name, result, score in zip(names, results, test_scores):
        counter += 1
        clf_name = name.split("_")[1]
        if prev_clf_name != clf_name:
            print()
            prev_clf_name = clf_name
        msg = "%s: %f (%f) [test_score:%.3f]" % (name, result.mean(), result.std(), score)
        if counter == max_mean_counter[clf_name]:
            print(Color.BOLD + msg)
        else:
            print(Color.END + msg)


def create_pipelines(seed, verbose=1):
    """
         Creates a list of pipelines with preprocessing(PCA), models and scalers.
    :param seed: Random seed for models who needs it
    :return:
    """

    models = [
              ('LR', LogisticRegression()),
              ('LDA', LinearDiscriminantAnalysis()),
              ('DT', DecisionTreeClassifier(random_state=seed)),
              ('RF', RandomForestClassifier(max_depth=3, random_state=seed)),
              ('XGB', xgb.XGBClassifier(random_state=seed)),
              ('AdA', AdaBoostClassifier(n_estimators=100, random_state=seed))

              ]
    scalers = [('StandardScaler', StandardScaler()),
               ('MinMaxScaler', MinMaxScaler()),
               ('MaxAbsScaler', MaxAbsScaler()),
               ('QuantileTransformer-Normal', QuantileTransformer(output_distribution='normal')),
               ('PowerTransformer-Yeo-Johnson', PowerTransformer(method='yeo-johnson')),
               ('Normalizer', Normalizer())
               ]

    # Create pipelines
    pipelines = []
    for model in models:
        # Append only model
        model_name = "_" + model[0]
        pipelines.append((model_name, Pipeline([model])))

        # Append model+scaler
        for scalar in scalers:
            model_name = scalar[0] + "_" + model[0]
            pipelines.append((model_name, Pipeline([scalar, model])))

    if verbose:
        print("Created these pipelines:")
        for pipe in pipelines:
            print(pipe[0])

    return pipelines


def run_cv_and_test(X_train, y_train, X_test, y_test, pipelines, scoring, seed, num_folds,
                    dataset_name, n_jobs):
    """
        Iterate over the pipelines, calculate CV mean and std scores, fit on train and predict on test.
        Return the results in a dataframe
    """

    # List that contains the rows for a dataframe
    rows_list = []

    # Lists for the pipeline results
    results = []
    names = []
    test_scores = []
    prev_clf_name = pipelines[0][0].split("_")[1]
    print("First name is : ", prev_clf_name)

    for name, model in pipelines:
        kfold = model_selection.KFold(n_splits=num_folds, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, n_jobs=n_jobs, scoring=scoring)
        results.append(cv_results)
        names.append(name)

        # Print CV results of the best CV classier
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

        # fit on train and predict on test
        model.fit(X_train, y_train)
        if scoring == "accuracy":
            curr_test_score = model.score(X_test, y_test)
        elif scoring == "roc_auc":
            y_pred = model.predict_proba(X_test)[:, 1]
            curr_test_score = roc_auc_score(y_test, y_pred)

        test_scores.append(curr_test_score)

        # Add separation line if different classifier applied
        rows_list, prev_clf_name = check_seperation_line(name, prev_clf_name, rows_list)

        # Add for final dataframe
        results_dict = {"Dataset": dataset_name,
                        "Classifier_Name": name,
                        "CV_mean": cv_results.mean(),
                        "CV_std": cv_results.std(),
                        "Test_score": curr_test_score
                        }
        rows_list.append(results_dict)

    print_results(names, results, test_scores)

    df = pd.DataFrame(rows_list)
    return df[["Dataset", "Classifier_Name", "CV_mean", "CV_std", "Test_score"]]


def run_cv_and_test_hypertuned_params(X_train, y_train, X_test, y_test, pipelines, scoring, seed, num_folds,
                                      dataset_name, hypertuned_params, n_jobs):
    """
        Iterate over the pipelines, calculate CV mean and std scores, fit on train and predict on test.
        Return the results in a dataframe
    :param X_train:
    :param y_train:
    :param X_test:
    :param y_test:
    :param scoring:
    :param seed:
    :param num_folds:
    :param dataset_name:
    :return:
    """

    # List that contains the rows for a dataframe
    rows_list = []

    # Lists for the pipeline results
    results = []
    names = []
    test_scores = []
    prev_clf_name = pipelines[0][0].split("_")[1]
    print("First name is : ", prev_clf_name)

    # To be used within GridSearch (5 in your case)
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    # To be used in outer CV (you asked for num_folds)
    outer_cv = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    for name, model in pipelines:

        # Get model's hyper parameters
        model_name = name.split("_")[1]
        if "-" in model_name:
            model_name = model_name.split("-")[0]

        if model_name in hypertuned_params.keys():
            random_grid = hypertuned_params[model_name]
        else:
            continue

        # Train nested-CV
        clf = GridSearchCV(estimator=model, param_grid=random_grid, cv=inner_cv, scoring=scoring,
                           verbose=2, n_jobs=n_jobs, refit=True)
        cv_results = model_selection.cross_val_score(clf, X_train, y_train, cv=outer_cv, n_jobs=n_jobs, scoring=scoring)
        results.append(cv_results)
        names.append(name)

        # Print CV results of the best CV classier
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

        # fit on train and predict on test
        model.fit(X_train, y_train)
        if scoring is "accuracy":
            curr_test_score = model.score(X_test, y_test)
        elif scoring is "roc_auc":
            y_pred = model.predict(X_test)
            curr_test_score = accuracy_score(y_test, y_pred)

        test_scores.append(curr_test_score)

        # Add separation line if different classifier applied
        rows_list, prev_clf_name = check_seperation_line(name, prev_clf_name, rows_list)

        # Add for final dataframe
        results_dict = {"Dataset": dataset_name,
                        "Classifier_Name": name,
                        "CV_mean": cv_results.mean(),
                        "CV_std": cv_results.std(),
                        "Test_score": curr_test_score
                        }
        rows_list.append(results_dict)

    print_results(names, results, test_scores)

    df = pd.DataFrame(rows_list)
    return df[["Dataset", "Classifier_Name", "CV_mean", "CV_std", "Test_score"]]


def check_seperation_line(name, prev_clf_name, rows_list):
    """
        Add empty row if different classifier ending
    """

    clf_name = name.split("_")[1]
    if prev_clf_name != clf_name:
        empty_dict = {"Dataset": "",
                      "Classifier_Name": "",
                      "CV_mean": "",
                      "CV_std": "",
                      "Test_acc": ""
                      }
        rows_list.append(empty_dict)
        prev_clf_name = clf_name
    return rows_list, prev_clf_name
