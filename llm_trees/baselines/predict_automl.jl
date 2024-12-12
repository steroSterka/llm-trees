using PyCall


py"""
import numpy as np
import pandas as pd
from copy import deepcopy

def predict_automl(train_X, train_y, test_X, test_y, feature_names, feature_inds, method):
    # handle conversion issues for missing values between Julia and Python
    X_train = np.array([[np.nan if not np.isscalar(x) else x for x in train_x] for train_x in train_X])
    X_test = np.array([[np.nan if not np.isscalar(x) else x for x in test_x] for test_x in test_X])
    
    # AutoGluon 1.1.1
    if method == "autogluon":
        # load library
        from autogluon.tabular import FeatureMetadata, TabularPredictor

        # preparations for training
        feature_names_autogluon = deepcopy(feature_names)
        feature_names_autogluon.insert(0, "label")
        feature_inds_autogluon = deepcopy(feature_inds)
        feature_inds_autogluon.update({"label": "float"})
        train_data = pd.DataFrame(
            np.hstack((np.array(train_y)[:, np.newaxis], X_train)),
            columns=feature_names_autogluon
        )
        
        classifier = TabularPredictor(
            label="label",
			eval_metric="f1_macro"
        )
        classifier.fit(
            train_data,
            # pass nominal feature indicators
            feature_metadata=FeatureMetadata(feature_inds_autogluon),
            presets=["best_quality"],
            time_limit=3600
        )
        eval_y = classifier.predict(
            pd.DataFrame(X_test, columns=feature_names_autogluon[1:]),
            as_pandas=False
        )
        
    # AutoPrognosis 0.1.21
    elif method == "autoprognosis":
        # load library
        from autoprognosis.studies.classifiers import ClassifierStudy

        # preparations for training
        feature_names_autoprognosis = deepcopy(feature_names)
        feature_names_autoprognosis.insert(0, "target")
        train_data = pd.DataFrame(
            np.hstack((np.array(train_y)[:, np.newaxis], X_train)),
            columns=feature_names_autoprognosis
        )
        
        study = ClassifierStudy(
            train_data,
            target="target",
            num_study_iter=2,
            score_threshold = 0.3,
            metric="f1_score_macro"
        )
        classifier = study.fit()
        eval_y = classifier.predict(X_test)

    # TabPFN 0.1.10
    elif method == "tabpfn":
        # load library
        from tabpfn import TabPFNClassifier

        classifier = TabPFNClassifier(N_ensemble_configurations=32)
        classifier.fit(X_train, train_y)
        eval_y = classifier.predict(
            X_test,
            # pass nominal feature indicators
            categorical_feats=[feature_ind - 1 for feature_ind in feature_inds]
        )
        
    return eval_y
"""
