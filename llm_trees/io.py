import os
import warnings

import pandas as pd
from cachetools import TTLCache
from cachetools import cached
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split

from .config import Config

warnings.simplefilter(action='ignore', category=FutureWarning)
imputing_neighbors = 10


def get_feature_types(dataset_name, X):

    # Default: no separate imputing for numeric and categorical features
    numerical_features = X.keys().to_list()
    categorical_features = []

    if dataset_name == "heart_h":
        numerical_features = [
            'ca',
            'chol',
            'thalach',
            'trestbps',
        ]
        categorical_features = [col for col in X.columns if col not in numerical_features]

    elif dataset_name == "acl":
        numerical_features = [
            'Group',
            'Sex',
            'Dominant_Leg',
        ]
        categorical_features = [col for col in X.columns if col not in numerical_features]

    elif dataset_name == "posttrauma":
        categorical_features = [
            'gender_birth',
            'ethnic_group',
            'education_age',
            'working_at_baseline',
            'smoker',
            'iss_category',
            'penetrating_injury',
        ]
        numerical_features = [col for col in X.columns if col not in categorical_features]

    elif dataset_name == "penguins":
        numerical_features = [
            'culmen_length_mm',
            'culmen_depth_mm',
            'flipper_length_mm',
            'body_mass_g',
        ]
        categorical_features = [col for col in X.columns if col not in numerical_features]


    elif dataset_name == "labor":
        numerical_features = [
            'duration',
            'shift differential',
            'standby pay',
            'statutory holidays',
            'wage increase first year',
            'wage increase second year',
            'wage increase third year',
            'working hours',
        ]
        categorical_features = [col for col in X.columns if col not in numerical_features]

    elif dataset_name == "irish":
        numerical_features = ['Prestige_score']
        categorical_features = [col for col in X.columns if col not in numerical_features]

    elif dataset_name in ["vote", "house_votes_84"]:
        categorical_features = X.keys().to_list()
        numerical_features = []

    elif dataset_name == "hepatitis":
        categorical_features = [
            'ANOREXIA',
            'ANTIVIRALS',
            'ASCITES',
            'FATIGUE',
            'HISTOLOGY',
            'LIVER BIG',
            'LIVER FIRM',
            'MALAISE',
            'SEX',
            'SPIDERS',
            'SPLEEN PALPABLE',
            'STEROID',
            'VARICES',
        ]
        numerical_features = [
            'AGE',
            'ALBUMIN',
            'ALK PHOSPHATE',
            'BILIRUBIN',
            'PROTIME',
            'SGOT',
        ]

    elif dataset_name == "colic":
        numerical_features = [
            'abdomcentesis_total_protein',
            'nasogastric_reflux_PH',
            'packed_cell_volume',
            'pulse',
            'rectal_temperature',
            'respiratory_rate',
            'total_protein',
        ]
        categorical_features = [col for col in X.columns if col not in numerical_features]

    elif dataset_name == "analcatdata_japansolvent":
        categorical_features = ["Firm"]
        numerical_features = [col for col in X.columns if col not in categorical_features]

    elif dataset_name == "analcatdata_creditscore":
        categorical_features = ["Self.employed", "Own.home"]
        numerical_features = [col for col in X.columns if col not in categorical_features]

    elif dataset_name == "analcatdata_bankruptcy":
        categorical_features = ["Company"]
        numerical_features = [col for col in X.columns if col not in categorical_features]

    elif dataset_name in ["analcatdata_boxing1", "analcatdata_boxing2"]:
        numerical_features = []
        categorical_features = X.keys().to_list()

    return numerical_features, categorical_features


def load_and_split_data(config: Config):

    X, y = get_data(config.root, config.dataset)
    data_available = X is not None and y is not None
    if not data_available:
        raise FileNotFoundError(f"Data not found for dataset: {config.dataset}")

    # Init train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=config.train_split, random_state=config.seed
    )

    numerical_features, categorical_features = get_feature_types(config.dataset, X)
    X_train, X_test = impute_data(X_train, X_test, numerical_features, categorical_features)

    return X_train, X_test, y_train, y_test

@cached(cache=TTLCache(maxsize=1024, ttl=86400))
def get_data(root: str, dataset_name: str):

    try:
        if dataset_name == "acl":

            # the private ACL data is not included in the repository
            X = pd.read_csv(os.path.join(root, "data_sets/acl/X.csv"))
            y = pd.read_csv(os.path.join(root, "data_sets/acl/y.csv"))

            X["Group"] = X["Group"].replace("recon", 2)
            X["Group"] = X["Group"].replace("nc", 1)
            X["Group"] = X["Group"].replace("c", 0)

            X["Sex"] = X["Sex"].replace("female", 0)
            X["Sex"] = X["Sex"].replace("male", 1)

            X["Dominant_Leg"] = X["Dominant_Leg"].replace("left", 0)
            X["Dominant_Leg"] = X["Dominant_Leg"].replace("right", 1)

            # rename keys to match the other datasets
            X = X.rename(columns={
                " ccMF.D.T2.Me": "ccMF.D.T2.Me",
                " ccMF.S.T2.Me": "ccMF.S.T2.Me",
                "  Age": "Age",
            })

            # Mapping [-1  1] to [0 1] for the XGBoost model
            y = y > 0

        else:
            path = os.path.join(root, f"data_sets/{dataset_name}")
            X = pd.read_csv(os.path.join(path, "X.csv"))
            y = pd.read_csv(os.path.join(path, "y.csv"))

        return X, y

    except FileNotFoundError:
        return None, None


def impute_data(X_train, X_test, numerical_features=None, categorical_features=None):

    if numerical_features is not None:

        if numerical_features:
            mean_imputer = KNNImputer(n_neighbors=imputing_neighbors, keep_empty_features=True)
            X_train[numerical_features] = mean_imputer.fit_transform(X_train[numerical_features])
            if X_test is not None:
                X_test[numerical_features] = mean_imputer.transform(X_test[numerical_features])

        if categorical_features:
            mode_imputer = SimpleImputer(strategy='most_frequent', keep_empty_features=True)
            X_train[categorical_features] = mode_imputer.fit_transform(X_train[categorical_features])
            if X_test is not None:
                X_test[categorical_features] = mode_imputer.transform(X_test[categorical_features])

        # cast categorical features to int
        for col in categorical_features:
            X_train[col] = X_train[col].astype(int)
            if X_test is not None:
                X_test[col] = X_test[col].astype(int)

    else:
        # imputer = KNNImputer(n_neighbors=n_neighbors)
        imputer = SimpleImputer(strategy='most_frequent')
        X_train = imputer.fit_transform(X_train)
        if X_test is not None:
            X_test = imputer.transform(X_test)

    return X_train, X_test
