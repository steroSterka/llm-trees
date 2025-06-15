from evo_trees.utils import load_initial_population_from_folder
from gatree.methods.gatreeclassifier import GATreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from llm_trees.config import Config  
import os



config = Config()
config.method = "llama3.3:70b"
config.dataset = "irish"
config.task_type = "classification"

#change this 
data_sets = "heart_h"


# all available datasets
all_datasets = [
    "bankruptcy",
    "boxing1",
    "boxing2",
    "creditscore",
    "japansolvent",
    "colic",
    "heart_h",
    "hepatitis",
    "house_votes_84",
    "irish",
    "labor",
    "penguins",
    "vote",
]


# Daten laden
path = os.path.join(".", f"data_sets/{data_sets}")
X1 = pd.read_csv(os.path.join(path, "X.csv"))
Y1 = pd.read_csv(os.path.join(path, "y.csv"))["target"]



X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.33, random_state=42)

# Initialpopulation laden
initial_population = load_initial_population_from_folder(f"trees/{data_sets}/claude", config, X_train)
print(f"Initialpopulation geladen, Länge: {len(initial_population) if initial_population else 0}")
pop_size = len(initial_population) if initial_population is not None else 150


clf = GATreeClassifier(
    max_depth=10,
    random_state=42
)

clf.fit(
    X_train,
    y_train,
    population_size=pop_size,
    max_iter=40,
    initial_population=initial_population
)


# Vorhersage und Evaluation
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)
 



# def train_on_dataset(data_set_name):
#     config = Config()
#     config.method = "llama3.3:70b"
#     config.dataset = data_set_name
#     config.task_type = "classification"

#     # Daten laden
#     path = os.path.join("data_sets", data_set_name)
#     X1 = pd.read_csv(os.path.join(path, "X.csv"))
#     Y1 = pd.read_csv(os.path.join(path, "y.csv"))["target"]

#     X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.33, random_state=42)

#     initial_population = load_initial_population_from_folder(f"trees/{data_set_name}/claude", config, X_train)
#     print(f"Initialpopulation geladen, Länge: {len(initial_population) if initial_population else 0}")
#     pop_size = len(initial_population) if initial_population else 150

#     clf = GATreeClassifier(max_depth=10, random_state=42)
#     clf.fit(X_train, y_train, population_size=pop_size, max_iter=40, initial_population=initial_population)

#     y_pred = clf.predict(X_test)
#     acc = accuracy_score(y_test, y_pred)
#     print(f"Accuracy for {data_set_name}: {acc}")
#     return acc
