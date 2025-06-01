from evo_trees.utils import load_initial_population_from_folder
from gatree.methods.gatreeclassifier import GATreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from llm_trees.config import Config  
import inspect

config = Config()
config.llm = "llama3.3:70b"
config.dataset_name = "iris"
config.task_type = "classification"

# Daten laden
iris = load_iris()
X = iris.data
y = iris.target

# In pandas konvertieren (wichtig für dein GATreeClassifier)
X = pd.DataFrame(X)
y = pd.Series(y)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Initialpopulation laden
initial_population = load_initial_population_from_folder("trees/acl/claude", config, X_train)
print(f"Initialpopulation geladen, Länge: {len(initial_population) if initial_population else 0}")

print(inspect.signature(GATreeClassifier.__init__))

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

print(inspect.signature(GATreeClassifier.__init__))

# Vorhersage und Evaluation
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Test Accuracy:", acc)


# from evo_trees.utils import load_initial_population_from_folder
# from gatree.methods.gatreeclassifier import GATreeClassifier
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from llm_trees.config import Config  
# import inspect


# config = Config()
# config.llm = "llama3.3:70b"
# config.dataset_name = "iris"
# config.task_type = "classification"

# # Daten laden
# iris = load_iris()
# X = iris.data
# y = iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# # Initialpopulation laden
# initial_population = load_initial_population_from_folder("trees/acl/claude", config, X_train)

# print(inspect.signature(GATreeClassifier.__init__))



# clf = GATreeClassifier(
#     max_depth=10,
#     random_state=42
# )

# clf.fit(
#     X_train,
#     y_train,
#     population_size=100,
#     max_iter=40,
#     initial_population=initial_population
# )


# print(inspect.signature(GATreeClassifier.__init__))


# # GATree trainieren
# # clf = GATreeClassifier(
# #     # generations=40,
# #     # population_size=100,
# #     initial_population=initial_population
# # )
# # clf.fit(X_train, y_train)

# # Testen
# y_pred = clf.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# print("Test Accuracy:", acc)
