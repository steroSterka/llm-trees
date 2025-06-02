from evo_trees.utils import load_initial_population_from_folder
from gatree.methods.gatreeclassifier import GATreeClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from llm_trees.config import Config  
import inspect

config = Config()
config.method = "llama3.3:70b"
config.dataset = "iris"
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
 