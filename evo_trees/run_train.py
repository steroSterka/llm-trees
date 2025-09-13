from evo_trees.utils import load_initial_population_from_folder
from gatree.methods.gatreeclassifier import GATreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from llm_trees.config import Config  
import os
import re
import numpy as np
from sklearn.metrics import accuracy_score


# folder_path = "/Users/stervan/Documents/Projects/llm-trees/trees/hepatitis/gemma3:27b/"
# # IDs mit Fehlern (supported expression / malformed / left side issues)
# iter_list = [
#     "1","106","108","110","123","127","12","131","134","138","13","140","141","145","146","148","149",
#     "150","153","156","164","166","167","168","169","171","172","173","175","178","180","184","190",
#     "193","194","205","208","212","214","216","217","222","223","230","231","232","233","234"
# ]




# for file_name in os.listdir(folder_path):
#     file_path = os.path.join(folder_path, file_name)
    
#     if os.path.isfile(file_path):
#         # Suche genau die Zahl nach 'iter_' bis zum nächsten '_'
#         match = re.search(r'iter_(\d+)', file_name)  # nur Ziffern
#         if match:
#             iter_number = match.group(1)
#             if iter_number in iter_list:
#                 try:
#                     os.remove(file_path)
#                     print(f"Gelöscht: {file_name}")
#                 except Exception as e:
#                     print(f"Fehler beim Löschen {file_name}: {e}")



config = Config()
config.method = "llama3.1:70b"
config.dataset = "creditscore"


config.root = "." 
config.tree_file = ""          
config.dataset_name = "creditscore"    
config.llm = "llama3.1:70b"           
config.task_type = "classification" 
config.temperature = 1  
config.iter = 0 
config.num_iters = 5 
config.train_split = 0.67
config.classifier = "mlp" 
config.append_raw_features = True 
config.force_decision_tree = True  
config.include_description = False  
config.llm_dialogue = True  
config.max_tree_depth = 2
config.num_examples = 1  
config.num_retry_llm = 10 
config.use_role_prompt = False  
config.num_trees = 1
config.seed = 42
config.generate_tree_if_missing = True
config.regenerating_invalid_trees = True
config.skip_existing = True





# load data
path = os.path.join(".", f"data_sets/{config.dataset}")
X1 = pd.read_csv(os.path.join(path, "X.csv"))
Y1 = pd.read_csv(os.path.join(path, "y.csv"))["target"]



X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.67, random_state=45)

# load initialpopulation
initial_population = load_initial_population_from_folder(f"trees/{config.dataset}/{config.method}", config, X_train)
print(f"Initialpopulation geladen, Länge: {len(initial_population) if initial_population else 0}")
pop_size = len(initial_population) if initial_population is not None else 150



clf = GATreeClassifier(
    max_depth=2,
    random_state=42
)

clf.fit(
    X_train,
    y_train,
    population_size=pop_size,
    max_iter=100,
    initial_population=initial_population
)


# prediction and evaluation
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("After optimization:", acc)