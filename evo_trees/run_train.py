from evo_trees.utils import load_initial_population_from_folder
from gatree.methods.gatreeclassifier import GATreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from llm_trees.config import Config  
import os



config = Config()
config.method = "llama3.3:70b"
config.dataset = "bankruptcy"


config.root = "." 
config.tree_file = ""          
config.dataset_name = "bankruptcy"    
config.llm = "llama3.3:70b"           
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



X_train, X_test, y_train, y_test = train_test_split(X1, Y1, test_size=0.9, random_state=45)

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
print("Test Accuracy:", acc)
 