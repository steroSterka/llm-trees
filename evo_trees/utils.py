import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gatree.methods.gatreeclassifier import GATreeClassifier
from gatree.tree.node import Node
import ast
from gatree.tree.node import Node
import os
from gatree.methods.gatreeclassifier import GATreeClassifier
import re
from llm_trees.utils import postprocess_prompting_result

def parse_predict_function(predict_fn_str: str) -> Node:
    tree = ast.parse(predict_fn_str)
    func = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "predict"), None)
    print("this is the func", func)
    if func is None:
        raise ValueError("Keine predict()-Funktion gefunden.")
    return convert_ast_to_node(func.body)

def load_initial_population_from_folder(folder_path, config, X):
    population = []
    stop_loading = False
    for filename in sorted(os.listdir(folder_path)):
        if stop_loading:
            # print(f"Überspringe Datei {filename} (nach Stop-Datei).")
            continue

        if filename == "acl_claude_examples_1_desc_False_temp_100_maxdepth_3_iter_3_dialogue.txt":
            print(f"Stoppe Laden bei Datei {filename}.")
            stop_loading = True
            continue  

        if not filename.endswith(".txt"):
            continue

        config.tree_file = filename  
        try:
            with open(os.path.join(folder_path, filename), "r") as file:
                content = file.read()
            predict_str = postprocess_prompting_result(config, content)
            root_node = parse_predict_function(predict_str)
            population.append(root_node)
            print(f"predict_str {predict_str}")
            print(f"Root Node: {root_node}")
            print(f"population: {population}")
            print(f"Datei {filename} erfolgreich geladen.")
        except Exception as e:
            print(f"Fehler beim Parsen von {filename}: {e}")
    print(f"Population geladen, Anzahl Bäume: {len(population)}")
    return population


def convert_ast_to_node(body: list) -> Node:
    for stmt in body:
        if isinstance(stmt, ast.If):
            condition = ast.unparse(stmt.test)
            true_branch = convert_ast_to_node(stmt.body)
            false_branch = convert_ast_to_node(stmt.orelse)
            return Node(condition=condition, true_branch=true_branch, false_branch=false_branch)
        elif isinstance(stmt, ast.Assign):
            if isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "prediction":
                if isinstance(stmt.value, ast.Constant):
                    label = stmt.value.value
                else:
                    try:
                        label = int(ast.unparse(stmt.value))
                    except Exception:
                        raise ValueError("Unbekannter Label-Typ.")
                return Node(prediction=label)

    raise ValueError("Kein gültiger Ast-Zweig oder Vorhersage gefunden.")



# def convert_ast_to_node(body: list) -> Node:
#     """
#     Rekursive Umwandlung einer Liste von AST-Nodes in eine Baumstruktur.
#     """
#     for stmt in body:
#         if isinstance(stmt, ast.If):
#             condition = ast.unparse(stmt.test)
#             true_branch = convert_ast_to_node(stmt.body)
#             false_branch = convert_ast_to_node(stmt.orelse)
#             return Node(condition=condition, true_branch=true_branch, false_branch=false_branch)
#         elif isinstance(stmt, ast.Assign):
#             if isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "prediction":
#                 label = stmt.value.value if isinstance(stmt.value, ast.Constant) else int(ast.unparse(stmt.value))
#                 return Node(prediction=label)
#     raise ValueError("Kein gültiger Ast-Zweig oder Vorhersage gefunden.")




# def load_initial_population_from_folder(folder_path, config, X):
#     population = []
#     for filename in os.listdir(folder_path):
#         if filename == "acl_claude_examples_1_desc_False_temp_100_maxdepth_3_iter_3_dialogue.txt":
#             break  
#         if filename.endswith(".txt"):
#             config.tree_file = filename  
#             try:
#                 with open(os.path.join(folder_path, filename), "r") as file:
#                     content = file.read()
#                 predict_str = postprocess_prompting_result(config, content)
#                 root_node = parse_predict_function(predict_str)
#                 population.append(root_node)
#             except Exception as e:
#                 print(f"Fehler beim Parsen von {filename}: {e}")
#                 break 
#     return population



# def load_initial_population_from_folder(folder_path, config, X):
#     population = []
#     for filename in os.listdir(folder_path):
#         if filename.endswith(".txt"):
#             config.tree_file = filename  
#             try:
#                 # predict-Funktion aus Datei extrahieren
#                 with open(os.path.join(folder_path, filename), "r") as file:
#                     content = file.read()
#                 predict_str = postprocess_prompting_result(config, content)

#                 # In Node-Struktur umwandeln
#                 root_node = parse_predict_function(predict_str)
#                 population.append(root_node)
#             except Exception as e:
#                 print(f"Fehler beim Parsen von {filename}: {e}")
#     return population
