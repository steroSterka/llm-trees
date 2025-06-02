import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from gatree.methods.gatreeclassifier import GATreeClassifier
from gatree.tree.node import Node
import ast
from gatree.tree.node import Node
import os
import re
from llm_trees.utils import postprocess_prompting_result

def parse_predict_function(predict_fn_str: str) -> Node:
    tree = ast.parse(predict_fn_str)
    print("tree", tree)
    print("tree.body", tree.body)
    func = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "predict"), None)
    if func is None:
        raise ValueError("Keine predict()-Funktion gefunden.")
    return convert_ast_to_node(func.body)

def load_initial_population_from_folder(folder_path, config, X):
    population = []
    stop_loading = False
    for filename in sorted(os.listdir(folder_path)):
        if stop_loading:
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
            print("das ist predict_str", predict_str)
            root_node = parse_predict_function(predict_str)
            print("das ist root_node", root_node)
            population.append(root_node)
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

            # Fall 1: Bedingung mit Vergleich, z.B. nodes[1] <= 0.35
            match = re.match(r"(?:x|nodes)\[(\d+)\] *([><=]=?) *([0-9\.]+)", condition)
            if match:
                att_index = int(match.group(1))
                operator = match.group(2)
                att_value = float(match.group(3))
                node = Node(att_index=att_index, att_value=att_value)
                node.set_left(true_branch)
                node.set_right(false_branch)
                return node

            # Fall 2: Bedingung nur wie nodes[1], interpretieren als bool (True/False)
            match_bool = re.match(r"(?:x|nodes)\[(\d+)\]", condition)
            if match_bool:
                att_index = int(match_bool.group(1))
                # Da kein Schwellenwert, können wir z.B. att_value=0.5 nehmen als Dummy
                node = Node(att_index=att_index, att_value=0.5)
                node.set_left(true_branch)
                node.set_right(false_branch)
                return node

            raise ValueError(f"Konnte Attribut-Index und Wert aus Bedingung nicht extrahieren: {condition}")

        elif isinstance(stmt, ast.Assign):
            # Zuweisung einer Vorhersage
            if isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "prediction":
                value = stmt.value
                prediction = int(ast.literal_eval(value))
                return Node(att_index=-1, att_value=prediction)

        elif isinstance(stmt, ast.Return):
            continue  # Ignorieren

    raise ValueError("Kein gültiger Vorhersage-Zweig gefunden.")
