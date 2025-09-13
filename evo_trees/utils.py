import pandas as pd
from gatree.methods.gatreeclassifier import GATreeClassifier
from gatree.tree.node import Node
import ast
import os
import re
from llm_trees.utils import postprocess_prompting_result
from llm_trees.utils import generate_tree
import time
import ast
import re
from gatree.tree.node import Node

def parse_predict_function(predict_fn_str: str, feature_to_index: dict) -> Node:
    tree = ast.parse(predict_fn_str)
    func = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "predict"), None)
    if func is None:
        raise ValueError("Keine predict()-Funktion gefunden.")
    return convert_ast_to_node(func.body, feature_to_index)

def _extract_nodes_index(sub: ast.Subscript) -> int:
    sl = sub.slice
    if isinstance(sl, ast.Constant):
        return int(sl.value)
    if hasattr(ast, "Index") and isinstance(sl, ast.Index):
        v = sl.value
        if isinstance(v, ast.Constant):
            return int(v.value)
        return int(ast.literal_eval(v))
    return int(ast.literal_eval(sl))

def _num_from_ast(node) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return -float(node.operand.value)
    return float(ast.literal_eval(node))

def _feature_name_from_subscript(sub: ast.Subscript) -> str:
    if not (isinstance(sub.value, ast.Name) and sub.value.id == "X"):
        raise ValueError("Left side is not X 'X[...]'.")
    sl = sub.slice
    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
        return sl.value
    if hasattr(ast, "Index") and isinstance(sl, ast.Index):
        v = sl.value
        if isinstance(v, ast.Constant) and isinstance(v.value, str):
            return v.value
    s = ast.unparse(sub)
    m = re.search(r'X\[(?P<q>["\'])(.+?)(?P=q)\]', s)
    if not m:
        raise ValueError(f"could not extract featurename from: {s}")
    return m.group(2)

def _parse_compare_to_threshold(compare_node: ast.Compare, feature_to_index: dict):
    if not isinstance(compare_node, ast.Compare):
        raise ValueError("The condition is not a comparison expression.")
    if len(compare_node.ops) != 1 or len(compare_node.comparators) != 1:
        raise ValueError("Nur einfache Vergleiche mit einem Operator werden unterstützt.")
    left = compare_node.left
    op = compare_node.ops[0]
    right = compare_node.comparators[0]

    if not isinstance(left, ast.Subscript):
        raise ValueError("Left side must be x['feature']")
    feature_name = _feature_name_from_subscript(left)
    att_index = feature_to_index[feature_name]
    threshold = _num_from_ast(right)

    # We support operators, but GATree nodes internally check: go left if value > att_value else right
    # We ALWAYS represent the node as a threshold 'att_value = threshold'.
    # Child assignment:
    # - For condition (X <= t): true_branch goes to the right, false_branch to the left.
    # - For (X > t): true_branch goes to the left, false_branch to the right.
    if isinstance(op, (ast.LtE, ast.Lt)):
        cmp_type = "<=" if isinstance(op, ast.LtE) else "<"
        direction = "le"  # less-or-equal / less
    elif isinstance(op, (ast.GtE, ast.Gt)):
        cmp_type = ">=" if isinstance(op, ast.GtE) else ">"
        direction = "gt"  # greater-or-equal / greater
    else:
        raise ValueError(f"Not supported expression: {type(op).__name__}")

    return att_index, float(threshold), direction

def convert_ast_to_node(body: list, feature_to_index: dict, node_conditions=None) -> Node:
    if node_conditions is None:
        node_conditions = {}

    for stmt in body:
        if isinstance(stmt, ast.Assign):
            tgt = stmt.targets[0]
            if isinstance(tgt, ast.Subscript) and isinstance(tgt.value, ast.Name) and tgt.value.id == "nodes":
                try:
                    idx = _extract_nodes_index(tgt)
                    node_conditions[idx] = stmt.value 
                except Exception:
                    pass

    for stmt in body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "prediction":
            prediction = int(_num_from_ast(stmt.value))
            return Node(att_index=-1, att_value=prediction)

        if isinstance(stmt, ast.If):
            true_branch = convert_ast_to_node(stmt.body, feature_to_index, node_conditions)
            false_branch = convert_ast_to_node(stmt.orelse, feature_to_index, node_conditions)

            test = stmt.test
            if isinstance(test, ast.Subscript) and isinstance(test.value, ast.Name) and test.value.id in ("nodes", "x", "X"):
                try:
                    k = _extract_nodes_index(test)
                except Exception:
                    raise ValueError(f"Could not extract index from condition: {ast.unparse(test)}")
                comp = node_conditions.get(k)
                if comp is None:
                    raise ValueError(f"No stored condition found for nodes[{k}]")
                att_index, att_value, direction = _parse_compare_to_threshold(comp, feature_to_index)
                node = Node(att_index=att_index, att_value=att_value)
                if direction == "le":
                    node.set_right(true_branch)
                    node.set_left(false_branch)
                else:
                    node.set_left(true_branch)
                    node.set_right(false_branch)
                return node

            if isinstance(test, ast.Compare):
                att_index, att_value, direction = _parse_compare_to_threshold(test, feature_to_index)
                node = Node(att_index=att_index, att_value=att_value)
                if direction == "le":
                    node.set_right(true_branch)
                    node.set_left(false_branch)
                else:
                    node.set_left(true_branch)
                    node.set_right(false_branch)
                return node

    raise ValueError("No valid prediction branch found.")


def load_initial_population_from_folder(folder_path, config, X):
    population = []
    feature_to_index = {name: idx for idx, name in enumerate(X.columns)}
    for filename in sorted(os.listdir(folder_path)):
        if not filename.endswith(".txt"):
            continue
        config.tree_file = filename
        try:
            with open(os.path.join(folder_path, filename), "r") as file:
                content = file.read()
            predict_str = postprocess_prompting_result(config, content)
            root_node = parse_predict_function(predict_str, feature_to_index)
            population.append(root_node)
        except Exception as e:
            print(f"Error while parsing of {filename}: {e}")
    print(f"Population loaded, number of trees: {len(population)}")
    return population





# def parse_predict_function(predict_fn_str: str, feature_to_index: dict) -> Node:
#     tree = ast.parse(predict_fn_str)
#     func = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "predict"), None)
#     if func is None:
#         raise ValueError("Keine predict()-Funktion gefunden.")
#     return convert_ast_to_node(func.body, feature_to_index)


# def load_initial_population_from_folder(folder_path, config, X):
#     population = []
#     for filename in sorted(os.listdir(folder_path)):
#         if not filename.endswith(".txt"):
#             continue

#         config.tree_file = filename  
#         try:
#             with open(os.path.join(folder_path, filename), "r") as file:
#                 content = file.read()
#             predict_str = postprocess_prompting_result(config, content)
#             # root_node = parse_predict_function(predict_str)
#             feature_to_index = {name: idx for idx, name in enumerate(X.columns)}
#             root_node = parse_predict_function(predict_str, feature_to_index)
#             population.append(root_node)
#         except Exception as e:
#             print(f"Fehler beim Parsen von {filename}: {e}")
#     print(f"Population geladen, Anzahl Bäume: {len(population)}")
#     return population


# def convert_ast_to_node(body: list, feature_to_index: dict, node_conditions=None) -> Node:
#     if node_conditions is None:
#         node_conditions = {}

#     # 1. Erst alle Zuweisungen zu nodes[i] im aktuellen Body sammeln
#     for stmt in body:
#         if isinstance(stmt, ast.Assign):
#             if (isinstance(stmt.targets[0], ast.Subscript) and
#                 isinstance(stmt.targets[0].value, ast.Name) and
#                 stmt.targets[0].value.id == "nodes"):
#                 index = int(ast.literal_eval(stmt.targets[0].slice))
#                 node_conditions[index] = ast.unparse(stmt.value)

#     # 2. Dann den Body erneut durchgehen und Knoten bauen
#     for stmt in body:
#         # Leaf-Knoten
#         if isinstance(stmt, ast.Assign):
#             if (isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "prediction"):
#                 prediction = int(ast.literal_eval(stmt.value))
#                 return Node(att_index=-1, att_value=prediction)

#         # Entscheidungs-Knoten
#         elif isinstance(stmt, ast.If):
#             condition = ast.unparse(stmt.test)
#             true_branch = convert_ast_to_node(stmt.body, feature_to_index, node_conditions)
#             false_branch = convert_ast_to_node(stmt.orelse, feature_to_index, node_conditions)

#             match_bool = re.match(r"(?:x|nodes)\[(\d+)\]", condition)
#             if match_bool:
#                 node_idx = int(match_bool.group(1))
#                 original_condition = node_conditions.get(node_idx)
#                 if original_condition:
#                     match = re.match(r'X\["(.+?)"\]*([><=]=?) *(-?[0-9\.]+)', original_condition)
#                     if match:
#                         feature_name = match.group(1)
#                         att_index = feature_to_index[feature_name]
#                         att_value = float(match.group(3))
#                         node = Node(att_index=att_index, att_value=att_value)
#                         node.set_left(true_branch)
#                         node.set_right(false_branch)
#                         return node

#             raise ValueError(f"Konnte Bedingung nicht rekonstruieren: {condition}")

#     raise ValueError("Kein gültiger Vorhersage-Zweig gefunden.")