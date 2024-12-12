using PyCall


py"""
from sklearn.metrics import f1_score

def return_f1(y, eval_y, average="macro"):
    return f1_score(y, eval_y, average=average)
"""
