import os

import numpy as np

from .io import load_and_split_data
from .metrics import score_acc, score_f1
from .utils import get_tree_path, predict_tree_from_file, generate_tree, regenerate_tree

def eval_induction(config):

    try:

        X_train, X_test, y_train, y_test = load_and_split_data(config)

        # Generate trees if necessary
        generate_tree(config)

        # return 42, 42
        unique_labels = np.unique(y_test)
        y_pred = None
        for try_idx in range(config.num_retry_llm + 1):
            if try_idx == config.num_retry_llm:
                raise ValueError(f"Reached maximum number of retries for tree: {os.path.basename(get_tree_path(config))}")

            try:
                y_pred, _ = predict_tree_from_file(config, X_test)
                if any([label not in unique_labels for label in np.unique(y_pred)]):
                    raise ValueError(f"Invalid labels in prediction: {np.unique(y_pred)}")
                break
            except Exception as e:
                # y_pred, _ = predict_tree_from_file(config, X_test)
                regenerate_tree(config, e)

        if y_pred is None:
            raise ValueError(f"Prediction failed for tree: {get_tree_path(config)}")

        # calculate balanced accuracy and F1 score
        acc = score_acc(y_test, y_pred)
        f1 = score_f1(y_test, y_pred)

    except Exception as e:
        print(f"Scoring failed: {e}")
        acc, f1 = -1, -1

    return acc, f1
