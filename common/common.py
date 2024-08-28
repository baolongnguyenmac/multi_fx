import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from keras import metrics

def compute_metrics(pred_values:list[float], true_values=None):
    acc_fn = metrics.Accuracy()
    if true_values is None:
        # compute metrics across tasks or simply take the average of a list of metric
        return float(np.mean(pred_values)), float(np.std(pred_values))
    else:
        # compute metrics in a task
        acc_fn.update_state(true_values, pred_values)
        precision, recall, f1, _ = precision_recall_fscore_support(true_values, pred_values, average='macro')
        return float(acc_fn.result()), float(precision), float(recall), float(f1)