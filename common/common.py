import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from keras import metrics

from .constants import *

def compute_metrics(pred_values:list[float], true_values=None, mode:str=CLF):
    if true_values is None:
        # compute metrics across tasks or simply take the average of a list of metric
        return float(np.mean(pred_values)), float(np.std(pred_values))
    else:
        # compute metrics in a task
        if mode == CLF:
            metric_fn = metrics.Accuracy()
            metric_fn.update_state(y_true=true_values, y_pred=pred_values)
            precision, recall, f1, _ = precision_recall_fscore_support(y_true=true_values, y_pred=pred_values, average='macro')
            return float(metric_fn.result()), float(precision), float(recall), float(f1)
        elif mode == REG:
            metric_fn = metrics.MeanSquaredError()
            metric_fn.update_state(true_values, pred_values)
            return float(metric_fn.result())
