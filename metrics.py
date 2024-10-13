import math
import torch.nn as nn
import torch
import sklearn.metrics as metrics
import torch.nn.functional as F

def get_metrics(y):
    """
        Computes how accurately model learns correct matching of object with the caption in terms of accuracy

        Args:
            y(N,2): Tensor(cpu). the incongruity score of negataive class, positive class.

        Returns:
            predict_label (list): predict results
    """
    predict_label = (y[:,0]<y[:,1]).clone().detach().long().numpy().tolist()
    return predict_label

def get_four_metrics(labels, predicted_labels):
    confusion = metrics.confusion_matrix(labels, predicted_labels)
    total = confusion[0][0] + confusion[0][1] + confusion[1][0] + confusion[1][1]
    acc = (confusion[0][0] + confusion[1][1])/total
    # about sarcasm
    recall = confusion[1][1]/(confusion[1][1]+confusion[1][0])
    precision = confusion[1][1]/(confusion[1][1]+confusion[0][1])
    f1 = 2*recall*precision/(recall+precision)
    return acc,recall,precision,f1

from sklearn.metrics import precision_recall_fscore_support

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def get_four_metrics_macro_f1(y_true, y_pred):
    # preds = np.argmax(y_pred, axis=-1)
    preds = y_pred
    true = y_true
    p_macro, r_macro, f_macro, support_macro \
        = precision_recall_fscore_support(true, preds, average='macro')
    return p_macro, r_macro, f_macro
