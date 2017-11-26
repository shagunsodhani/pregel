from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from scipy.special import expit as sigmoid

def compute_auc_score(labels, predictions, mask):
    '''Method to compute AUC score'''
    labels = labels[mask[0][:, 0], mask[0][:, 1]]
    predictions = predictions[mask[0][:,0], mask[0][:,1]]
    return roc_auc_score(labels, predictions)

def compute_average_precision_recall(labels, predictions, mask):
    '''Method to compute the average precision recall score'''
    labels = labels[mask[0][:, 0], mask[0][:, 1]]
    predictions = predictions[mask[0][:,0], mask[0][:,1]]
    return average_precision_score(labels, predictions)