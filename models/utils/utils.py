import numpy as np

def thresholding(prediction, threshold):
    """Apply threshold on probability prediction"""
    out = [[0] if value < threshold else [1] for value in prediction]
    return np.array(out)

def F1_accuracy(prediction, true):
    """
    Parameters
    ----------
    prediction: ndarray of shape (#samples, 1)
        array of 0/1 prediction (of whether the ad is clicked)
    true: ndarray of shape (#samples, 1)
        ground truth array

    Returns
    -------
    F1 score and accuracy
    """
    hits_arr = np.logical_not(np.logical_xor(prediction, true))
    hits = np.count_nonzero(hits_arr) # the label correctly being predicted
    acc = hits / len(true)
    hits_being_one = np.count_nonzero(np.logical_and(prediction, true))
    recall = hits_being_one / np.count_nonzero(true) # hits / # (isClick == 1)
    precision = hits_being_one / np.count_nonzero(prediction) # hits / # (prediction == 1)
    F1 = (2 * recall * precision) / (recall + precision)
    return F1, acc