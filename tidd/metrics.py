import numpy as np
import torch


def confusion_matrix_scores(cm: np.ndarray) -> tuple:
    """
    Returns a tuple of classification metrics according to
    the anomalous class as True Positive

    :input cm: a numpy array of shape (2, 2) that is formatted like the output
    of the ClassificationInterpretation object in FastAI
    :output: 4-tuple of the accuracy, precision, recall, and F1 scores
    """

    accuracy = np.trace(cm)/np.sum(cm)

    # total anomalous values and predicted values
    total_anon = np.sum(cm[0]) if np.sum(cm[0]) > 0 else 1
    pred_anon = np.sum(cm[:, 0]) if np.sum(cm[:, 0]) > 0 else 1

    # precision/ recall
    precision = cm[0, 0] / total_anon
    recall = cm[0, 0] / pred_anon

    # denominator for the
    denom = (precision + recall) if (precision + recall) > 0 else 1
    F1 = 2 * precision * recall / denom
    return accuracy, precision, recall, F1


def calculating_coverage(predictions: torch.Tensor, targets: torch.Tensor,
                         threshold: float) -> tuple:
    """
    Given a N-sized validation set,
    predictions is an N x 2 tensor since this is a binary classification problem
    targets is an N x 1 tensor where each targets[i] is the correct class
    
    :input predictions: N x 2 tensor with the classification confidences
    :input targets: N x 1 tensor with the correct labels
    :input threshold: float number representing the target confidence threshold
    :output: 2-tuple with the coverage of the anomaly and normal classifications
    """
    anomalous = np.where(targets == 0)
    normal = np.where(targets == 1)
    normal_predictions = predictions[normal, 1]
    anomalous_predictions = predictions[anomalous, 0]
    anom_coverage = np.average(anomalous_predictions > threshold)
    normal_coverage = np.average(normal_predictions > threshold)
    return anom_coverage, normal_coverage


def precision_score(true_positive_count: int, false_positive_count: int) -> float: 
    """
    calculates the precision score using the count of true positives and count of 
    false positives. 
    :input: count of true positives 
    :input" count of false positives 
    """

    try:
        precision = true_positive_count / (true_positive_count + false_positive_count)
    except ZeroDivisionError:
        precision = 0.

    return precision


def recall_score(true_positive_count: int, false_negative_count: int) -> float:
    """
    calculates the recall score using the count of true positives and count of
    false negatives.
    :input: count of true positives
    :input: count of false negatives
    """

    try:
        recall = true_positive_count / (true_positive_count + false_negative_count)
    except ZeroDivisionError:
        recall = 0.

    return recall


def f1_score(precision_score: float, recall_score: float) -> float:
    """
    calculates the F1-score given a proviced precision score and recall score.
    :input: precision score
    :input: recall score
    """

    try:
        f_score = 2 * ((precision_score * recall_score) / (precision_score +
                                                           recall_score))
    except ZeroDivisionError:
        f_score = 0.

    return f_score
