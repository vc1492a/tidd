"""
A collection of functions covering metrics:

- Calculating a confusion matrix.
- Calculating coverage.
- Recall, precision, F1-Scores.

"""

# imports
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


def calculating_coverage(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.9) -> tuple:
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


def f1_score(precision: float, recall: float) -> float:
    """
    calculates the F1-score given a proviced precision score and recall score.
    :input: precision
    :input: recall
    """

    try:
        f_score = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        f_score = 0.

    return f_score


def confusion_matrix_classification(adjusted_ground_truth_sequence: list, anom_sequences: list) -> tuple:
    """
    Records the total count of true positives, false negatives, and false positives

    A true positive is recorded if any portion of a predicted sequence of anomalies falls within any
    true labeled sequence. Only one true positive is recorded
    even if portions of multiple predicted sequences fall within
    a labeled sequence.

    If no predicted sequences overlap with a positively labeled
    sequence, a false negative is recorded for the labeled sequence

    For all predicted sequences that do not overlap a labeled
    anomalous region, a false positive is recorded

    Also returns tp_lengths and fp_lengths


    :input adjusted_ground_truth_sequences: list of labels for each minute
    :input anom_sequences: list of indices where the minute sequences are considered anomalous
    :input tp_lengths: list of the length of the true anomalous seq
    :input fp_lengths: list of the lengths of the false anomalous seq
    :output: tuple of 3 integers and 2 lists (true positive, false negative, 
            false positive, tp_lengths, fp_lengths)
    """

    tp = 0
    fn = 0
    fp = 0
    tp_lengths = list()
    fp_lengths = list()

    # NOTE: current code assumes one anomalous sequence in ground truth

    # check for false positives and true positives
    for anom_seq in anom_sequences:

        intersection = list(set(adjusted_ground_truth_sequence) & set(anom_seq))

        if len(intersection) > 0:
            tp = 1
            fn = 0

            tp_lengths.append(len(anom_seq))

        else:
            fp += 1

            fp_lengths.append(len(anom_seq))

    # check for false negatives
    fn = 1
    for anom_seq in anom_sequences:
        intersection = list(set(adjusted_ground_truth_sequence) & set(anom_seq))
        if len(intersection) > 0:
            fn = 0
            break

    return tp, fn, fp, tp_lengths, fp_lengths
