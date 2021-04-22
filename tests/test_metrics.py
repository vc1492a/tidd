"""
set of unit tests for functionality within tidd/metrics.py
"""

from tidd.metrics import *


def test_confusion_matrix_scores() -> None:
    """
    Tests whether the confusion matrix score actually works as intended
    :output: None
    """

    # creating the confusion matrix with random numbers
    random_confusion_matrix = np.random.randint(0, 1000, size=(2, 2))
    scores = confusion_matrix_scores(random_confusion_matrix)

    # check we have a 4 tuple
    assert len(scores) == 4

    # checking that the calculations work
    total = np.sum(random_confusion_matrix)
    assert (scores[0] * total) == np.trace(random_confusion_matrix)

    known_values = np.array([[1, 1], [1, 1]])
    scores_from_known_matrix = confusion_matrix_scores(known_values)
    assert all(x == 0.5 for x in scores_from_known_matrix)


def test_calculating_coverage() -> None:
    """
    Tests if the calculating coverage works correctly
    """
    test_predictions = torch.tensor(
        [
            [0.9, 0.1],
            [0.8, 0.2],
            [0.3, 0.7]
        ]
    )
    test_targets = torch.tensor([1, 0, 1])
    threshold_1 = 0.9
    return_tuple_1 = calculating_coverage(test_predictions, test_targets,
                                          threshold_1)
    assert len(return_tuple_1) == 2
    assert return_tuple_1[0] == 0

    threshold_2 = 0.1

    return_tuple_2 = calculating_coverage(test_predictions, test_targets,
                                          threshold_2)
    assert len(return_tuple_2) == 2
    assert return_tuple_2[0] == 1


def test_recall() -> None:
    """
    Test the metrics functions
    """

    recall1 = recall_score(100, 100)
    assert recall1 == 0.5

    recall2 = recall_score(0, 0)
    assert recall2 == 0


def test_precision() -> None:
    """
    Test the metrics functions
    """

    precision1 = precision_score(100, 100)
    assert precision1 == 0.5

    precision2 = precision_score(0, 0)
    assert precision2 == 0


def test_f1() -> None:
    """
    Test the metrics functions
    """

    f1_score1 = f1_score(0.5, 0.5)
    assert f1_score1 == 0.5

    f1_score2 = f1_score(0, 0)
    assert f1_score2 == 0
