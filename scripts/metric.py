

def compute_f1(y_true, y_pred, label, beta=1):
    tp = sum((yt == label) and (yp == label) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt != label) and (yp == label) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == label) and (yp != label) for yt, yp in zip(y_true, y_pred))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    if precision + recall == 0:
        return 0

    f1 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall)
    return f1


def compute_recall(y_true, y_pred, label):
    tp = sum((yt == label) and (yp == label) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == label) and (yp != label) for yt, yp in zip(y_true, y_pred))
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return recall


def normalized_inverse_class_frequency(y_true):
    """Compute normalized inverse class frequency (NICF) for each class"""
    calculate_class_frequencies = lambda y_true: {label: list(y_true).count(label) / len(y_true) for label in set(y_true)}
    inverse_class_frequency = {label: 1.0 / freq for label, freq in calculate_class_frequencies(y_true).items()}
    normalized_icf = {label: icf / sum(inverse_class_frequency.values()) for label, icf in inverse_class_frequency.items()}
    return normalized_icf


def weighted_metric_using_icf(y_true, y_pred, compute_metric, metric_args={}):
    normalized_icf = normalized_inverse_class_frequency(y_true)
    weighted_metric = 0.0
    for label, weight in normalized_icf.items():
        metric = compute_metric(y_true, y_pred, label, **metric_args)
        weighted_metric += weight * metric
    return weighted_metric
