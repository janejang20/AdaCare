# from __future__ import absolute_import
# from __future__ import print_function

# import numpy as np
# from sklearn import metrics


# # for decompensation, in-hospital mortality

# def print_metrics_binary(y_true, predictions, verbose=1):
#     predictions = np.array(predictions)
#     if len(predictions.shape) == 1:
#         predictions = np.stack([1 - predictions, predictions]).transpose((1, 0))

#     cf = metrics.confusion_matrix(y_true, predictions.argmax(axis=1))
#     if verbose:
#         print("confusion matrix:")
#         print(cf)
#     cf = cf.astype(np.float32)

#     acc = (cf[0][0] + cf[1][1]) / np.sum(cf)
#     prec0 = cf[0][0] / (cf[0][0] + cf[1][0])
#     prec1 = cf[1][1] / (cf[1][1] + cf[0][1])
#     rec0 = cf[0][0] / (cf[0][0] + cf[0][1])
#     rec1 = cf[1][1] / (cf[1][1] + cf[1][0])
#     auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

#     (precisions, recalls, thresholds) = metrics.precision_recall_curve(y_true, predictions[:, 1])
#     auprc = metrics.auc(recalls, precisions)
#     minpse = np.max([min(x, y) for (x, y) in zip(precisions, recalls)])

#     if verbose:
#         print("accuracy = {}".format(acc))
#         print("precision class 0 = {}".format(prec0))
#         print("precision class 1 = {}".format(prec1))
#         print("recall class 0 = {}".format(rec0))
#         print("recall class 1 = {}".format(rec1))
#         print("AUC of ROC = {}".format(auroc))
#         print("AUC of PRC = {}".format(auprc))
#         print("min(+P, Se) = {}".format(minpse))

#     return {"acc": acc,
#             "prec0": prec0,
#             "prec1": prec1,
#             "rec0": rec0,
#             "rec1": rec1,
#             "auroc": auroc,
#             "auprc": auprc,
#             "minpse": minpse}


# # for phenotyping

# def print_metrics_multilabel(y_true, predictions, verbose=1):
#     y_true = np.array(y_true)
#     predictions = np.array(predictions)

#     auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
#     ave_auc_micro = metrics.roc_auc_score(y_true, predictions,
#                                           average="micro")
#     ave_auc_macro = metrics.roc_auc_score(y_true, predictions,
#                                           average="macro")
#     ave_auc_weighted = metrics.roc_auc_score(y_true, predictions,
#                                              average="weighted")

#     if verbose:
#         print("ROC AUC scores for labels:", auc_scores)
#         print("ave_auc_micro = {}".format(ave_auc_micro))
#         print("ave_auc_macro = {}".format(ave_auc_macro))
#         print("ave_auc_weighted = {}".format(ave_auc_weighted))

#     return {"auc_scores": auc_scores,
#             "ave_auc_micro": ave_auc_micro,
#             "ave_auc_macro": ave_auc_macro,
#             "ave_auc_weighted": ave_auc_weighted}


# # for length of stay

# def mean_absolute_percentage_error(y_true, y_pred):
#     return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100


# def print_metrics_regression(y_true, predictions, verbose=1):
#     predictions = np.array(predictions)
#     predictions = np.maximum(predictions, 0).flatten()
#     y_true = np.array(y_true)

#     y_true_bins = [get_bin_custom(x, CustomBins.nbins) for x in y_true]
#     prediction_bins = [get_bin_custom(x, CustomBins.nbins) for x in predictions]
#     cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
#     if verbose:
#         print("Custom bins confusion matrix:")
#         print(cf)

#     kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins,
#                                       weights='linear')
#     mad = metrics.mean_absolute_error(y_true, predictions)
#     mse = metrics.mean_squared_error(y_true, predictions)
#     mape = mean_absolute_percentage_error(y_true, predictions)

#     if verbose:
#         print("Mean absolute deviation (MAD) = {}".format(mad))
#         print("Mean squared error (MSE) = {}".format(mse))
#         print("Mean absolute percentage error (MAPE) = {}".format(mape))
#         print("Cohen kappa score = {}".format(kappa))

#     return {"mad": mad,
#             "mse": mse,
#             "mape": mape,
#             "kappa": kappa}


# class LogBins:
#     nbins = 10
#     means = [0.611848, 2.587614, 6.977417, 16.465430, 37.053745,
#              81.816438, 182.303159, 393.334856, 810.964040, 1715.702848]


# def get_bin_log(x, nbins, one_hot=False):
#     binid = int(np.log(x + 1) / 8.0 * nbins)
#     if binid < 0:
#         binid = 0
#     if binid >= nbins:
#         binid = nbins - 1

#     if one_hot:
#         ret = np.zeros((LogBins.nbins,))
#         ret[binid] = 1
#         return ret
#     return binid


# def get_estimate_log(prediction, nbins):
#     bin_id = np.argmax(prediction)
#     return LogBins.means[bin_id]


# def print_metrics_log_bins(y_true, predictions, verbose=1):
#     y_true_bins = [get_bin_log(x, LogBins.nbins) for x in y_true]
#     prediction_bins = [get_bin_log(x, LogBins.nbins) for x in predictions]
#     cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
#     if verbose:
#         print("LogBins confusion matrix:")
#         print(cf)
#     return print_metrics_regression(y_true, predictions, verbose)


# class CustomBins:
#     inf = 1e18
#     bins = [(-inf, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 14), (14, +inf)]
#     nbins = len(bins)
#     means = [11.450379, 35.070846, 59.206531, 83.382723, 107.487817,
#              131.579534, 155.643957, 179.660558, 254.306624, 585.325890]


# def get_bin_custom(x, nbins, one_hot=False):
#     for i in range(nbins):
#         a = CustomBins.bins[i][0] * 24.0
#         b = CustomBins.bins[i][1] * 24.0
#         if a <= x < b:
#             if one_hot:
#                 ret = np.zeros((CustomBins.nbins,))
#                 ret[i] = 1
#                 return ret
#             return i
#     return None


# def get_estimate_custom(prediction, nbins):
#     bin_id = np.argmax(prediction)
#     assert 0 <= bin_id < nbins
#     return CustomBins.means[bin_id]


# def print_metrics_custom_bins(y_true, predictions, verbose=1):
#     return print_metrics_regression(y_true, predictions, verbose)

import numpy as np
from sklearn import metrics

# Binary classification metrics
def print_metrics_binary(y_true, predictions, verbose=1):
    predictions = np.asarray(predictions)
    if predictions.ndim == 1:
        predictions = np.vstack([1 - predictions, predictions]).T

    y_pred = predictions.argmax(axis=1)
    cf = metrics.confusion_matrix(y_true, y_pred)
    if verbose:
        print("Confusion matrix:")
        print(cf)
    cf = cf.astype(float)

    tn, fp, fn, tp = cf.ravel()
    acc = (tp + tn) / cf.sum()
    prec0 = tn / (tn + fn) if (tn + fn) > 0 else 0
    prec1 = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec0 = tn / (tn + fp) if (tn + fp) > 0 else 0
    rec1 = tp / (tp + fn) if (tp + fn) > 0 else 0
    auroc = metrics.roc_auc_score(y_true, predictions[:, 1])

    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, predictions[:, 1])
    auprc = metrics.auc(recalls, precisions)
    minpse = np.max(np.minimum(precisions, recalls))

    if verbose:
        print(f"Accuracy = {acc}")
        print(f"Precision class 0 = {prec0}")
        print(f"Precision class 1 = {prec1}")
        print(f"Recall class 0 = {rec0}")
        print(f"Recall class 1 = {rec1}")
        print(f"AUC of ROC = {auroc}")
        print(f"AUC of PRC = {auprc}")
        print(f"min(+P, Se) = {minpse}")

    return {
        "acc": acc,
        "prec0": prec0,
        "prec1": prec1,
        "rec0": rec0,
        "rec1": rec1,
        "auroc": auroc,
        "auprc": auprc,
        "minpse": minpse
    }

# Multilabel classification metrics
def print_metrics_multilabel(y_true, predictions, verbose=1):
    y_true = np.asarray(y_true)
    predictions = np.asarray(predictions)

    auc_scores = metrics.roc_auc_score(y_true, predictions, average=None)
    ave_auc_micro = metrics.roc_auc_score(y_true, predictions, average="micro")
    ave_auc_macro = metrics.roc_auc_score(y_true, predictions, average="macro")
    ave_auc_weighted = metrics.roc_auc_score(y_true, predictions, average="weighted")

    if verbose:
        print(f"ROC AUC scores for labels: {auc_scores}")
        print(f"Average AUC (micro) = {ave_auc_micro}")
        print(f"Average AUC (macro) = {ave_auc_macro}")
        print(f"Average AUC (weighted) = {ave_auc_weighted}")

    return {
        "auc_scores": auc_scores,
        "ave_auc_micro": ave_auc_micro,
        "ave_auc_macro": ave_auc_macro,
        "ave_auc_weighted": ave_auc_weighted
    }

# Regression metrics
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 0.1))) * 100

def print_metrics_regression(y_true, predictions, verbose=1):
    predictions = np.maximum(predictions.flatten(), 0)
    y_true = np.asarray(y_true)

    y_true_bins = get_bin_custom(y_true)
    prediction_bins = get_bin_custom(predictions)
    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("Custom bins confusion matrix:")
        print(cf)

    kappa = metrics.cohen_kappa_score(y_true_bins, prediction_bins, weights='linear')
    mad = metrics.mean_absolute_error(y_true, predictions)
    mse = metrics.mean_squared_error(y_true, predictions)
    mape = mean_absolute_percentage_error(y_true, predictions)

    if verbose:
        print(f"Mean Absolute Deviation (MAD) = {mad}")
        print(f"Mean Squared Error (MSE) = {mse}")
        print(f"Mean Absolute Percentage Error (MAPE) = {mape}")
        print(f"Cohen Kappa Score = {kappa}")

    return {
        "mad": mad,
        "mse": mse,
        "mape": mape,
        "kappa": kappa
    }

# Binning classes and functions
class CustomBins:
    bins = np.array([0, 24, 48, 72, 96, 120, 144, 168, 192, 336, np.inf])  # in hours
    nbins = len(bins) - 1
    means = np.array([11.45, 35.07, 59.21, 83.38, 107.49, 131.58, 155.64, 179.66, 254.31, 585.33])

def get_bin_custom(x):
    x = np.asarray(x)
    bin_ids = np.digitize(x, CustomBins.bins) - 1
    bin_ids = np.clip(bin_ids, 0, CustomBins.nbins - 1)
    return bin_ids

def get_estimate_custom(predictions):
    predictions = np.asarray(predictions)
    bin_ids = predictions.argmax(axis=1)
    return CustomBins.means[bin_ids]

def print_metrics_custom_bins(y_true, predictions, verbose=1):
    return print_metrics_regression(y_true, predictions, verbose)

# If you need log bins (optional)
class LogBins:
    nbins = 10
    bins = np.logspace(0, 8, num=nbins + 1, base=np.e) - 1
    means = np.array([0.61, 2.59, 6.98, 16.47, 37.05, 81.82, 182.30, 393.33, 810.96, 1715.70])

def get_bin_log(x):
    x = np.asarray(x)
    bin_ids = np.digitize(np.log(x + 1), LogBins.bins) - 1
    bin_ids = np.clip(bin_ids, 0, LogBins.nbins - 1)
    return bin_ids

def get_estimate_log(predictions):
    predictions = np.asarray(predictions)
    bin_ids = predictions.argmax(axis=1)
    return LogBins.means[bin_ids]

def print_metrics_log_bins(y_true, predictions, verbose=1):
    y_true_bins = get_bin_log(y_true)
    prediction_bins = get_bin_log(predictions)
    cf = metrics.confusion_matrix(y_true_bins, prediction_bins)
    if verbose:
        print("Log bins confusion matrix:")
        print(cf)
    return print_metrics_regression(y_true, predictions, verbose)