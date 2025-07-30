import warnings
from itertools import cycle
from pathlib import Path
from statistics import mean
from typing import List, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import mannwhitneyu
from sklearn.metrics import auc, roc_curve

warnings.filterwarnings("ignore")
matplotlib.use("Agg")


def compute_ci(arr: np.ndarray, ci: float) -> Tuple[float, float]:
    # TODO docs
    """

    Args:
        arr:
        ci:

    Returns:

    """
    p = ((1.0 - ci) / 2.0) * 100
    lower = max(0.0, np.percentile(arr, p))
    p = (ci + ((1.0 - ci) / 2.0)) * 100
    upper = min(1.0, np.percentile(arr, p))

    return lower, upper


def find_results(model, dataloader, num_classes):
    with torch.no_grad():
        all_labels = torch.empty(size=(len(dataloader),), dtype=torch.int64).cpu()
        all_predicts = torch.empty(size=(len(dataloader), num_classes), dtype=torch.float32).cpu()

        for idx, (image, label) in enumerate(dataloader):
            all_labels[idx] = label.detach().cpu()
            all_predicts[idx] = model(image.cuda(non_blocking=True))[-1].detach().cpu()

    return all_labels, all_predicts


def compute_auc_multi(labels: List[List[int]], preds: List[List[float]]) -> List[float]:
    """
    Modified version of code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html

    labels:
        k-element-list of n-element-list, where k is the number of classes
        and n is the number of observations. Each inner list represents a one-hot vector.
        ex)
        [[0,0,1],[0,1,0],[0,0,1],[1,0,0]]
    preds:
        k-element-list of n-element-list, where k is the number of classes
        and n is the number of predictions. Again, each inner list represents a one-hot vector.
        ex)
        [[0.1,0.1,0.8],[0.7,0.2,0.1],[0.2,0.3,0.5],[0.9,0.0,0.1]]
    """
    n_classes = len(labels[0])
    fprs, tprs = [list(), list()]
    roc_aucs = list()
    # roc curve for each class
    for i in range(n_classes):
        fpr, tpr, __ = roc_curve([e[i] for e in labels], [e[i] for e in preds])
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(auc(fpr, tpr))

    return roc_aucs


def calculate_confusion_matrix(all_labels: np.ndarray, all_predicts: np.ndarray,
                               classes: Tuple[str, ...]) -> pd.DataFrame:
    """
    Compute the confusion matrix for the provided ground-truth labels and predictions and print.

    Args:
        all_labels: Ground-truth labels
        all_predicts: Predicted labels
        classes: Class names for printing on x/y axes
    """
    remap_classes = {x: classes[x] for x in range(len(classes))}

    pd.options.display.float_format = "{:.2f}".format
    pd.options.display.width = 0

    actual = pd.Series(pd.Categorical(pd.Series(all_labels).replace(remap_classes), categories=classes), name="Actual")
    predicted = pd.Series(pd.Categorical(pd.Series(all_predicts).replace(remap_classes), categories=classes),
                          name="Predicted")

    cm = pd.crosstab(index=actual, columns=predicted, normalize="index", dropna=False)

    cm.style.hide(axis="index")

    return cm


def save_confusion_matrix(cm: pd.DataFrame, class_names: Tuple[str, ...], output_name: Path) -> None:
    # TODO docs
    """

    Args:
        cm:
        class_names:
        output_name:
    """
    plt.figure()
    df = pd.DataFrame(cm, columns=class_names, index=class_names)
    df.index.name = "Actual"
    df.columns.name = "Predicted"
    sns.heatmap(df, annot=True, cbar=False, vmin=0.0, vmax=1.0)
    plt.savefig(output_name)
    plt.close("all")


def estimate_optimal_threshold(obs, pred):
    # TODO docs
    def find_optimal(minv, maxv, step):
        candidates = []
        while minv <= maxv:
            candidates.append(minv)
            minv += step
        best_score = 0
        best_c = 0
        for c in candidates:
            use_gpu = torch.cuda.is_available()
            score = compute_measures(obs, pred, c, use_gpu=use_gpu)['f1score']
            if best_score < score:
                best_score = score
                best_c = c
        return best_score, best_c

    minv = 0
    maxv = 1
    step = 0.1
    best_f1, opt_threshold = [0, 0]
    for i in range(3):
        best_f1, opt_threshold = find_optimal(minv, maxv, step)
        minv = opt_threshold - step
        maxv = opt_threshold + step
        step /= 10
    return best_f1, opt_threshold


def compute_measures(obs, pred, threshold=0.5, use_gpu=False):
    # TODO docs
    """
    Args:
        obs: list-like object for labels
        pred: list-like object for predicted values

    Returns:
        Acc
        TPR
        PPN
        F1
    """
    if not use_gpu:
        obs = np.array(obs)
        pred = np.array(pred)
        p_inds = obs == 1
        n_inds = obs == 0
        tp = sum(pred[p_inds] >= threshold)
        fp = sum(pred[n_inds] >= threshold)
        tn = sum(pred[n_inds] < threshold)
        fn = sum(pred[p_inds] < threshold)
        n = sum(n_inds)
        p = sum(p_inds)
    else:
        with torch.no_grad():
            obs = torch.FloatTensor(obs).cuda()
            pred = torch.FloatTensor(pred).cuda()
            p_inds = obs == 1
            n_inds = obs == 0
            tp = (pred[p_inds] >= threshold).sum().item()
            fp = (pred[n_inds] >= threshold).sum().item()
            tn = (pred[n_inds] < threshold).sum().item()
            fn = (pred[p_inds] < threshold).sum().item()
            n = n_inds.sum().item()
            p = p_inds.sum().item()
    result = dict()
    result['accuracy'] = (tp + tn) / (p + n)
    result['sensitivity'] = tp / p
    result['f1score'] = 2 * tp / (2 * tp + fp + fn)
    result['precision'] = tp / (tp + fp)
    return result


def compute_pvalue(obs, pred):
    # TODO docs
    obs = np.array(obs)
    pred = np.array(pred)
    test = mannwhitneyu(x=pred[obs == 1], y=pred[obs == 0], alternative='greater')
    return test.pvalue


def save_roc_curve_multi(obs_lists, pred_lists, class_names, figdest='roc_curve_multi.png', figformat='png',
                         figsize_inches=[7, 7], linewidth=2, dpi=300, show_micro_avg=False, show_macro_avg=False):
    # TODO docs
    """
    obs_lists: k-element-list of n-element-list, where k is the number of classes
        and n is the number of observations. Each inner list represents a one-hot vector.
        ex)
        [[0,0,1],[0,1,0],[0,0,1],[1,0,0]]
    pred_lists: k-element-list of n-element-list, where k is the number of classes
        and n is the number of predictions. Again, each inner list represents a one-hot vector.
        ex)
        [[0.1,0.1,0.8],[0.7,0.2,0.1],[0.2,0.3,0.5],[0.9,0.0,0.1]]
    """
    assert len(obs_lists) == len(pred_lists)
    n_classes = len(obs_lists[0])
    fprs, tprs = [list(), list()]
    roc_aucs = list()
    # roc curve for each class
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve([e[i] for e in obs_lists], [e[i] for e in pred_lists])
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(auc(fpr, tpr))
    if show_micro_avg:
        # micro-average roc curve
        fpr, tpr, _ = roc_curve([e for v in obs_lists for e in v], [e for v in pred_lists for e in v])
        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(auc(fpr, tpr))

    if show_macro_avg:
        # macro-average roc curve
        all_fpr = list(set([e for i in range(n_classes) for e in fprs[i]]))
        all_fpr = sorted(all_fpr)
        mean_tpr = list()
        from scipy import interp
        for i in range(n_classes):
            mean_tpr.append(interp(all_fpr, fprs[i], tprs[i]))
        mean_tpr = [mean([mean_tpr[k][i] for k in range(n_classes)]) for i in range(len(all_fpr))]
        fprs.append(all_fpr)
        tprs.append(mean_tpr)
        roc_aucs.append(auc(all_fpr, mean_tpr))

    fig = plt.figure(figsize=figsize_inches)
    if show_micro_avg:
        index = -2 if show_macro_avg else -1
        plt.plot(fprs[index], tprs[index],
                 label='micro-average ROC curve (area = {:0.2f})'.format(roc_aucs[-2]),
                 color='deeppink', linestyle=':', linewidth=4)
    if show_macro_avg:
        plt.plot(fprs[-1], tprs[-1],
                 label='macro-average ROC curve (area = {:0.2f})'.format(roc_aucs[-1]),
                 color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fprs[i], tprs[i], color=color, lw=linewidth,
                 label="{} (area = {:0.2f})".format(class_names[i], roc_aucs[i]))

    plt.plot([0, 1], [0, 1], color='black', lw=linewidth, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristics')

    plt.legend(loc='lower right')
    plt.savefig(figdest, format=figformat, dpi=dpi)
    return fig


def save_roc_curve(obs_list, pred_list, figdest='roc_curve.png', figformat='png',
                   figsize_inches=[7, 7], linecolor='darkorange', linewidth=2, dpi=300, advanced=False):
    # TODO docs
    fpr, tpr, _ = roc_curve(obs_list, pred_list)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure(figsize=figsize_inches)
    label = "ROC curve (AUC = {:0.2f}".format(roc_auc)
    if advanced:
        label += ", p = {:0.5f})".format(compute_pvalue(obs_list, pred_list))
    else:
        label += ')'
    plt.plot(fpr, tpr, color=linecolor, lw=linewidth,
             label=label)
    plt.plot([0, 1], [0, 1], color='grey', lw=linewidth, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.01])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if advanced:
        use_gpu = torch.cuda.is_available()
        measures = compute_measures(obs_list, pred_list, use_gpu)
        plt.title('Receiver operating characteristics' +
                  "\nAccuracy:{:0.2f} Sensitivity:{:0.2f} Precision:{:0.2f} F-1 Score:{:0.2f}".format(
                      measures['accuracy'],
                      measures['sensitivity'],
                      measures['precision'],
                      measures['f1score']))

    plt.legend(loc='lower right')
    plt.savefig(figdest, format=figformat, dpi=dpi)
    return fig


def convert_labels_to_onehot_vectors(label_list, num_class):
    # TODO docs
    """
    label_list: a list of label in int
    ex)
        [0, 1, 0, 2, 1, 0,...] with num_class=3
        note:0-indexed
    Return: a list of one-hot vectors
        [[1,0,0],[0,1,0],[0,0,1],[0,1,0],[1,0,0],...]

    """
    return [[1 if i == e else 0 for i in range(num_class)] for e in label_list]


def compute_metrics(model, dataloader, num_classes, class_names, auc_dest, cm_dest):
    all_labels, all_predicts = find_results(model=model, dataloader=dataloader, num_classes=num_classes)

    loss = torch.nn.CrossEntropyLoss()(input=all_predicts, target=all_labels).item()
    # TODO Check if boolean casts to float properly
    acc = torch.mean((torch.max(all_predicts, dim=1)[1] == all_labels).float()).item()

    # Save the ROC curves.
    __ = save_roc_curve_multi(obs_lists=torch.nn.functional.one_hot(all_labels).tolist(),
                              pred_lists=all_predicts.tolist(), figdest=auc_dest, class_names=class_names,
                              show_micro_avg=True, show_macro_avg=True)

    # Compute the AUCs.
    auc = compute_auc_multi(labels=torch.nn.functional.one_hot(all_labels).tolist(), preds=all_predicts.tolist())

    # Save the confusion matrices.
    cm = calculate_confusion_matrix(all_labels=all_labels.numpy(),
                                    all_predicts=torch.max(all_predicts, dim=1)[1].numpy(), classes=class_names)
    save_confusion_matrix(cm=cm, class_names=class_names, output_name=cm_dest)

    return loss, acc, auc