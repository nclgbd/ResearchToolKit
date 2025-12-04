import logging
import mlflow as MLflow
import numpy as np
import os
import pandas as pd
import warnings
from omegaconf import DictConfig
from rich.markdown import Markdown
from tabulate import tabulate
from typing import List, Union

# sklearn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# irmetrics
from irmetrics import topk

# huggingface
from transformers import TrainerState

# rtk
from rtk.utils import get_console, get_logger

METRICS_DIR = "metrics"
console = get_console()
logger = get_logger(__name__, level=logging.INFO)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def retrieval_at_k(y_true: np.ndarray, retrieved_labels: np.ndarray, k: int, suffix: str = ""):
    metrics = {}
    rr = topk.rr(y_true, retrieved_labels, k=k).mean()
    ap = topk.ap(y_true, retrieved_labels, k=k).mean()
    ndcg = topk.ndcg(y_true, retrieved_labels, k=k).mean()
    ndcg = ndcg if not np.isnan(ndcg) else 0.0
    metrics[f"mrr{suffix}:k"] = round(rr, 4)
    metrics[f"map{suffix}:k"] = round(ap, 4)
    metrics[f"ndcg{suffix}:k"] = round(ndcg, 4)

    return metrics


def recall_at_k(y_true: int, retrieved_labels: list, k: int) -> float:
    """
    Calculate recall@k for a single query.

    Args:
        y_true: Ground truth label (0 or 1)
        retrieved_labels: List of labels from retrieved samples
        k: Number of top results to consider

    Returns:
        Recall score (1.0 if any positive sample found in top-k when y_true=1, else 0.0)
    """
    top_k_labels = retrieved_labels[:k]

    if y_true == 1:  # Ground truth is positive
        recall_p = 1.0 if any(label == 1 for label in top_k_labels) else 0.0
        recall_n = None
    else:  # Ground truth is negative
        recall_n = 1.0 if any(label == 0 for label in top_k_labels) else 0.0
        recall_p = None

    return (recall_p, recall_n)


def precision_at_k(y_true: int, retrieved_labels: list, k: int) -> float:
    """
    Calculate precision@k for a single query.

    Args:
        y_true: Ground truth label (0 or 1)
        retrieved_labels: List of labels from retrieved samples
        k: Number of top results to consider

    Returns:
        Precision score (fraction of retrieved positive samples in top-k)
    """
    top_k_labels = retrieved_labels[:k]

    if y_true == 1:  # Ground truth is positive
        num_correct = sum(1 for label in top_k_labels if label == 1)
        precision_p = num_correct / k
        precision_n = None
    else:  # Ground truth is negative
        num_correct = sum(1 for label in top_k_labels if label == 0)
        precision_n = num_correct / k
        precision_p = None

    return (precision_p, precision_n)


def f1_at_k(y_true: int, retrieved_labels: list, k: int) -> float:
    """
    Calculate F1@k for a single query.

    Args:
        y_true: Ground truth label (0 or 1)
        retrieved_labels: List of labels from retrieved samples
        k: Number of top results to consider

    Returns:
        F1 score
    """
    recall_p, recall_n = recall_at_k(y_true, retrieved_labels, k)
    precision_p, precision_n = precision_at_k(y_true, retrieved_labels, k)

    # Calculate F1 for positive class
    if recall_p is None or precision_p is None:
        f1_p = None
    elif precision_p + recall_p == 0:
        f1_p = 0.0
    else:
        f1_p = 2 * (precision_p * recall_p) / (precision_p + recall_p)

    # Calculate F1 for negative class
    if recall_n is None or precision_n is None:
        f1_n = None
    elif precision_n + recall_n == 0:
        f1_n = 0.0
    else:
        f1_n = 2 * (precision_n * recall_n) / (precision_n + recall_n)

    return (f1_p, f1_n)


def generate_classification_report(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_score: Union[np.ndarray, pd.Series] = None,
    target_names=["No Pneumonia", "Pneumonia"],
    log: bool = False,
    state: TrainerState = None,
    split: str = "",
    metrics_dir: str = METRICS_DIR,
):
    logger.info(f"Generating classification report...")
    positive_class: str = target_names[1]
    if positive_class == "No Finding":
        negative_class: str = positive_class
        positive_class = "Finding"
        target_names = [positive_class, negative_class]
    else:
        negative_class: str = target_names[0]
    curr_step = "000000" if state is None else str(state.global_step).zfill(6)
    # Classification report
    labels = list(range(len(target_names)))
    cr: dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        target_names=target_names,
        zero_division=0.0,
    )
    console.print(Markdown("## Summary Report:"))
    split_prefix = f"{split}_" if split else ""
    summary_dict = {
        f"{split_prefix}f1": round(cr["macro avg"]["f1-score"], 4),
        f"{split_prefix}sensitivity": round(cr[positive_class]["recall"], 4),
        f"{split_prefix}specificity": round(cr[negative_class]["recall"], 4),
        f"{split_prefix}recall": round(cr["macro avg"]["recall"], 4),
        f"{split_prefix}precision": round(cr["macro avg"]["precision"], 4),
        # "auc-score": round(roc_auc_score(y_true, y_score, labels=target_names), 4),
        f"{split_prefix}accuracy": round(cr["accuracy"], 4),
    }
    index = curr_step
    summary = pd.DataFrame(
        summary_dict,
        index=[index],
    )
    summary.index.name = "step"
    console.print(Markdown("## Classification Report"))
    console.print(tabulate(summary, headers="keys", tablefmt="rounded_grid"))

    # Confusion matrix
    console.print(Markdown("## Confusion Matrix"))
    cfm = confusion_matrix(y_true, y_pred, labels=labels)
    console.print(
        tabulate(cfm, headers=target_names, showindex=target_names, tablefmt="rounded_grid")
    )
    if log:
        subfolder = curr_step if split == "" else f"{split}/{positive_class}"
        metrics_dir = f"{metrics_dir}/{subfolder}".strip()
        os.makedirs(metrics_dir, exist_ok=True)
        # Classification summary
        summary_path = os.path.join(metrics_dir, f"classification_summary.csv")
        logger.debug(f"Saving classification summary to '{summary_path}'")
        summary.to_csv(
            summary_path,
        )
        # Confusion matrix
        cfm_df = pd.DataFrame(cfm, index=target_names, columns=target_names)
        cfm_df.index.name = "labels"
        cfm_path = os.path.join(metrics_dir, f"confusion_matrix.csv")
        logger.debug(f"Saving confusion matrix to '{cfm_path}'")
        cfm_df.to_csv(cfm_path)

        if state:
            MLflow.log_artifact(metrics_dir, metrics_dir)
        else:
            summary_dict = {
                f"{k}_{positive_class.lower().replace(' ', '_')}": v
                for k, v in summary_dict.items()
            }
            MLflow.log_metrics(summary_dict, step=int(curr_step))

    return summary_dict
