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
from rtk import console
from rtk.utils import get_logger

METRICS_DIR = "metrics"
logger = get_logger(__name__, console=console)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def retrieval_at_k(y_true: np.ndarray, retrieved_labels: np.ndarray, k: int, suffix: str = ""):
    """
    Calculate retrieval metrics at k.

    Args:
        y_true: Ground truth labels for each query (shape: [n_queries])
        retrieved_labels: Labels of retrieved items for each query (shape: [n_queries, n_retrieved])
        k: Number of top results to consider
        suffix: Suffix to append to metric names

    Returns:
        Dictionary containing mean mAP, mRR, nDCG
    """
    metrics = {}

    # Mean Reciprocal Rank (MRR)
    rr = topk.rr(y_true, retrieved_labels, k=k).mean()
    metrics[f"mrr{suffix}"] = round(rr, 4)

    # Mean Average Precision (MAP)
    ap = topk.ap(y_true, retrieved_labels, k=k).mean()
    metrics[f"map{suffix}"] = round(ap, 4)

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


def generate_retrieval_report(
    args: DictConfig = DictConfig({}), results: dict = {}, log: bool = False, **kwargs
):
    """
    Generate evaluation report for different k values.
    """
    import pandas as pd
    from tabulate import tabulate

    k_values = [1, 2, 4, 8, 10]

    if log:
        MLflow.log_param("k_values", k_values)
        MLflow.log_param("num_samples", len(results))

    title = "# Retrieval VLM Report\n\n"
    description = title

    # Calculate irmetrics (mrr, map) for each k
    pred_data = pd.DataFrame.from_records(results).T

    # Calculate recall and precision metrics for each k
    metrics_data = []

    for k in k_values:
        k_metric = {}
        recalls_p = []
        recalls_n = []
        precisions_p = []
        precisions_n = []
        f1s_p = []
        f1s_n = []

        ## Negative
        neg_pred_data = pred_data[pred_data["y_true"] == 0]
        neg_true = neg_pred_data["y_true"].to_numpy(np.int32)
        neg_pred = np.vstack(neg_pred_data["retrieved_labels"])
        neg_pred = neg_pred.astype(np.int32)
        neg_metrics = retrieval_at_k(neg_true, neg_pred, suffix="_n", k=k)
        k_metric.update(neg_metrics)

        ## Positive
        pos_pred_data = pred_data[pred_data["y_true"] == 1]
        pos_true = pos_pred_data["y_true"].to_numpy(np.int32)
        pos_pred = np.vstack(pos_pred_data["retrieved_labels"])
        pos_pred = pos_pred.astype(np.int32)
        pos_metrics = retrieval_at_k(pos_true, pos_pred, suffix="_p", k=k)
        k_metric.update(pos_metrics)

        # Macro average
        avg_mrr = (neg_metrics[f"mrr_n"] + pos_metrics[f"mrr_p"]) / 2
        avg_map = (neg_metrics[f"map_n"] + pos_metrics[f"map_p"]) / 2
        k_metric.update(
            {
                "map": round(avg_map, 4),
                "mrr": round(avg_mrr, 4),
            }
        )

        for _, data in results.items():
            y_true: list = data["y_true"]
            retrieved_labels: List[list] = data["retrieved_labels"]

            recall_p, recall_n = recall_at_k(y_true, retrieved_labels, k)
            precision_p, precision_n = precision_at_k(y_true, retrieved_labels, k)
            f1_p, f1_n = f1_at_k(y_true, retrieved_labels, k)

            # Only append non-None values
            if recall_p is not None:
                recalls_p.append(recall_p)
            if recall_n is not None:
                recalls_n.append(recall_n)

            if precision_p is not None:
                precisions_p.append(precision_p)
            if precision_n is not None:
                precisions_n.append(precision_n)

            if f1_p is not None:
                f1s_p.append(f1_p)
            if f1_n is not None:
                f1s_n.append(f1_n)

        # Calculate mean metrics
        mean_recall_p = sum(recalls_p) / len(recalls_p) if recalls_p else 0
        mean_recall_n = sum(recalls_n) / len(recalls_n) if recalls_n else 0
        mean_precision_p = sum(precisions_p) / len(precisions_p) if precisions_p else 0
        mean_precision_n = sum(precisions_n) / len(precisions_n) if precisions_n else 0
        mean_f1_p = sum(f1s_p) / len(f1s_p) if f1s_p else 0
        mean_f1_n = sum(f1s_n) / len(f1s_n) if f1s_n else 0

        macro_recall = (mean_recall_p + mean_recall_n) / 2
        macro_precision = (mean_precision_p + mean_precision_n) / 2
        macro_f1 = (mean_f1_p + mean_f1_n) / 2

        k_metric.update(
            {
                f"f1": round(macro_f1, 4),
                f"f1_n": round(mean_f1_n, 4),
                f"f1_p": round(mean_f1_p, 4),
                f"precision": round(macro_precision, 4),
                f"precision_n": round(mean_precision_n, 4),
                f"precision_p": round(mean_precision_p, 4),
                f"recall": round(macro_recall, 4),
                f"recall_n": round(mean_recall_n, 4),
                f"recall_p": round(mean_recall_p, 4),
            }
        )
        if log:
            MLflow.log_metrics(
                k_metric,
                step=k,
            )
        k_metric["k"] = k
        metrics_data.append(k_metric)

    # Save metrics to CSV
    metrics_df = pd.DataFrame(metrics_data).set_index("k")
    metrics_df = metrics_df.sort_index()
    metrics_df_string = tabulate(
        metrics_df.T, headers="keys", showindex=True, tablefmt="fancy_grid"
    )
    description += f"```\n{metrics_df_string}"

    model_name: str = kwargs.get("model_name", args.get("model_name", "model"))
    retrieval_modality: str = kwargs.get(
        "retrieval_modality", args.get("retrieval_modality", "vlm")
    )
    metrics_file = os.path.join(
        METRICS_DIR, f"{model_name}_{retrieval_modality}_retrieval_report.csv"
    )
    metrics_df.to_csv(metrics_file)

    logger.info(f"✓ Metrics saved to: {metrics_file}")
    content = description + "\n"
    console.print(Markdown(description))
    if log:
        MLflow.set_tag("mlflow.note.content", f"{content}```")

    return metrics_df


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
