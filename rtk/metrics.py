import logging
import mlflow as MLflow
import numpy as np
import os
import pandas as pd
from omegaconf import DictConfig
from rich.markdown import Markdown
from tabulate import tabulate
from typing import List, Union

# sklearn
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# huggingface
from transformers import TrainerState

# rtk
from rtk.utils import get_console, get_logger

METRICS_DIR = "metrics"
console = get_console()
logger = get_logger(__name__, level=logging.DEBUG)


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
        tabulate(
            cfm, headers=target_names, showindex=target_names, tablefmt="rounded_grid"
        )
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
