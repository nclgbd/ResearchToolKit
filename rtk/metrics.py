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

# rtk
from rtk.utils import get_console, get_logger

METRICS_DIR = "metrics"
console = get_console()
logger = get_logger(__name__, level=logging.DEBUG)


def generate_classification_report(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_score: Union[np.ndarray, pd.Series] = None,
    target_names=[f"No Pneumonia", "Pneumonia"],
    log: bool = False,
):
    logger.info(f"Generating classification report...")
    curr_time = pd.Timestamp.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Classification report
    cr: dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )
    console.print(Markdown("## Summary Report:"))
    summary_dict = {
        "f1-score": round(cr["macro avg"]["f1-score"], 4),
        "sensitivity": round(cr["Pneumonia"]["recall"], 4),
        "specificity": round(cr["No Pneumonia"]["recall"], 4),
        "recall": round(cr["macro avg"]["recall"], 4),
        "precision": round(cr["macro avg"]["precision"], 4),
        # "auc-score": round(roc_auc_score(y_true, y_score, labels=target_names), 4),
        "accuracy": round(cr["accuracy"], 4),
    }
    summary = pd.DataFrame(
        summary_dict,
        index=[f"{curr_time}"],
    )
    summary.index.name = "timestamp"
    console.print(Markdown("## Classification Report"))
    console.print(tabulate(summary, headers="keys", tablefmt="rounded_grid"))

    # Confusion matrix
    console.print(Markdown("## Confusion Matrix"))
    cfm = confusion_matrix(y_true, y_pred)
    console.print(
        tabulate(
            cfm, headers=target_names, showindex=target_names, tablefmt="rounded_grid"
        )
    )
    if log:
        metrics_dir = f"{METRICS_DIR}/{curr_time}".strip()
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
        cfm_df.to_csv(
            cfm_path,
        )

        MLflow.log_artifact(metrics_dir, METRICS_DIR)

    return summary_dict
