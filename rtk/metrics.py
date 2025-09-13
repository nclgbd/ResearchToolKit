import os
from typing import List, Union
import mlflow
import numpy as np
import pandas as pd

# rtk
from rtk.utils import get_console, get_logger

console = get_console()
logger = get_logger(__name__)


def generate_classification_report(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    target_names=[f"No Pneumonia", "Pneumonia"],
    log: bool = False,
    epoch: int = 0,
    split: str = "",
    model_name: str = "model",
):
    """ """
    from sklearn.metrics import classification_report, confusion_matrix
    from tabulate import tabulate

    metrics_dir = f"metrics/{split}"
    os.makedirs(metrics_dir, exist_ok=True)
    # Classification report
    cr: dict = classification_report(
        y_true, y_pred, target_names=target_names, output_dict=True
    )
    console.print("Classification Report:")
    console.print(
        classification_report(
            y_true, y_pred, target_names=target_names, zero_division=0.0
        )
    )
    summary = pd.DataFrame(
        {
            f"{split}_f1-score": round(cr["macro avg"]["f1-score"], 4),
            # "roc auc": round(roc_auc_score(y_true, y_pred), 4),
            f"{split}_sensitivity": round(cr["Pneumonia"]["recall"], 4),
            f"{split}_specificity": round(cr["No Pneumonia"]["recall"], 4),
            f"{split}_recall": round(cr["macro avg"]["recall"], 4),
            f"{split}_precision": round(cr["macro avg"]["precision"], 4),
            f"{split}_accuracy": round(cr["accuracy"], 4),
        },
        index=[model_name],
    )
    console.print(summary)
    if log:
        header = epoch == 0
        summary_path = os.path.join(
            metrics_dir, f"{model_name}-classification_summary.csv"
        )
        summary.to_csv(
            summary_path,
            mode="a",
            header=header,
        )
        mlflow.log_artifact(summary_path)

    # Confusion matrix
    console.print("Confusion Matrix:")
    cfm = confusion_matrix(y_true, y_pred)
    console.print(
        tabulate(
            cfm, headers=target_names, showindex=target_names, tablefmt="rounded_grid"
        )
    )
    cfm_df = pd.DataFrame(cfm, index=target_names, columns=target_names)
    if log:
        cfm_path = os.path.join(metrics_dir, f"confusion_matrix_epoch={epoch}.csv")
        cfm_df.to_csv(
            cfm_path,
        )
        mlflow.log_artifact(
            cfm_path,
        )
