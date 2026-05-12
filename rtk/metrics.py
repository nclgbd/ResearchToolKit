"""
Metrics module for CRUX: Case Retrieval Using Clinical Indications and X-rays.

This module provides two families of metrics:

1. RETRIEVAL METRICS — evaluate ranked lists returned by a retrieval system.
   Identified by their @k index: any metric that varies across k values
   is a retrieval metric.

   - hit@k:       Did at least one same-label case appear in the top-k?
   - prec@k: What fraction of the top-k share the query's label?
   - f1@k:        Harmonic mean of hit and prec.
   - mrr@k:       Mean reciprocal rank of the first same-label case.
   - map@k:       Mean average prec over the ranked list.

2. CLASSIFICATION METRICS — evaluate binary label predictions against
   ground truth. Computed at k=1 (where the top-1 retrieved label is
   treated as an implicit prediction), and usable standalone for
   direct classification tasks (e.g. VLM-based labeling).

   - sens:      TP / (TP + FN)
   - spec:      TN / (TN + FP)
   - ppv:              TP / (TP + FP)
   - npv:              TN / (TN + FN)
   - cls_f1_p:  Harmonic mean of sens and PPV
   - cls_f1_n:  Harmonic mean of spec and NPV
   - macro_f1:     (cls_f1_p + cls_f1_n) / 2
   - bal_acc: (sens + spec) / 2
   - mcc:              Matthews correlation coefficient

The distinction is structural: retrieval metrics appear at every k
value in the output DataFrame. Classification metrics appear only at
k=1 and are NaN elsewhere.

See metrics_reference.md for a full lookup table.
"""

import logging
import math
import mlflow as MLflow
import numpy as np
import os
import pandas as pd
import warnings
from omegaconf import DictConfig
from rich.markdown import Markdown
from rich.progress import track
from tabulate import tabulate
from textwrap import dedent
from typing import Dict, List, Optional, Tuple, Union

# metrics
from irmetrics import topk
from sklearn.metrics import classification_report, confusion_matrix
from spacy.util import logger

# huggingface
from transformers import TrainerState

# rtk
from rtk import console
from rtk.datasets import extract_sections
from rtk.utils import get_logger

logger = get_logger(__name__, console=console)
logger.setLevel(logging.WARNING)
warnings.filterwarnings("ignore", category=RuntimeWarning)
logging.getLogger("root").propagate = False
logging.getLogger("root").disabled = True

K_VALUES = [1, 3, 5, 8, 10]
METRICS_DIR = "metrics"

# =============================================================================
# Utility
# =============================================================================


def _safe_div(numerator: float, denominator: float) -> float:
    """Avoid division by zero; returns 0.0 when denominator is 0."""
    return numerator / denominator if denominator > 0 else 0.0


# =============================================================================
# 1. RETRIEVAL METRICS
#    Operate on (query_label, ranked_retrieved_labels) pairs.
#    Column names carry no prefix — the @k index in the output DataFrame
#    is what identifies them as retrieval metrics.
# =============================================================================


def retrieval_rank_metrics_at_k(
    y_true: np.ndarray,
    retrieved_labels: np.ndarray,
    k: int,
    suffix: str = "",
) -> Dict[str, float]:
    """
    Position-sensitive retrieval metrics via irmetrics.

    Computes Mean Reciprocal Rank (MRR) and Mean Average prec (MAP)
    over a batch of queries. These reward systems that place relevant
    items earlier in the ranked list.

    Args:
        y_true: Ground truth labels for each query (shape: [n_queries]).
        retrieved_labels: Labels of retrieved items (shape: [n_queries, n_retrieved]).
        k: Cutoff depth.
        suffix: Suffix appended to metric keys (e.g. "_p", "_n").

    Returns:
        Dict with keys ``mrr{suffix}`` and ``map{suffix}``.
    """
    return {
        f"mrr{suffix}": round(float(np.mean(topk.rr(y_true, retrieved_labels, k=k))), 4),
        f"map{suffix}": round(float(np.mean(topk.ap(y_true, retrieved_labels, k=k))), 4),
    }


def retrieval_hit_at_k(
    y_true: int,
    retrieved_labels: list,
    k: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Binary retrieval hit rate at k for a single query.

    Asks: "Does at least one item in the top-k share the query's label?"
    Returns 1.0 if yes, 0.0 if no.  The value is placed in the slot
    corresponding to the query's class; the other slot is None.

    NOTE: This is NOT classification sens/spec. It measures
    whether the retrieval system surfaced *any* same-label case, not
    whether a classifier predicted the correct label.

    Args:
        y_true: Ground truth label of the query (0 or 1).
        retrieved_labels: Ordered labels of retrieved candidates.
        k: Number of top results to consider.

    Returns:
        (hit_positive, hit_negative) — one will be None.
    """
    top_k = retrieved_labels[:k]
    if y_true == 1:
        return (1.0 if any(l == 1 for l in top_k) else 0.0, None)
    else:
        return (None, 1.0 if any(l == 0 for l in top_k) else 0.0)


def retrieval_prec_at_k(
    y_true: int,
    retrieved_labels: list,
    k: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Retrieval prec at k for a single query.

    Asks: "What fraction of the top-k items share the query's label?"

    NOTE: This is NOT classification PPV/NPV. It measures the purity
    of the retrieved set, not the trustworthiness of a prediction.

    Args:
        y_true: Ground truth label of the query (0 or 1).
        retrieved_labels: Ordered labels of retrieved candidates.
        k: Number of top results to consider.

    Returns:
        (prec_positive, prec_negative) — one will be None.
    """
    top_k = retrieved_labels[:k]
    if y_true == 1:
        return (sum(1 for l in top_k if l == 1) / k, None)
    else:
        return (None, sum(1 for l in top_k if l == 0) / k)


def retrieval_recall_at_k(
    y_true: int,
    retrieved_labels: list,
    k: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Retrieval recall at k for a single query.

    Asks: "Of all same-label items in the retrieved list, what fraction
    appear in the top-k?"

    NOTE: This is NOT classification sens/spec. It measures coverage of
    relevant items within the ranked list, not whether a classifier
    predicted the correct label.

    Args:
        y_true: Ground truth label of the query (0 or 1).
        retrieved_labels: Ordered labels of retrieved candidates.
        k: Number of top results to consider.

    Returns:
        (recall_positive, recall_negative) — one will be None.
    """
    top_k = retrieved_labels[:k]
    if y_true == 1:
        total = sum(1 for l in retrieved_labels if l == 1)
        return (_safe_div(sum(1 for l in top_k if l == 1), total), None)
    else:
        total = sum(1 for l in retrieved_labels if l == 0)
        return (None, _safe_div(sum(1 for l in top_k if l == 0), total))


def retrieval_f1_at_k(
    y_true: int,
    retrieved_labels: list,
    k: int,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Retrieval F1 at k for a single query.

    Harmonic mean of ``retrieval_hit_at_k`` and ``retrieval_prec_at_k``.

    Args:
        y_true: Ground truth label of the query (0 or 1).
        retrieved_labels: Ordered labels of retrieved candidates.
        k: Number of top results to consider.

    Returns:
        (f1_positive, f1_negative) — one will be None.
    """
    hit_p, hit_n = retrieval_hit_at_k(y_true, retrieved_labels, k)
    prec_p, prec_n = retrieval_prec_at_k(y_true, retrieved_labels, k)

    def _hm(a, b):
        if a is None or b is None:
            return None
        return _safe_div(2 * a * b, a + b)

    return (_hm(hit_p, prec_p), _hm(hit_n, prec_n))


# =============================================================================
# 2. CLASSIFICATION METRICS
#    Operate on (y_true_array, y_pred_array) — no ranked lists.
# =============================================================================


def confusion_matrix_counts(
    y_true: Union[list, np.ndarray, pd.Series],
    y_pred: Union[list, np.ndarray, pd.Series],
    metrics: dict = None,
) -> Dict[str, int]:
    """
    Compute the four cells of a binary confusion matrix.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        Dict with keys ``tp``, ``tn``, ``fp``, ``fn``.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    if isinstance(metrics, dict):
        metrics.update({"tp": tp, "tn": tn, "fp": fp, "fn": fn})

    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def classification_scores(
    y_true: Union[list, np.ndarray, pd.Series],
    y_pred: Union[list, np.ndarray, pd.Series],
    prefix: str = "",
) -> Dict[str, float]:
    """
    Compute the full suite of binary classification metrics.

    This function derives every metric from the confusion matrix, so each column name corresponds to exactly one well-defined quantity.

    Metrics returned (all prefixed with ``prefix`` if provided):
        sens          TP / (TP + FN)   — positive recall / hit rate
        spec          TN / (TN + FP)   — negative recall / hit rate
        ppv                  TP / (TP + FP)   — positive predictive value
        npv                  TN / (TN + FN)   — negative predictive value
        cls_f1_p      2·(PPV·Sens) / (PPV+Sens)
        cls_f1_n      2·(NPV·Spec) / (NPV+Spec)
        macro_f1         (cls_f1_p + cls_f1_n) / 2
        bal_acc    (sens + spec) / 2
        mcc                  Matthews correlation coefficient
        tp, tn, fp, fn       Raw confusion matrix counts

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.
        prefix: Optional string prepended to every key.

    Returns:
        Dict of metric name → rounded value.
    """
    cm = confusion_matrix_counts(y_true, y_pred)
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]

    sens = _safe_div(tp, tp + fn)
    spec = _safe_div(tn, tn + fp)
    ppv = _safe_div(tp, tp + fp)
    npv = _safe_div(tn, tn + fn)

    f1_pos = _safe_div(2 * ppv * sens, ppv + sens)
    f1_neg = _safe_div(2 * npv * spec, npv + spec)
    macro_f1 = (f1_pos + f1_neg) / 2
    balanced_acc = (sens + spec) / 2

    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = _safe_div(tp * tn - fp * fn, mcc_denom)

    p = prefix
    return {
        f"{p}sens": round(sens, 4),
        f"{p}spec": round(spec, 4),
        f"{p}ppv": round(ppv, 4),
        f"{p}npv": round(npv, 4),
        f"{p}cls_f1_p": round(f1_pos, 4),
        f"{p}cls_f1_n": round(f1_neg, 4),
        f"{p}macro_f1": round(macro_f1, 4),
        f"{p}bal_acc": round(balanced_acc, 4),
        f"{p}mcc": round(mcc, 4),
        f"{p}tp": tp,
        f"{p}tn": tn,
        f"{p}fp": fp,
        f"{p}fn": fn,
    }


# =============================================================================
# 3. RRG (Report Retrieval Generation) METRICS
# =============================================================================


# def calculate_rrg_metrics(
#     run_id: str,
#     # metrics_data: pd.DataFrame,
#     results: dict,
#     k_values: list = K_VALUES,
#     log: bool = False,
# ):
#     """
#     Calculate RRG metrics (RadGraph-F1 and RaTEScore) for retrieved reports.
#     """
#     from RaTEScore import RaTEScore
#     from radgraph import F1RadGraph

#     f1radgraph = F1RadGraph(model_type="modern-radgraph-xl", reward_level="all")
#     ratescore = RaTEScore()

#     all_rate_scores = []
#     all_radgraph_scores = []
#     metrics_data = {run_id: {k: {} for k in k_values}}
#     for _, data in track(results.items(), description="Computing RRG metrics", total=len(results)):
#         candidates = data["retrieved_reports"]
#         references = [data["reports"]] * len(candidates)
#         cands = [extract_sections(c)["cleaned_report"] for c in candidates]
#         refs = [extract_sections(r)["cleaned_report"] for r in references]
#         mean_reward, _, _, _ = f1radgraph(hyps=cands, refs=refs)

#         _, rg_er, _ = mean_reward
#         rscore = ratescore.compute_score(cands, refs)
#         all_rate_scores.append(rscore)
#         all_radgraph_scores.append(rg_er)

#     # TODO: fix later if wrong (likely)
#     for k in k_values:
#         mean_rate = float(np.mean([np.mean(scores[:k]) for scores in all_rate_scores]))
#         metrics_data[run_id][k]["rate"] = round(mean_rate, 4)
#         if log:
#             MLflow.log_metrics({"rate": metrics_data[run_id][k]["rate"]}, step=k, run_id=run_id)
#         mean_radgraph = float(np.mean([np.mean(scores[:k]) for scores in all_radgraph_scores]))
#         metrics_data[run_id][k]["radgraph"] = round(mean_radgraph, 4)
#         if log:
#             MLflow.log_metrics(
#                 {"radgraph": metrics_data[run_id][k]["radgraph"]}, step=k, run_id=run_id
#             )
#     return metrics_data


# =============================================================================
# 4. REPORT GENERATORS
# =============================================================================


def generate_retrieval_report(
    args: DictConfig = DictConfig({}),
    results: dict = {},
    log: bool = False,
    k_values: list = K_VALUES,
    **kwargs,
) -> pd.DataFrame:
    """
    Generate evaluation report for different k values.

    Retrieval metrics (hit, prec, f1, mrr, map) are computed at
    every k.  Classification metrics (sens, spec, ppv,
    npv, cls_f1, bal_acc, mcc) are computed at k=1 only,
    where the top-1 retrieved label serves as an implicit prediction.
    These columns are NaN at other k values, which is itself the
    structural marker that distinguishes the two metric families.
    """
    model_name: str = kwargs.get("model_name", args.get("model_name", "model"))
    positive_class: str = kwargs.get("positive_class", args.get("positive_class", "Pneumonia"))
    retrieval_modality: str = kwargs.get(
        "retrieval_modality", args.get("retrieval_modality", "vlm")
    )
    if log:
        MLflow.log_param("k_values", k_values)
        MLflow.log_param("num_samples", len(results))

    pred_data = pd.DataFrame.from_records(results).T
    metrics_data = []

    for k in k_values:
        k_metric = {}

        # ----- Position-sensitive metrics (MRR, MAP) via irmetrics -----
        neg_pred_data = pred_data[pred_data["y_true"] == 0]
        neg_true = neg_pred_data["y_true"].to_numpy(np.int32)
        neg_pred = np.vstack(neg_pred_data["retrieved_labels"]).astype(np.int32)
        neg_rank = retrieval_rank_metrics_at_k(neg_true, neg_pred, suffix="_n", k=k)
        k_metric.update(neg_rank)

        pos_pred_data = pred_data[pred_data["y_true"] == 1]
        pos_true = pos_pred_data["y_true"].to_numpy(np.int32)
        pos_pred = np.vstack(pos_pred_data["retrieved_labels"]).astype(np.int32)
        pos_rank = retrieval_rank_metrics_at_k(pos_true, pos_pred, suffix="_p", k=k)
        k_metric.update(pos_rank)

        k_metric["mrr"] = round((neg_rank["mrr_n"] + pos_rank["mrr_p"]) / 2, 4)
        k_metric["map"] = round((neg_rank["map_n"] + pos_rank["map_p"]) / 2, 4)

        # ----- Set-based retrieval metrics (hit, prec, F1) -----
        hits_p, hits_n = [], []
        precs_p, precs_n = [], []
        f1s_p, f1s_n = [], []
        recs_p, recs_n = [], []

        for _, data in results.items():
            y_true = data["y_true"]
            retrieved_labels = data["retrieved_labels"]

            hp, hn = retrieval_hit_at_k(y_true, retrieved_labels, k)
            pp, pn = retrieval_prec_at_k(y_true, retrieved_labels, k)
            fp, fn = retrieval_f1_at_k(y_true, retrieved_labels, k)
            rp, rn = retrieval_recall_at_k(y_true, retrieved_labels, k)

            if hp is not None:
                hits_p.append(hp)
            if hn is not None:
                hits_n.append(hn)
            if pp is not None:
                precs_p.append(pp)
            if pn is not None:
                precs_n.append(pn)
            if fp is not None:
                f1s_p.append(fp)
            if fn is not None:
                f1s_n.append(fn)
            if rp is not None:
                recs_p.append(rp)
            if rn is not None:
                recs_n.append(rn)

        mean_hit_p = np.mean(hits_p) if hits_p else 0.0
        mean_hit_n = np.mean(hits_n) if hits_n else 0.0
        mean_prec_p = np.mean(precs_p) if precs_p else 0.0
        mean_prec_n = np.mean(precs_n) if precs_n else 0.0
        mean_f1_p = np.mean(f1s_p) if f1s_p else 0.0
        mean_f1_n = np.mean(f1s_n) if f1s_n else 0.0
        mean_rec_p = np.mean(recs_p) if recs_p else 0.0
        mean_rec_n = np.mean(recs_n) if recs_n else 0.0

        k_metric.update(
            {
                "hit": round(float((mean_hit_p + mean_hit_n) / 2), 4),
                "hit_p": round(float(mean_hit_p), 4),
                "hit_n": round(float(mean_hit_n), 4),
                "rec": round(float((mean_rec_p + mean_rec_n) / 2), 4),
                "rec_p": round(float(mean_rec_p), 4),
                "rec_n": round(float(mean_rec_n), 4),
                "prec": round(float((mean_prec_p + mean_prec_n) / 2), 4),
                "prec_p": round(float(mean_prec_p), 4),
                "prec_n": round(float(mean_prec_n), 4),
                "f1": round(float((mean_f1_p + mean_f1_n) / 2), 4),
                "f1_p": round(float(mean_f1_p), 4),
                "f1_n": round(float(mean_f1_n), 4),
            }
        )

        # ----- Classification metrics (k=1 only) -----
        if k == 1:
            y_true_all = [int(data["y_true"]) for data in results.values()]
            y_pred_k1 = [
                int(data["retrieved_labels"][0]) if data["retrieved_labels"] else 0
                for data in results.values()
            ]
            cls_metrics = classification_scores(y_true_all, y_pred_k1)
            MLflow.log_metrics(cls_metrics, step=k) if log else None

        if log:
            MLflow.log_metrics(k_metric, step=k)

        k_metric["k"] = k
        metrics_data.append(k_metric)

    # Build DataFrame and display
    cls_metrics_df = pd.DataFrame(cls_metrics, index=[f"{model_name}_{retrieval_modality}"])
    cls_metrics_df_string = tabulate(
        cls_metrics_df, headers="keys", showindex=True, tablefmt="fancy_grid"
    )
    cls_metrics_file = os.path.join(METRICS_DIR, f"{model_name}_{retrieval_modality}_cls_k=1.csv")
    cls_metrics_df.to_csv(cls_metrics_file, index_label="model")

    ret_metrics_df = pd.DataFrame(metrics_data).set_index("k")
    ret_metrics_df = ret_metrics_df.sort_index()
    ret_metrics_df_string = tabulate(
        ret_metrics_df.T, headers="keys", showindex=True, tablefmt="fancy_grid"
    )
    ret_metrics_file = os.path.join(
        METRICS_DIR, f"{model_name}_{retrieval_modality}_retrieval_report.csv"
    )
    ret_metrics_df.to_csv(ret_metrics_file, index_label="k")

    description = (
        f"# Retrieval and TopK@1 Performance\n\n"
        + f"```\n{cls_metrics_df_string}\n"
        + f"{ret_metrics_df_string}\n```"
    )

    logger.info(f"✓ Classification metrics saved to: {cls_metrics_file}")
    logger.info(f"✓ Retrieval metrics saved to: {ret_metrics_file}")
    # content = description + "\n"
    console.print(Markdown(description))
    if log:
        MLflow.set_tag("mlflow.note.content", f"{description}")

    return ret_metrics_df


def generate_classification_report(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    y_score: Union[np.ndarray, pd.Series] = None,
    target_names: list = ["No Pneumonia", "Pneumonia"],
    log: bool = False,
    state: TrainerState = None,
    split: str = "",
    metrics_dir: str = METRICS_DIR,
) -> Dict[str, float]:
    """
    Generate a full classification report with correctly labeled metrics.

    Uses ``classification_scores`` for sens, spec, PPV,
    NPV, F1, balanced acc, and MCC.
    """
    logger.info("Generating classification report...")
    positive_class: str = target_names[1]
    if positive_class == "No Finding":
        negative_class: str = positive_class
        positive_class = "Finding"
        target_names = [positive_class, negative_class]
    else:
        negative_class: str = target_names[0]
    curr_step = "000000" if state is None else str(state.global_step).zfill(6)

    labels = list(range(len(target_names)))
    cr: dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        output_dict=True,
        target_names=target_names,
        zero_division=0.0,
    )

    cls = classification_scores(y_true, y_pred)

    console.print(Markdown("## Summary Report:"))
    split_prefix = f"{split}_" if split else ""
    summary_dict = {
        f"{split_prefix}sens": cls["sens"],
        f"{split_prefix}spec": cls["spec"],
        f"{split_prefix}ppv": cls["ppv"],
        f"{split_prefix}npv": cls["npv"],
        f"{split_prefix}cls_f1_p": cls["cls_f1_p"],
        f"{split_prefix}cls_f1_n": cls["cls_f1_n"],
        f"{split_prefix}macro_f1": cls["macro_f1"],
        f"{split_prefix}bal_acc": cls["bal_acc"],
        f"{split_prefix}mcc": cls["mcc"],
        # f"{split_prefix}acc": round(cr["acc"], 4),
    }
    index = curr_step
    summary = pd.DataFrame(summary_dict, index=[index])
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
        summary_path = os.path.join(metrics_dir, "classification_summary.csv")
        logger.debug(f"Saving classification summary to '{summary_path}'")
        summary.to_csv(summary_path)
        cfm_df = pd.DataFrame(cfm, index=target_names, columns=target_names)
        cfm_df.index.name = "labels"
        cfm_path = os.path.join(metrics_dir, "confusion_matrix.csv")
        logger.debug(f"Saving confusion matrix to '{cfm_path}'")
        cfm_df.to_csv(cfm_path)

        if state:
            MLflow.log_artifact(metrics_dir, metrics_dir)
        else:
            tagged = {
                f"{k}_{positive_class.lower().replace(' ', '_')}": v
                for k, v in summary_dict.items()
            }
            MLflow.log_metrics(tagged, step=int(curr_step))

    return summary_dict


# =============================================================================
# 5. BACKWARD COMPATIBILITY ALIASES
#    Preserve old function signatures so existing call sites continue
#    to work. Each alias emits a deprecation warning on first use.
# =============================================================================


def _deprecation_alias(old_name, new_func):
    """Create a wrapper that warns once then delegates."""
    _warned = [False]

    def wrapper(*args, **kwargs):
        if not _warned[0]:
            import warnings as _w

            _w.warn(
                f"`{old_name}` is deprecated. Use `{new_func.__name__}` instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            _warned[0] = True
        return new_func(*args, **kwargs)

    wrapper.__name__ = old_name
    wrapper.__doc__ = f"DEPRECATED — use ``{new_func.__name__}`` instead."
    return wrapper


recall_at_k = _deprecation_alias("recall_at_k", retrieval_hit_at_k)
prec_at_k = _deprecation_alias("prec_at_k", retrieval_prec_at_k)
f1_at_k = _deprecation_alias("f1_at_k", retrieval_f1_at_k)
retrieval_at_k = _deprecation_alias("retrieval_at_k", retrieval_rank_metrics_at_k)
