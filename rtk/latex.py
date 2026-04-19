import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


def prepare_layout():
    pd.set_option("display.float_format", lambda x: f"{x:.3f}".replace("0.", "."))
    plt.style.use("seaborn-v0_8-whitegrid")

    # Set up publication-quality style
    plt.rcParams.update(
        {
            "axes.labelsize": 14,
            "axes.linewidth": 1.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.titlesize": 21,
            "figure.dpi": 300,
            "font.family": "serif",
            "font.size": 20,
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


metrics = ["recall_p", "precision_p", "f1_p", "mrr_p"]
display_metrics = ["Rec@k", "Prec@k", "F1@k", "mRR@k"]

PATHOLOGIES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion", "Pneumonia"]

# Publication-quality color palette (colorblind-friendly)
PATHOLOGY_COLORS = {
    "Atelectasis": "#009E73",  # Bluish green
    "Cardiomegaly": "#E9566A",  # Sky blue
    "Consolidation": "#F0E442",  # Yellow
    "Edema": "#CC79A7",  # Reddish purple
    "Effusion": "#D55E00",  # Vermillion
    "Pneumonia": "#0072B2",  # Blue
}

# Line styles for different metrics
METRIC_STYLES = {
    "recall": {"linestyle": "-", "marker": "o", "label": "Rec (Macro)"},
    "recall_p": {"linestyle": "-", "marker": "o", "label": "Rec (P)"},
    "recall_n": {"linestyle": "-", "marker": "o", "label": "Rec (N)"},
    "precision": {"linestyle": "--", "marker": "s", "label": "Prec (Macro)"},
    "precision_p": {"linestyle": "--", "marker": "s", "label": "Prec (P)"},
    "precision_n": {"linestyle": "--", "marker": "s", "label": "Prec (N)"},
    "f1": {"linestyle": ":", "marker": "^", "label": "F1 (Macro)"},
    "f1_p": {"linestyle": ":", "marker": "^", "label": "F1 (P)"},
    "f1_n": {"linestyle": ":", "marker": "^", "label": "F1 (N)"},
    "mrr": {"linestyle": "-.", "marker": "D", "label": "mRR (Macro)"},
    "mrr_p": {"linestyle": "-.", "marker": "D", "label": "mRR (P)"},
    "mrr_n": {"linestyle": "-.", "marker": "D", "label": "mRR (N)"},
    "map": {"linestyle": (0, (3, 1, 1, 1)), "marker": "v", "label": "mAP (Macro)"},
    "map_p": {"linestyle": (0, (3, 1, 1, 1)), "marker": "v", "label": "mAP (P)"},
    "map_n": {"linestyle": (0, (3, 1, 1, 1)), "marker": "v", "label": "mAP (N)"},
}
