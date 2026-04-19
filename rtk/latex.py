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


# Model display names and colors
MODEL_COLORS = {
    "biomed-clip": "#0072B2",  # Blue
    "med-siglip": "#D55E00",  # Vermillion
    "xray-siglip": "#009E73",  # Bluish green
    "medclip-vit": "#E9566A",  # Sky blue
    "medclip-resnet": "#F0E442",  # Yellow
    "radir": "#CC79A7",  # Reddish purple
}

MODEL_LABELS = {
    "biomed-clip": "BiomedCLIP",
    "med-siglip": "MedSigLIP",
    "xray-siglip": "XraySigLIP",
    "medclip-vit": "MedCLIP-ViT",
    "medclip-resnet": "MedCLIP-ResNet",
    "radir": "RadIR",
}

# Modality styles
MODALITY_STYLES = {
    "text": {"hatch": None, "alpha": 1.0, "label": "Text"},
    "image": {"hatch": "//", "alpha": 0.6, "label": "Image"},
}

PATHOLOGIES_DICT = {
    "Atelectasis": "Atel",
    "Cardiomegaly": "Card",
    "Consolidation": "Cons",
    "Edema": "Edem",
    "Effusion": "Effu",
    "Pneumonia": "Pneu",
}
RETRIEVAL_MODALITY_DICT = {
    "text": "Txt",
    "image": "Img",
    "full": "Full",
}
