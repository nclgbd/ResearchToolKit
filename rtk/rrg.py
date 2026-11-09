"""Report-level clinical similarity: RadGraph-F1 and RaTEScore, as reusable matrices.

Reviewers of the retrieval work objected that relevance was defined as "identical
pathology label", which saturates on negative classes.  RadGraph-F1 and RaTEScore answer
that objection directly: both compare two radiology *reports*, giving a graded,
label-free notion of how clinically similar a retrieved case is to the query.

Both metrics are normally used report-generation style, one candidate against one
reference.  Used that way over a retrieval benchmark they are ruinously expensive, since
the same report is re-parsed once per pair.  Both, however, factor cleanly:

* ``RadGraph`` annotates a report into entities and relations; ``compute_reward`` then
  compares two annotations with pure set arithmetic.
* ``RaTEScore`` runs NER over a report and embeds its entities; ``compute`` then compares
  two entity sets with a weighted nearest-neighbour match.

So the neural work is per *report*, not per pair.  For the ICLR 2027 benchmark the query
set and the corpus are the same 482 MIMIC-CXR studies holding 434 distinct reports, so a
single 434x434 matrix per metric serves every run, pathology and table, and scoring a run
afterwards is a table lookup.

This module builds and loads those matrices.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RADGRAPH_MODEL = "radgraph-xl"
RADGRAPH_F1_FILE = "radgraph_f1.npy"
RATESCORE_FILE = "ratescore.npy"
REPORTS_FILE = "reports.parquet"
RAW_INDEX_FILE = "raw_report_index.parquet"
ANNOTATIONS_FILE = "radgraph_annotations.json"
META_FILE = "meta.json"

# ``compute_reward(..., "all")`` returns (simple, partial, complete) rewards. The partial
# reward is the entity-and-relation F1 that the literature reports as "RadGraph-F1".
RADGRAPH_REWARD_INDEX = 1


def report_key(report: str) -> str:
    """Stable id for a report, used to join retrieved text to matrix columns."""
    return hashlib.sha1(report.strip().encode("utf-8")).hexdigest()


@dataclass
class ReportSimilarity:
    """A pair of report-by-report similarity matrices plus their text index."""

    keys: list[str]
    radgraph_f1: np.ndarray
    ratescore: np.ndarray
    meta: dict
    dicom_index: dict[str, int] | None = None
    raw_index: dict[str, int] | None = None

    def __post_init__(self) -> None:
        self._index = {key: i for i, key in enumerate(self.keys)}
        self.dicom_index = self.dicom_index or {}
        self.raw_index = self.raw_index or {}
        self._corpus: np.ndarray | None = None
        self._ideal_cache: dict = {}

    @property
    def n(self) -> int:
        return len(self.keys)

    def index_of(self, cleaned_report: str) -> int | None:
        """Matrix index of a report that has already been cleaned."""
        return self._index.get(report_key(cleaned_report))

    def index_of_raw(self, report: str) -> int | None:
        """Matrix index of a raw MIMIC report, as stored in a run's results jsonl.

        Hashing the raw text avoids re-running section extraction for every retrieved
        candidate of every run; the corpus is closed, so every candidate a run can return
        is already in the index. Falls back to cleaning the text for anything unseen.
        """
        idx = self.raw_index.get(report_key(report))
        if idx is not None:
            return idx
        from rtk.datasets import extract_sections

        return self.index_of(extract_sections(report)["cleaned_report"])

    def index_of_dicom(self, dicom_id: str) -> int | None:
        """Matrix index of the report belonging to a study."""
        return self.dicom_index.get(dicom_id)

    def matrix(self, metric: str) -> np.ndarray:
        return self.radgraph_f1 if metric == "radgraph" else self.ratescore

    def lookup(self, metric: str, query_idx: int, candidate_idx: int) -> float:
        return float(self.matrix(metric)[query_idx, candidate_idx])

    def corpus_indices(self) -> np.ndarray:
        """Matrix index of every study in the corpus, one entry per study.

        Reports repeat across studies, so this is deliberately a multiset: a report
        shared by two studies can be retrieved twice and must be able to appear twice in
        an ideal ranking.
        """
        if self._corpus is None:
            self._corpus = np.array(
                [self.dicom_index[d] for d in sorted(self.dicom_index)], dtype=np.int32
            )
        return self._corpus

    def ideal_gains(self, metric: str, dicom_id: str, k: int) -> list[float]:
        """The k highest similarities any ranking of the corpus could have put on top.

        This is the oracle a graded nDCG should be normalised against: the best the
        retriever could possibly have done for this query, had it ranked the entire
        corpus by clinical similarity. The query's own study is excluded, matching the
        leave-one-out protocol the runs use.
        """
        query_idx = self.index_of_dicom(dicom_id)
        if query_idx is None:
            return []

        cache_key = (metric, dicom_id)
        if cache_key not in self._ideal_cache:
            row = self.matrix(metric)[query_idx]
            candidates = [
                self.dicom_index[d] for d in sorted(self.dicom_index) if d != dicom_id
            ]
            gains = np.sort(row[np.array(candidates, dtype=np.int32)])[::-1]
            self._ideal_cache[cache_key] = gains
        return [float(g) for g in self._ideal_cache[cache_key][:k]]

    @classmethod
    def load(cls, directory: str | Path) -> "ReportSimilarity":
        import pandas as pd

        directory = Path(directory)
        reports = pd.read_parquet(directory / REPORTS_FILE)
        reports = reports.sort_values("report_idx")
        meta = json.loads((directory / META_FILE).read_text())

        dicom_index: dict[str, int] = {}
        for idx, dicom_ids in zip(reports["report_idx"], reports["dicom_ids"]):
            for dicom_id in dicom_ids:
                dicom_index[dicom_id] = int(idx)

        raw_index: dict[str, int] = {}
        raw_path = directory / RAW_INDEX_FILE
        if raw_path.exists():
            raw = pd.read_parquet(raw_path)
            raw_index = dict(zip(raw["raw_sha1"], raw["report_idx"].astype(int)))

        return cls(
            keys=reports["sha1"].tolist(),
            radgraph_f1=np.load(directory / RADGRAPH_F1_FILE),
            ratescore=np.load(directory / RATESCORE_FILE),
            meta=meta,
            dicom_index=dicom_index,
            raw_index=raw_index,
        )


# --------------------------------------------------------------------------- #
# RadGraph
# --------------------------------------------------------------------------- #
def annotate_radgraph(reports: list[str], model_type: str = RADGRAPH_MODEL, batch_size: int = 64) -> list[dict]:
    """Annotate each report once with RadGraph, returning one annotation per report."""
    from radgraph import RadGraph

    radgraph = RadGraph(model_type=model_type)
    annotations: list[dict] = []
    for start in range(0, len(reports), batch_size):
        chunk = reports[start : start + batch_size]
        result = radgraph(chunk)
        annotations.extend(result[str(i)] for i in range(len(chunk)))
    return annotations


def radgraph_matrix(annotations: list[dict], progress=None) -> np.ndarray:
    """Pairwise RadGraph-F1 over pre-computed annotations.

    The reward is symmetric (it is an F1 of two annotation sets), so only the upper
    triangle is evaluated and then mirrored.
    """
    from radgraph.rewards import compute_reward

    n = len(annotations)
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        out[i, i] = float(compute_reward(annotations[i], annotations[i], "all")[RADGRAPH_REWARD_INDEX])
        for j in range(i + 1, n):
            value = float(compute_reward(annotations[i], annotations[j], "all")[RADGRAPH_REWARD_INDEX])
            out[i, j] = out[j, i] = value
        if progress is not None:
            progress(i + 1, n)
    return out


# --------------------------------------------------------------------------- #
# RaTEScore
# --------------------------------------------------------------------------- #
def embed_ratescore(reports: list[str], scorer, batch_size: int = 32):
    """Run RaTEScore's NER and entity encoder once per report.

    Returns a list of ``(entity_embeddings, entity_types)`` exactly as
    ``RaTEScore.compute_score`` would compute them for that report on its own, so the
    pairwise step can be handed straight to the package's own scoring function. This is
    the whole optimisation: the neural work happens once per report instead of once per
    pair, while the arithmetic stays the package's.
    """
    from RaTEScore.score import process_embedding, run_ner

    pairs = run_ner(
        reports,
        scorer.idx2label,
        scorer.tokenizer,
        scorer.model,
        scorer.device,
        batch_size,
    )
    return [
        process_embedding(pair, scorer.eval_tokenizer, scorer.eval_model, scorer.device)
        for pair in pairs
    ]


def ratescore_pair(a, b, affinity) -> float:
    """RaTEScore between two reports, from their cached entity embeddings.

    Delegates the arithmetic to ``RaTEScore.utils.compute``. A vectorised
    reimplementation was tried and rejected: reports repeat entity mentions, so exact
    ties in the nearest-entity search are common, and ``torch.topk`` breaks them
    differently depending on tensor rank and device. A tie broken the other way selects a
    different entity *type*, changes the affinity weight, and moves the score by up to
    0.01 -- small, but not something to introduce silently into published numbers.
    """
    from RaTEScore.utils import compute

    (gt_embeds, gt_types), (pred_embeds, pred_types) = a, b
    if len(gt_embeds) == 0 or len(pred_embeds) == 0:
        # Matches RaTEScore.compute_score: an empty entity set scores a flat 0.5.
        return 0.5
    precision = compute(gt_embeds, pred_embeds, gt_types, pred_types, affinity)
    recall = compute(pred_embeds, gt_embeds, pred_types, gt_types, affinity)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def ratescore_matrix(embedded, affinity, progress=None) -> np.ndarray:
    """Pairwise RaTEScore over cached entity embeddings; symmetric, upper triangle only."""
    n = len(embedded)
    out = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        out[i, i] = ratescore_pair(embedded[i], embedded[i], affinity)
        for j in range(i + 1, n):
            value = ratescore_pair(embedded[i], embedded[j], affinity)
            out[i, j] = out[j, i] = value
        if progress is not None:
            progress(i + 1, n)
    return out
