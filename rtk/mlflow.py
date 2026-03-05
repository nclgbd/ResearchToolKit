#
import os
import pandas as pd
import tempfile
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# mlflow
import mlflow
from mlflow.entities import RunData
from mlflow.data.huggingface_dataset import *

# huggingface
from datasets import DatasetDict

# rtk
from rtk import console
from rtk.utils import get_logger

CACHE_DIR = "runs"
logger = get_logger(__name__, console=console)
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))


def load_runs(csv: str, return_runs: bool = False) -> pd.DataFrame:
    # Load the CSV file into a DataFrame
    runs = pd.read_csv(csv)
    finished_runs = runs[runs["Status"] == "FINISHED"]
    run_ids = finished_runs["Run ID"].tolist()
    print(f"Loaded {len(run_ids)} finished runs from '{csv}'")

    ret = {"run_ids": run_ids}
    if return_runs:
        ret["runs"] = runs
    return ret


class ExperimentCollection:

    def __init__(self, run_ids: list, name: str = ""):
        self.name = name if name else "Experiment"
        self.run_ids: list = run_ids
        self.run_data = {run_id: mlflow.get_run(run_id).data for run_id in run_ids}
        self.artifacts = {
            run_id: {"report": None, "predictions": None} for run_id in self.run_data.keys()
        }
        self.artifacts = self._gather_artifacts()

    def __len__(self):
        return len(self.run_ids)

    def _create_storage_dst_path(self, cache_dir: str, **params) -> str:
        token_modality: str = params["token_modality"]
        positive_class: str = params["positive_class"]
        report_baseline: bool = eval(params["report_baseline"])
        use_context: bool = eval(params.get("use_context", "True"))
        use_filter: bool = eval(params.get("filter_sequence", "False"))
        use_self_dataset: bool = eval(params.get("uses_own_dataset", "True"))
        corpus_split: str = params.get("split", "test" if use_self_dataset else "train")

        dst_path = ""
        if report_baseline == True:
            dst_path = os.path.join(
                cache_dir,
                corpus_split,
                "stage1",
                positive_class,
            )
        else:
            lambda_value = str(eval(params["sinkhorn"])["lmb"])

            dst_path = os.path.join(
                cache_dir,
                corpus_split,
                "stage2",
                positive_class,
                token_modality,
                lambda_value,
            )
        if use_context:
            dst_path = os.path.join(dst_path, "context")
            if "prompt" in params:
                prompt_params = eval(params["prompt"])
                dst_path = os.path.join(dst_path, prompt_params["name"], prompt_params["label"])

        if use_filter:
            dst_path = os.path.join(dst_path, "token_filter")

        return dst_path

    def _gather_artifacts(self) -> dict:
        with tempfile.TemporaryDirectory() as cache_dir:
            logger.debug(f"Created temporary directory: '{cache_dir}'")
            for run_id in self.run_ids:
                run = mlflow.get_run(run_id)
                params = run.data.params
                model_name = params["model_name"]
                modality = params["retrieval_modality"]
                dst_path = self._create_storage_dst_path(cache_dir=cache_dir, **params)
                os.makedirs(dst_path, exist_ok=True)

                # Download artifacts
                logger.debug(f"Downloading artifacts for run_id: '{run_id}' to '{dst_path}'")
                report_file = f"{model_name}_{modality}_retrieval_report.csv"
                report_file = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=f"outputs/metrics/{report_file}", dst_path=dst_path
                )
                report_df = pd.read_csv(report_file).set_index("k")
                report_df = report_df.T
                report_df.index.name = "metric"
                self.artifacts[run_id]["report"] = report_df

                pred_file = f"{model_name}_{modality}_retrieval_results.jsonl"
                pred_file = mlflow.artifacts.download_artifacts(
                    run_id=run_id, artifact_path=f"outputs/metrics/{pred_file}", dst_path=dst_path
                )
                pred_jsonl = pd.read_json(pred_file, lines=True)
                self.artifacts[run_id]["predictions"] = pred_jsonl.set_index("dicom_id")

        return self.artifacts


def create_experiment_collection(run_ids: list, name: str = "") -> ExperimentCollection:
    return ExperimentCollection(run_ids=run_ids, name=name)


def gather_reruns(runs: pd.DataFrame):

    reruns = runs[runs["Status"] == "FAILED"]
    run_ids = reruns["Run ID"].tolist()
    if run_ids:
        for run_id in run_ids:
            run = mlflow.get_run(run_id)
            params = run.data.params

            rerun_params = {run_id: params}
            rerun_bash_command = (
                f"python scripts/retrieval_vlm.py -m positive_class={params['positive_class']} "
                f"model_name={params['model_name']} "
                f"retrieval_modality={params['retrieval_modality']} "
                f"token_modality={params['token_modality']} "
                f"report_baseline={params['report_baseline']}"
            )
            print(rerun_bash_command)

            with open("rerun_failed_runs.sh", "a") as f:
                f.write(rerun_bash_command + "\n")
    else:
        print("No failed runs to rerun.")


def prepare_mlflow(args: DictConfig, **kwargs):
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    logger.debug(f"Setting MLflow tracking URI to: {tracking_uri}")
    # if "azureml" in tracking_uri:
    #     from rtk.azure import login

    #     azml = login()
    mlflow.set_tracking_uri(tracking_uri)
    experiment_name = kwargs.get(
        "experiment_name", args.get("experiment_name", HydraConfig.get().job.name)
    )

    postfix = "-test" if args.get("dry_run", True) else ""
    experiment_name = f"{experiment_name}{postfix}"
    mlflow.set_experiment(experiment_name)

    mlflow_args = {
        "experiment_name": experiment_name,
        "tags": args.get("tags", {}),
    }
    return mlflow_args
