import mlflow
import os
from dotenv import load_dotenv
from huggingface_hub import create_branch, delete_branch
from omegaconf import DictConfig, OmegaConf
from rich import pretty, print
from rich.console import Console

console = Console()
pretty.install()
load_dotenv("../.env")
console.clear()
dry_run = True
suffix = "-test" if dry_run else ""
experiment_name = f"siglip-finetune{suffix}"
repo_id = f"nclgbd/medsiglip-448-pneumonia-finetune{suffix}"
exclude_branches = [
    "main",
    "1fc60c8ddcdc490ea2b1054059ad12b3",
    "98603c6b894040d7b963673969974df3",
    "2551070f24a3411fa6af17291e862e13",
]
repo_type = "model"


def gather_runs_from_experiment(experiment_name: str) -> int:
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment '{experiment_name}' does not exist.")
    experiment_id = experiment.experiment_id
    runs = mlflow.search_runs(experiment_ids=[experiment_id])
    branches_to_delete = runs["run_id"].tolist()
    branches_to_delete = list(
        filter(lambda b: b not in exclude_branches, branches_to_delete)
    )
    return branches_to_delete


def delete_branches(
    repo_id: str, branches_to_delete: list[str], repo_type: str = "model"
):
    total = len(branches_to_delete)
    for branch in branches_to_delete:
        try:
            if dry_run:
                import time

                time.sleep(1)
                console.log(f"Dry run: would delete branch '{branch}'")
            else:
                delete_branch(repo_id=repo_id, repo_type=repo_type, branch=branch)
                mlflow.delete_run(branch)
                console.log(f"Deleted branch: '{branch}'")
        except Exception as e:
            console.log(f"Could not delete branch '{branch}': {e}")

    total -= 1


if __name__ == "__main__":
    branches_to_delete = gather_runs_from_experiment(experiment_name)
    num_deleted = delete_branches(repo_id, branches_to_delete, repo_type)
    console.log(f"Deleted {num_deleted} branches.")
