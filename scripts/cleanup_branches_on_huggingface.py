import argparse
import mlflow
import os
import textwrap
from rich import pretty, print
from rich_argparse import RichHelpFormatter
from rich.console import Console

# 🤗
from huggingface_hub import delete_branch, list_repo_refs


# rtk
from rtk.utils import intro, namespace_to_configuration

console = Console()
pretty.install()
console.clear()
exclude_branches = [
    "main",
    "1fc60c8ddcdc490ea2b1054059ad12b3",
    "98603c6b894040d7b963673969974df3",
    "2551070f24a3411fa6af17291e862e13",
]


def gather_runs_from_experiment(experiment_name: str) -> list[str]:
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
    repo_id: str, branches_to_delete: list[str], repo_type: str = "model", dry_run=True
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

    console.log(f"Deleted {total} branches.")


def main(args: argparse.Namespace):
    repo_id: str = args.repo_id
    repo_type: str = args.repo_type
    experiment_name: str = args.experiment_name
    branches_to_delete = gather_runs_from_experiment(experiment_name)
    delete_branches(repo_id, branches_to_delete, repo_type)


if __name__ == "__main__":
    description = textwrap.dedent(
        """
        Utility script for mass deleting dangling branches from 🤗 (HuggingFace). Partially automated using mlflow, as the runIDs are linked to a respective branch within 🤗.
        """
    )
    argparser = argparse.ArgumentParser(
        description=description, formatter_class=RichHelpFormatter
    )
    argparser.add_argument("-d", "--dry-run", default=True)
    argparser.add_argument("-e", "--experiment-name", default="default")
    argparser.add_argument("-r", "--repo-id", default=str())
    argparser.add_argument(
        "-t", "--repo-type", default=str(), choices=["model", "dataset"]
    )
    args = argparser.parse_args()
    config = namespace_to_configuration(args)
    intro(config, title="MLflow and 🤗 Branch Deleter")
    main(args)
