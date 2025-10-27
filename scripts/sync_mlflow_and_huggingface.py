import argparse
import mlflow
import os
import textwrap
import time

# rich
from rich import pretty, print
from rich_argparse import RichHelpFormatter
from tqdm.rich import tqdm

# 🤗
from huggingface_hub import delete_branch, list_repo_refs

# rtk
from rtk.utils import intro, namespace_to_configuration, get_console

console = get_console()
MAIN_BRANCH_NAME = "[bold][red]'main'[/red][/bold]"
pretty.install()
exclude_branches = [
    "main",
    "1fc60c8ddcdc490ea2b1054059ad12b3",
    "98603c6b894040d7b963673969974df3",
    "2551070f24a3411fa6af17291e862e13",
]


def gather_runs_from_mlflow_experiment(experiment_name: str) -> list[str]:
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
    console.log(f"Branches to delete from mlflow:\t{len(branches_to_delete)}")
    return branches_to_delete


def gather_branches_from_huggingface(
    repo_id: str, repo_type: str, include_pull_requests=False
):
    git_refs = list_repo_refs(
        repo_id=repo_id,
        repo_type=repo_type,
        include_pull_requests=include_pull_requests,
    )
    branches_to_delete = [ref.name for ref in git_refs.branches]
    if include_pull_requests:
        branches_to_delete += [ref.ref for ref in git_refs.pull_requests]
    branches_to_delete = list(
        filter(lambda b: b not in exclude_branches, branches_to_delete)
    )
    console.log(f"Branches to delete from 🤗:\t{len(branches_to_delete)}")
    return branches_to_delete


def delete_branches(
    repo_id: str, branches_to_delete: list[str], repo_type: str, dry_run=True
):
    total = len(branches_to_delete)
    for i, branch in enumerate(branches_to_delete):
        if dry_run:
            time.sleep(1)
            console.log(f"Dry run: would delete branch: ({i+1}) '{branch}'.")
        else:
            flag = False
            hf_flag = False
            mlflow_flag = False
            try:
                delete_branch(repo_id=repo_id, repo_type=repo_type, branch=branch)
                console.log(f"Deleted branch: ({i+1}) '{branch}'.")
            except Exception as e:
                console.log(
                    f"Could not delete 🤗 branch: ({i+1}) '{branch}' due to error:\t'{e.__class__.__name__}'."
                )
                hf_flag = True

            try:
                mlflow.delete_run(branch)
                console.log(f"Deleted mlflow run: ({i+1}) '{branch}'.")
            except Exception as e:
                console.log(
                    f"Could not delete mlflow run: ({i+1}) '{branch}' due to error::\t'{e.__class__.__name__}'."
                )
                mlflow_flag = True

            if flag:
                total -= 1

    console.log(f"Deleted {total} branches.")


def main(args: argparse.Namespace):
    repo_id: str = args.repo_id
    repo_type: str = args.repo_type
    experiment_name: str = args.experiment_name
    mlflow_branches_to_delete = gather_runs_from_mlflow_experiment(experiment_name)
    hf_branches_to_delete = gather_branches_from_huggingface(
        repo_id, repo_type, args.include_pull_requests
    )
    branches_to_delete = list(
        set(mlflow_branches_to_delete) | set(hf_branches_to_delete)
    )
    if "main" in branches_to_delete:
        console.log(
            f"The {MAIN_BRANCH_NAME} branch was detected in the list of branches to delete. This is [red]dangerous[/red]... are you sure you want to delete it?"
        )
        response = console.input(
            f"Type {MAIN_BRANCH_NAME} to delete {MAIN_BRANCH_NAME} branch?"
        )
        if response.lower() != "main":
            console.log(
                f"Invalid response. Removing {MAIN_BRANCH_NAME} from the list of branches to delete."
            )
            branches_to_delete.remove("main")
        else:
            console.log(f"{MAIN_BRANCH_NAME} branch included for deletion...")
            time.sleep(5)

    console.log(f"Branches to delete: {branches_to_delete}")
    with console.status(
        f"Attempting to delete a total of {len(branches_to_delete)} branches..."
    ):
        delete_branches(repo_id, branches_to_delete, repo_type, dry_run=args.dry_run)


if __name__ == "__main__":
    description = textwrap.dedent(
        """
        Utility script for mass deleting dangling branches from 🤗 (HuggingFace). Partially automated using mlflow, as the runIDs are linked to a respective branch within 🤗.
        """
    )
    argparser = argparse.ArgumentParser(
        description=description, formatter_class=RichHelpFormatter
    )
    argparser.add_argument("-d", "--dry-run", action="store_true")
    argparser.add_argument("-e", "--experiment-name", default="default")
    argparser.add_argument("-r", "--repo-id", default=str())
    argparser.add_argument("-pr", "--include-pull-requests", action="store_true")
    argparser.add_argument(
        "-t", "--repo-type", default=str(), choices=["model", "dataset"]
    )
    args = argparser.parse_args()
    config = namespace_to_configuration(args)
    intro(config, title="MLflow and 🤗 Branch Deleter")
    main(args)
