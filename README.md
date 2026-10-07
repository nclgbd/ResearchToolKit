# ResearchToolkit (rtk)

A collection of common functions and tools that I use for my research. Focus is on AI in radiology, medical imaging, and natural language processing. Integration with AzureML, Hugging Face, and MLflow for experiment tracking and model management. Designed to work using Hydra for configuration management.

## Installation

To begin, first fork or clone this repo. Then, install the package in editable mode using pip:

```bash
pip install -e .
```

## MLflow tracking server

Experiment tracking expects an MLflow server. A self-contained one lives in
`docker/mlflow/` — Postgres for run metadata, plus the MLflow server itself
proxying artifacts, so clients only need HTTP access (no shared filesystem or
S3). Bring it up with:

```bash
cd docker/mlflow
cp .env.example .env   # set POSTGRES_PASSWORD at minimum
docker compose up -d
```

On a host with only legacy Compose v1 (`docker-compose`, hyphenated — this is
what's currently installed on the lab box), every command in this section and in
`docker/mlflow/README.md` works unchanged apart from the name: `docker-compose
up -d`, `docker-compose logs -f mlflow`, and so on. The subcommands and flags
used here (`up -d`, `down`, `logs -f`, `run --rm --entrypoint`) all exist in v1.

The UI and API are then on `http://<host>:5000` (change the host port with
`MLFLOW_PORT` in `.env`). Check it came up:

```bash
curl -s http://localhost:5000/version
```

Point clients at it by setting `MLFLOW_TRACKING_URI` in the consuming project's
`.env` (or exporting it in your shell). `rtk.mlflow` reads it at import time and
falls back to `http://localhost:5000`, so call `load_dotenv()` before importing
anything that touches MLflow:

```
MLFLOW_TRACKING_URI=http://<host>:5000
```

Run history lives in the host paths named by `POSTGRES_DATA_HOST` and
`MLFLOW_ARTIFACT_ROOT_HOST`, so back those up like any other dataset. Everyday
operations:

```bash
docker compose logs -f mlflow   # server logs
docker compose down             # stop (data persists)
```

Keep the client's `mlflow` pin in this project's `pyproject.toml` at or below
the server version pinned in `docker/mlflow/Dockerfile`. See
[`docker/mlflow/README.md`](docker/mlflow/README.md) for the upgrade rules, why
the client floor is 3.4.0, and when a bump needs `mlflow db upgrade`.

## Example usage:

### Logger and Console

```python
from rtk.utils import get_logger, get_console

logger = get_logger(__name__)
console = get_console()
```

### Example configuration file

First, create a folder at the project directory level called `conf` and add a YAML configuration file, e.g., `config.yaml`. Below is a good template to start with:

```yaml
# conf/config.yaml

defaults:
  - _self_
  - models: default-model
  - override hydra/hydra_logging: colorlog
  - override hydra/job_logging: colorlog

env_file: /data/nicoleg/workspaces/dissertation/.env
name: "default"
dry_run: true
push_to_hub: true
date: ${now:%Y-%m-%d}
timestamp: ${now:%H-%M-%S}
dataset: ""
data_dir: /data/nicoleg/datasets/physionet.org/files/mimic-cxr-jpg/2.1.0
batched: false
batch_size:
positive_class: Pneumonia
with_rank: false
model_id: ""
models:
  pretrained_model_id: ${model_id}

hydra:
  job:
    chdir: true
  sweep:
    dir: outputs/${hydra.job.config_name}/${date}/${timestamp}
    subdir: ${hydra.job.override_dirname}
  run:
    dir: outputs/${hydra.job.config_name}/${date}/${timestamp}
```

### REPL

I often open up a terminal and do some debugging, so I have created some helpful code in setting up that environment

```python
import os
from dotenv import load_dotenv
from rtk import repl

repl.install()
load_dotenv()
console = repl.console
clear = console.clear

```
