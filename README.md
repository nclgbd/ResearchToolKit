# ResearchToolkit (rtk)

A collection of common functions and tools that I use for my research. Focus is on AI in radiology, medical imaging, and natural language processing. Integration with AzureML, Hugging Face, and MLflow for experiment tracking and model management. Designed to work using Hydra for configuration management.

## Installation

To begin, first fork or clone this repo. Then, install the package in editable mode using pip:

```bash
pip install -e .
```

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
