# `rtk.configs` README.md

## Hydra settings

```bash
CONFIG_NAME=""
python scripts/run_train.py --config-dir=$(pwd)/configs/ --config-name=$CONFIG_NAME
```

```yaml
defaults:
  - _self_
  - override hydra/hydra_logging: colorlog
  - override hydra/job_logging: colorlog

env_file: /data/nicoleg/workspaces/dissertation/.env
name:
dry_run: true
push_to_hub: true
date: ${now:%Y-%m-%d}
timestamp: ${now:%H-%M-%S}
dataset:
data_dir: /data/nicoleg/datasets/physionet.org/files/mimic-cxr-jpg/2.1.0
batched: false
batch_size:
positive_class: Pneumonia
with_rank: false
model_id:

hydra:
  job:
    chdir: true
  sweep:
    dir: outputs/${hydra.job.config_name}/${date}/${timestamp}
    subdir: ${hydra.job.override_dirname}
  run:
    dir: outputs/${hydra.job.config_name}/${date}/${timestamp}
```
