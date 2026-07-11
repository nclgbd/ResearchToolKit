# Shared MLflow tracking server

A small, self-contained MLflow tracking server for the team: Postgres backend
store (experiment/run metadata) + the MLflow server itself, proxying artifact
reads/writes so clients only need HTTP access — no shared filesystem, S3, or
MinIO required.

## Start the server

```bash
cd docker/mlflow
cp .env.example .env   # edit POSTGRES_PASSWORD at minimum
docker compose up -d
```

The UI/API is now at `http://<host>:5000` (port from `MLFLOW_PORT`, default
`5000`). Whoever runs this should host it somewhere reachable by the rest of
the team (a lab server, a shared VM, etc.) — it's not meant to run on a single
person's laptop.

Data persists in `./data/postgres` (run metadata) and `./data/artifacts`
(PNGs, models, summaries) on the host, per the paths in `.env`. Back these up
like you would any other durable dataset — losing them loses run history.

## Point `classify.py` at it

Set `MLFLOW_TRACKING_URI` in `drish/.env` (or export it in your shell) to the
server's URL:

```
MLFLOW_TRACKING_URI=http://<host>:5000
```

`pipelines/classify.py` reads this via `${oc.env:MLFLOW_TRACKING_URI,...}` in
`pipelines/configs/classify.yaml` — see that file's `mlflow` block, and
`pipelines/configs/README.md` for the full option reference. Set
`mlflow.enabled=false` (or don't set `MLFLOW_TRACKING_URI`/don't run this
server at all) to fall back to the fully offline, Hydra-only pipeline.

## Notes / future hardening

- No authentication is configured — anyone with network access to the port
  can read/write. Fine for an internal team network; put it behind existing
  VPN/firewall boundaries, or look at `mlflow server`'s basic-auth plugin if
  that's not sufficient.
- Artifacts are proxied through the MLflow server onto a local bind mount.
  If storage needs outgrow one host, swap `--artifacts-destination` for an
  S3/MinIO URI without changing anything on the client side.
