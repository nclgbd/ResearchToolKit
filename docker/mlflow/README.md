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

## Point clients at it

Set `MLFLOW_TRACKING_URI` to the server's URL in the consuming project's `.env`
(or export it in your shell):

```
MLFLOW_TRACKING_URI=http://<host>:5000
```

`rtk/mlflow.py` reads this with `os.getenv` at import time, defaulting to
`http://localhost:5000` if it is unset — so `load_dotenv()` has to run before
you import anything that pulls in `rtk.mlflow`. Pointing the variable at a
server that is not running will surface as connection errors on the first
logging call, not at import.

## Version pinning and upgrades

The server image is pinned to an exact MLflow version in `Dockerfile`, and the
client is pinned to a compatible range in the top-level `pyproject.toml`
(`mlflow>=3.4.0,<3.15`). Keep those two in step, and prefer client <= server.

The `>=3.4.0` floor is not cosmetic. MLflow's `BaseMlflowSpanProcessor.__init__`
did not call `super().__init__()` before 3.4.0, while opentelemetry-sdk >= 1.44
moved span-processor state (`_metrics`, and `_shutdown` before it) into that
parent constructor and reads it in `on_end`. Since MLflow only constrains
`opentelemetry-sdk<3,>=1.9.0`, a fresh resolve picks up a new otel-sdk and every
span end then logs:

```
WARNING mlflow.entities.span: Failed to end span <id>:
  'MlflowV3SpanProcessor' object has no attribute '_metrics'.
```

Spans are still exported (the exporter runs before the failing line), but the
exception skips OpenTelemetry's `detach(token)`, leaking trace context on every
span. 3.4.0 added the `super().__init__()` call and fixes it.

### Does a version bump need `mlflow db upgrade`?

Bumping the **client** does not. The server owns the Postgres schema; clients
only speak HTTP to it and never connect to Postgres.

Bumping the **server** image may. Back up `./data/postgres` first, then let the
new container start and check its logs — `mlflow server` reports when the schema
is behind, and the migration is:

```bash
docker compose run --rm --entrypoint mlflow mlflow \
  db upgrade "postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}"
```

Schema migrations are one-way: downgrading the server image after one has run is
not supported, so the Postgres backup is the only way back.

## Notes / future hardening

- No authentication is configured — anyone with network access to the port
  can read/write. Fine for an internal team network; put it behind existing
  VPN/firewall boundaries, or look at `mlflow server`'s basic-auth plugin if
  that's not sufficient.
- Artifacts are proxied through the MLflow server onto a local bind mount.
  If storage needs outgrow one host, swap `--artifacts-destination` for an
  S3/MinIO URI without changing anything on the client side.
