# datavalgen-test

A `datavalgen` model and factory for a small fictional patient dataset, packaged
so it can be validated and generated via the `datavalgen` CLI, as a standalone
Docker image, or as a vantage6 algorithm that validates data federated across
nodes. This file is the single source of truth for all of that — nothing
datavalgen-related lives only in the repo root README.

## Schema

Registered under model/factory name `example`:

| column              | type / rule                                  |
|---------------------|-----------------------------------------------|
| `patient_id`        | string, fixed format `PT-000001`              |
| `sex`                | `female` or `male`                            |
| `age_at_diagnosis`  | int, 0-110                                    |
| `bmi`               | float, 10.0-70.0                              |
| `charlson_ci`       | int, 0-20 (Charlson Comorbidity Index)        |
| `disease_stage`     | `I`, `II`, `III`, or `IV`                     |

## Quick reference

This folder has **two Dockerfiles building two different images** for two
different audiences. Everything below is run with `data/datavalgen/` as your
working directory *except* the last row:

| I want to...                                    | Command |
|--------------------------------------------------|---------|
| Generate/validate data locally, no Docker         | `uv run datavalgen generate -f example -n 10 -o data/patients.csv` |
| Build the standalone CLI image                    | `docker build -t datavalgen-test:local .` |
| Build the federated vantage6 algorithm image      | `docker build -f v6-validate/Dockerfile -t datavalgen-validate:local .` |
| Run the federated task against the harness        | `uv run python data/datavalgen/v6-validate/run_study.py` **(from the repo root, not here)** |

> [!WARNING]
> Both `docker build` commands above use build context `data/datavalgen/` —
> **never `cd` into `v6-validate/` first.** `v6-validate/Dockerfile` copies
> `./src` and `./pyproject.toml` from this folder (the same model/factory
> package `./Dockerfile` uses), so building with `v6-validate/` as the context
> is missing those files and fails with `COPY failed: file not found`. The
> `-f v6-validate/Dockerfile` flag is what lets you point at that Dockerfile
> while keeping the context here.

> [!NOTE]
> `run_study.py` imports `vantage6.client`, which is a dependency of the
> **repo root's** `.venv` (see the root `pyproject.toml`), not this folder's
> `.venv`. Run it with `uv run` from the repo root — running it from inside
> `data/datavalgen/` will fail with `ModuleNotFoundError: No module named
> 'vantage6'`.

## Local development (no Docker)

> [!IMPORTANT]
> Unlike the Docker images, `uv sync` here does **not** fetch `datavalgen`
> from anywhere — `pyproject.toml` points at it via a local editable path
> (`../../../datavalgen`), so it requires a checkout of the
> [mdw-nl/datavalgen](https://github.com/mdw-nl/datavalgen) repo as a sibling
> of `v6-infrastructure-sh`, i.e.:
> ```
> work/
> ├── v6-infrastructure-sh/
> │   └── data/datavalgen/   <- this project
> └── datavalgen/            <- clone of mdw-nl/datavalgen
> ```
> Clone it first if you don't already have it.

```bash
uv sync
uv run datavalgen validate --list
uv run datavalgen generate --list
```

Generate fake data:
```bash
mkdir -p data
uv run datavalgen generate -f example -n 10 -o data/patients.csv
```

Validate a CSV:
```bash
uv run datavalgen validate -m example -d data/patients.csv
```

## Image 1: standalone CLI (`Dockerfile`)

For a human generating/validating data by hand. No `datavalgen` checkout
needed — the image is built `FROM ghcr.io/mdw-nl/datavalgen:v0.4.3`, which
already has `datavalgen` installed; this project's `Dockerfile` only adds the
local model/factory package on top.

Build the image:
```bash
docker build -t datavalgen-test:local .
```

List available models/factories baked into the image:
```bash
docker run --rm datavalgen-test:local validate --list
docker run --rm datavalgen-test:local generate --list
```

Generate fake data to a mounted `data` directory:
```bash
mkdir -p data
docker run \
  --network none --rm --read-only --cap-drop=ALL --security-opt no-new-privileges=true --log-driver=none --user "$(id -u):$(id -g)" \
  -v "$(pwd)/data:/data" \
  datavalgen-test:local \
  generate -f example -n 10 -o /data/patients.csv --force
```

Validate a CSV (mounted read-only):
```bash
docker run \
  --network none --rm --read-only --cap-drop=ALL --security-opt no-new-privileges=true --log-driver=none --user "$(id -u):$(id -g)" \
  -v "$(pwd)/data/patients.csv:/data.csv:ro" \
  datavalgen-test:local \
  validate -m example
```

> [!IMPORTANT]
> The flags `--network none --rm --read-only --cap-drop=ALL --security-opt
> no-new-privileges=true --log-driver=none --user $(id -u):$(id -g)` run the
> image in a locked-down way: no networking, no writes outside mounted
> volumes, dropped OS privileges, and no persisted container logs. Use them,
> especially with real (non-fake) data.

> [!NOTE]
> If validation errors are found, the tool prints the actual offending data
> value ("Got: ..") to help you fix it locally. Don't share that output
> outside your own environment.

## Image 2: federated vantage6 algorithm (`v6-validate/Dockerfile`)

`v6-validate/` wraps the same model/factory package into a vantage6 algorithm
that validates each node's data in place and reports only pass/fail + error
counts back centrally — no raw values ever leave a node. It's built on top of
this same `Dockerfile`'s base image, not a fresh `python` base, which is why
its build context is `data/datavalgen/` too (see the warning above).

Build the image:
```bash
docker build -f v6-validate/Dockerfile -t datavalgen-validate:local .
```

Run it against the local harness:

1. **Start the network with the datavalgen node data** — uses
   `infrastructure/nodes.datavalgen.env`:

   ```bash
   cd infrastructure
   ENVIRONMENT=DEV ./infra.sh --nodes datavalgen up
   ```

2. **Build the algorithm image** — the command above, run from
   `data/datavalgen/`.

3. **Run it** (from the repo root, using the root `.venv` — see the note
   above):

   ```bash
   uv run python data/datavalgen/v6-validate/run_study.py
   ```

   Submits the validation task, waits for every node, and prints an OK/FAILED
   line per organization.

Notes:
- Only validates CSV databases (`db_type=csv` in the node spec) — it validates
  file structure/values, not SQL sources.
- To check against a different registered model, edit `MODEL` in
  `run_study.py` (must be an entry point under `datavalgen.models` in the
  `datavalgen-test` distribution, e.g. new models added to
  `src/datavalgen_test/model.py`).
