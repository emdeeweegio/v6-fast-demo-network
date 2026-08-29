# datavalgen-test

A `datavalgen` model and factory for a small fictional patient dataset, packaged
so it can be validated and generated via the `datavalgen` CLI, or as a Docker
image (e.g. for a vantage6 node).

This folder builds two separate Docker images for two separate audiences:
`Dockerfile` (a standalone CLI image, for a human generating/validating data
by hand) and `v6-validate/Dockerfile` (which layers a vantage6 algorithm on
top of that same image, to run the identical validation as a federated task
without centralizing any data). See "Docker (standalone CLI image)" and
"Federated validation (vantage6)" below for each.

## Schema

Registered under model/factory name `example`:

| column              | type / rule                                  |
|---------------------|-----------------------------------------------|
| `patient_id`        | string, fixed format `PT-000001`              |
| `sex`                | `F` or `M`                                    |
| `age_at_diagnosis`  | int, 0-110                                    |
| `bmi`               | float, 10.0-70.0                              |
| `charlson_ci`       | int, 0-20 (Charlson Comorbidity Index)        |
| `disease_stage`     | `I`, `II`, `III`, or `IV`                     |

## Local development (no Docker)

> [!IMPORTANT]
> Unlike the Docker image, `uv sync` here does **not** fetch `datavalgen` from
> anywhere — `pyproject.toml` points at it via a local editable path
> (`../../../datavalgen`), so it requires a checkout of the
> [mdw-nl/datavalgen](https://github.com/mdw-nl/datavalgen) repo as a sibling
> of `v6-infrastructure-sh`, i.e.:
> ```
> work/
> ├── v6-infrastructure-sh/
> │   └── data/datavalgen/   <- this project
> └── datavalgen/            <- clone of mdw-nl/datavalgen
> ```
> Clone it first if you don't already have it:


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

## Docker (standalone CLI image)

No `datavalgen` checkout needed here — the image is built `FROM
ghcr.io/mdw-nl/datavalgen:v0.4.3`, which already has `datavalgen` installed;
this project's `Dockerfile` only adds the local model/factory package on top.

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

## Federated validation (vantage6)

`v6-validate/` wraps this same model/factory package into a vantage6 algorithm that validates each node's data in place and reports only pass/fail + error counts back centrally — no raw values ever leave a node. See the "Federated datavalgen validate" section in the repo root [README.md](../../README.md) for how to build and run it.
