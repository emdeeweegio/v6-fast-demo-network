# datavalgen-test

A `datavalgen` model and factory for a small fictional patient dataset, packaged
so it can be validated and generated via the `datavalgen` CLI, or as a Docker
image (e.g. for a vantage6 node).

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

## Docker

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
