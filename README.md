# Vantage6 Local Infrastructure Harness

This repository provides a reusable, config-driven local vantage6 infrastructure for any algorithm package and data layout.

The default runtime now assumes GitHub Container Registry images:

- `ghcr.io/mdw-nl/vantage6/infrastructure/server-lite`
- `ghcr.io/mdw-nl/vantage6/infrastructure/node-lite`
- `ghcr.io/mdw-nl/vantage6/infrastructure/ui`

Harbor is intentionally no longer part of the default path.

## What changed

The infrastructure is now driven by:

- `infrastructure/config.env`: runtime defaults (Python version, v6 version, server/UI settings, paths)
- `infrastructure/nodes.<profile>.env`: node specs (`name|api_key|db_uri|db_type|db_label`) — `nodes.lung1.env` is the default profile, used when no `--nodes` flag is given (see "Switching node profiles" below)
- generated runtime artifacts in `infrastructure/generated/`

No hardcoded `alpha/beta/gamma` logic is required anymore. Any number of nodes can be used.

## Quick start

1. Edit `infrastructure/config.env` and `infrastructure/nodes.lung1.env`.
2. Run preflight checks:

```bash
cd infrastructure
./infra.sh preflight
```

3. Start infrastructure:

```bash
cd infrastructure
ENVIRONMENT=DEV ./infra.sh up
```

4. Run smoke tests:

```bash
cd infrastructure
./infra.sh test
```

5. Tear down:

```bash
cd infrastructure
./infra.sh down
```

## CI compatibility

Legacy entrypoints remain and map to the same flow:

- `infrastructure/setup.sh`
- `infrastructure/shutdown.sh`

The authoritative smoke environment is `ubuntu-latest` or another amd64 host. ARM developer machines are supported on a best-effort basis only.

If the published GHCR `server-lite` / `node-lite` / `ui` images are only available locally as amd64 images, `infra.sh up` now first installs `qemu-x86_64` binfmt when needed, then retries them with `DOCKER_DEFAULT_PLATFORM=linux/amd64`. Set `V6_AUTO_INSTALL_BINFMT=false` if you want to manage emulation yourself. If Docker still cannot execute the image through emulation, the harness fails fast with a clear architecture probe error instead of starting a partial stack and failing later during entity import.

## Node spec examples

`infrastructure/nodes.<profile>.env` supports mixed backends:

```text
alpha|<api_key>|../data/alpha.csv|csv|default
beta|<api_key>|postgresql://user:pass@db:5432/demo|sql|warehouse
```

If `db_uri` is empty, it defaults to `${DATA_DIR_DEFAULT}/<name>.csv`.

An optional second, read-only **folder** database can be added via 3 extra columns — `name|api_key|db_uri|db_type|db_label|extra_db_uri|extra_db_type|extra_db_label` — bind-mounted read-only at `/mnt/<extra_db_label>` inside algorithm containers. `argos_cnn` uses this to mount each org's NIfTI folder alongside its CSV manifest; see `infrastructure/nodes.argos.env`.

### Switching node profiles

Every algorithm's data layout — including the default LUNG1 CSVs — lives in its own `infrastructure/nodes.<profile>.env` file (`nodes.lung1.env`, `nodes.beach.env`, `nodes.argos.env`, ...). There is no bare `nodes.env`; the default is just the `lung1` profile like any other. Point any command at a profile with `--nodes <profile>` (before or after the command name):

```bash
./infra.sh --nodes lung1 up    # same as plain `./infra.sh up`
./infra.sh --nodes beach up
./infra.sh preflight --nodes argos
```

With no `--nodes` flag, `infra.sh` falls back to the `lung1` profile (`config.env`'s `NODES_CONFIG` default). `./infra.sh help` lists every profile it discovered. Adding a new algorithm's data layout only requires dropping in a new `nodes.<profile>.env` file — no changes to `infra.sh` itself.

Generated node YAML keeps the runtime-critical fields explicit:

```yaml
databases:
  - label: default
    type: csv
    uri: /absolute/path/to/data.csv
    mount_mode: ro
images:
  node: ghcr.io/mdw-nl/vantage6/infrastructure/node-lite:4.14.0-rc8
share_config: false
share_algorithm_logs: false
run_context_file: true
prometheus:
  enabled: false
```

For operator-facing configs, prefer digest-pinned image refs.

## Entities and roles

`infra.sh up` always generates an `entities.generated.yaml` and uploads it into
the server container (`vserver-local import ...`).

The generated entities currently do not set explicit user roles. On
vantage6 `4.13.3` in this harness, imported org users receive an
organization-scoped `super` role by default (verified from the server DB).

If you see permission errors on task creation (`You lack the permission to do that!`),
it is usually stale local server state. Run `infra.sh down`, clear local server DB
state, and run `infra.sh up` again.

## Local image registry

If nodes report `non-existing Docker image`, use a local registry and submit
tasks with a registry-backed image reference:

```bash
docker run -d --restart unless-stopped -p 5000:5000 --name v6-local-registry registry:2
docker tag local/v6-sklearn-linear-py:dev localhost:5000/v6-sklearn-linear-py:dev
docker push localhost:5000/v6-sklearn-linear-py:dev
```

Then use `localhost:5000/v6-sklearn-linear-py:dev` in task creation.

## Data

The `data/` directory holds the default LUNG1 CSVs plus everything that generates or checks per-node data for the non-default node profiles:

| Path | What it is |
|---|---|
| `data/lung1/` | Default LUNG1 CSVs, used by `nodes.lung1.env` (the default profile) |
| `data/generate_beach_data.py` | Generates BEACH-schema CSVs for `20k_logreg_challenge`, used with `nodes.beach.env` |
| `data/generate_argos_data.py` | Generates synthetic CT/GTV NIfTI + manifest data for `argos_cnn`, used with `nodes.argos.env` |
| `data/datavalgen/` | Pydantic schema ("example" model) + CLI for generating/validating fictional patient CSVs by hand, and a vantage6 algorithm (`v6-validate/`) that runs that same validation federated across nodes — see [data/datavalgen/README.md](data/datavalgen/README.md) and "Federated datavalgen validate" below |

Each data source's matching node profile (`infrastructure/nodes.<profile>.env`) and the algorithm that consumes it are documented together in the algorithm-specific sections below.

## Algorithms

The `algorithms/` directory contains federated learning algorithms that run on top of the vantage6 infrastructure. Each algorithm has its own folder with:

- An algorithm module (the code that runs on each node and the central orchestrator)
- A `Dockerfile` to build the node image
- A `run_study.py` script to submit the task from your machine

Available algorithms:

| Folder | Description |
|---|---|
| `average/` | Federated average of a single column |
| `logistic_regression/` | Federated logistic regression with normalization, batch training, and per-node evaluation |
| `kaplan_meier/` | Federated Kaplan-Meier survival curve with 95% CI and a matplotlib plot |
| `coxph/` | Federated Cox proportional hazards regression (hazard ratios across covariates) via federated Newton-Raphson iterations |
| `fed_statistics/` | Federated descriptive statistics (counts, binned counts, min/max, mean, std, bootstrap quantiles, nrows, nans) with threshold + secondary suppression for privacy |
| `argos_cnn/` | Federated ModResNet (PyTorch) for CT/GTV tumor segmentation — see **Argos CNN** below |

Each algorithm file has a block of **user-configurable variables** near the top (feature columns, target column, learning rate, number of rounds, train/test ratio, etc.). These act only as fallback defaults — see **Changing variables without rebuilding the image** below for how to override the important ones (the data columns) per run from `run_study.py`.

Note there are two logistic regression implementations in this repo: `logistic_regression/` trains a PyTorch model with federated averaging of weights across rounds, while `20k_logreg_challenge/` (see below) uses ADMM consensus optimization, which converges to the same coefficients as a centralized/pooled fit.

### Python environment (uv)

A single `pyproject.toml` at the **repo root** covers all dependencies for both `algorithms/` (run study scripts, lint algorithm code) and `data/` (synthetic data generation, e.g. `datavalgen`). Install [uv](https://docs.astral.sh/uv/) if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the environment from the repo root:

```bash
uv sync
```

This creates one `.venv` at the repo root, shared by both `algorithms/` and `data/`. Activate it or prefix commands with `uv run` from the root (e.g. `uv run python algorithms/average/run_study.py`).

### Building an algorithm image

Each algorithm must be built as a Docker image before the nodes can execute it. Build from the repo root so the tag matches what `run_study.py` expects:

```bash
# Average
docker build -t average:latest algorithms/average/

# Logistic regression
docker build -t logistic_regression:latest algorithms/logistic_regression/

# Kaplan-Meier
docker build -t kaplan_meier:latest algorithms/kaplan_meier/

# Cox proportional hazards
docker build -t coxph:latest algorithms/coxph/

# Federated descriptive statistics
docker build -t fed_statistics:latest algorithms/fed_statistics/
```

Because the nodes run inside Docker on the same host daemon, no registry push is needed for local testing. If nodes report `non-existing Docker image` anyway, see the **Local image registry** section.

### Running a study

With the infrastructure up (`infra.sh up`) and the image built:

```bash
# from the repo root
uv run python algorithms/average/run_study.py
uv run python algorithms/logistic_regression/run_study.py
uv run python algorithms/kaplan_meier/run_study.py
uv run python algorithms/coxph/run_study.py
uv run python algorithms/fed_statistics/run_study.py
```

The script authenticates against the local vantage6 server, submits the central task, waits for all nodes to complete, and prints the results.

### Changing variables without rebuilding the image

To test a different column/variable (e.g. `VARIABLE` in `average/run_study.py`, `FEATURE_COLS`/`TARGET_COL` in `logistic_regression/run_study.py`, or `TIME_COL`/`OUTCOME_COL`/`EXPL_VARS` in `coxph/run_study.py`), just edit the constants at the top of that algorithm's `run_study.py` and re-run it — no image rebuild needed. The constants in the algorithm file itself (`average.py`, etc.) are only the fallback defaults.

`coxph/` runs against the same LUNG1 data as `kaplan_meier/` (`Survival.time`/`deadstatus.event`), with `age`, `clinical.T.Stage`, `Clinical.N.Stage`, `Clinical.M.Stage` as the default covariates — no new node profile needed. It ships a `test_math_correctness.py` (run with `uv run pytest algorithms/coxph/`) that validates the federated Newton-Raphson fit against a directly-fit reference Cox model on a bundled test fixture — the actual proof the ported math wasn't broken in translation, independent of any live vantage6 infrastructure.

`fed_statistics/` also runs against LUNG1 (no new node profile needed, since it works on any tabular CSV) — override `STATISTICS`/`FILTERS`/`OPTIONS` in `fed_statistics/run_study.py` to request different columns/statistics per run. `OPTIONS` controls privacy suppression (`suppress_threshold`, `suppress_secondary`/`suppress_key`) and quantile bootstrap iterations; see `algorithms/fed_statistics/test_stats_correctness.py` (run with `uv run pytest algorithms/fed_statistics/`) for worked examples of each statistic plus the secondary-suppression mechanism.

### Adding your own algorithm

1. Create a new folder under `algorithms/` with an algorithm module and a `Dockerfile`.
2. Add any new dependencies to the root `pyproject.toml` and run `uv sync`.
3. Build the image: `docker build -t my-algo:latest algorithms/my-algo/`
4. Write a `run_study.py` that points at `my-algo:latest` and calls `"method": "central"`.

If the nodes report `non-existing Docker image`, see the **Local image registry** section below.

## 20kChallenge logistic regression (BEACH-schema)

`algorithms/20k_logreg_challenge/` runs the federated ADMM logistic regression from `mdw-nl/20kChallengeVantage6`, vendored directly into this repo. It needs BEACH-schema node data (`patient_t_stage`, `patient_n_stage`, `patient_m_stage`, `patient_overall_stage`, `year_of_diagnosis`, `vital_status`, `interval_diagnosis_to_last_visit_in_days`) — not the default LUNG1 data.

1. **Generate the node data** (from repo root, needs the root `.venv`, see above):

   ```bash
   uv run python data/generate_beach_data.py
   ```

   Writes per-node CSVs to `data/beach/splits_4nodes/{alpha,beta,gamma,theta}.csv`. Use `--num-subjects`, `--node-counts`, `--seed`, etc. to customize (see `parse_args()` in the script).

2. **Start the network with that data** — use the BEACH node spec (`infrastructure/nodes.beach.env`), not the default:

   ```bash
   cd infrastructure
   ENVIRONMENT=DEV ./infra.sh --nodes beach up
   ```

3. **Build the algorithm image** (from repo root):

   ```bash
   docker build -t 20klogregchallenge:latest algorithms/20k_logreg_challenge/
   ```

4. **Run it on the network** (from repo root):

   ```bash
   uv run python algorithms/20k_logreg_challenge/run_study.py
   ```

   Submits the ADMM task, waits for all nodes, and prints coefficients, AUC, and calibration.

## Argos CNN (federated CT/GTV segmentation)

`algorithms/argos_cnn/` is a PyTorch port of `mod_resnet` from the original `argosfeddeep` (TensorFlow) repo: a ResNet-encoder / PSP-style multi-scale-fusion decoder that segments lung tumors (GTV) on 2.5D (3-slice) CT stacks. It needs a per-slice NIfTI manifest, not the default LUNG1 CSVs.

1. **Generate the node data** (synthetic phantom CT/GTV NIfTI files + manifests — see `generate_argos_data.py`'s docstring for exactly what's realistic vs. fake about it):

   ```bash
   uv run python data/generate_argos_data.py
   ```

   Writes `data/argos/{alpha,beta,gamma,theta}.csv` and `data/argos/nifti/<org>/...`.

2. **Start the network with that data** — uses `infrastructure/nodes.argos.env`, which also mounts each org's `data/argos/nifti/<org>` folder read-only at `/mnt/nifti` (see the folder-database note above):

   ```bash
   cd infrastructure
   ENVIRONMENT=DEV ./infra.sh --nodes argos up
   ```

3. **Build the algorithm image** (from repo root — copies both `argos_cnn.py` and `model.py`):

   ```bash
   docker build -t argos_cnn:latest algorithms/argos_cnn/
   ```

4. **Run it on the network**:

   ```bash
   uv run python algorithms/argos_cnn/run_study.py
   ```

   Submits federated training (FedAvg, weighted by each node's dataset size), waits for all rounds, and writes the trained weights to `argos_cnn_trained_weights.pt`.

Notes:
- Unlike the other algorithms, hyperparameters (`N_ROUNDS`/`LOCAL_STEPS`/`BATCH_SIZE`/`LEARNING_RATE`) live only in `argos_cnn.py` — `run_study.py` intentionally passes none, so there's a single source of truth. Edit them there and rebuild the image.
- Trained weights ride inside the normal vantage6 task payload as a gzip+base64 string (tens of MB for this model) rather than a separate file-transfer channel — this won't scale indefinitely to much larger models; see `argos_cnn.py`'s module docstring and vantage6's blob-storage mechanism (`vantage6/common/client/blob_storage.py`) as the eventual fix.
- `data/argos/` is synthetic phantom data for exercising the pipeline end to end (I/O, normalization, FedAvg), not real anatomy — don't draw conclusions about model quality from it.

## Federated datavalgen validate

`data/datavalgen/v6-validate/` is a vantage6 algorithm that checks every node's local CSV against a `datavalgen` pydantic model *without* centralizing any data: each node validates its own file and returns only a pass/fail verdict plus an error count — never the offending cell values (same privacy stance as datavalgen's own `safe_validate`; see `data/datavalgen/README.md`'s "Got: .." note). The central task then reports which organizations have correctly formatted data.

It's built on top of `data/datavalgen`'s own image (`ghcr.io/mdw-nl/datavalgen:v0.4.3` + the `example` model/factory package), not a fresh `python` base, so its build context is `data/datavalgen`, not `data/datavalgen/v6-validate`.

1. **Start the network with the datavalgen node data** — uses `infrastructure/nodes.datavalgen.env`:

   ```bash
   cd infrastructure
   ENVIRONMENT=DEV ./infra.sh --nodes datavalgen up
   ```

2. **Build the algorithm image** (build context is `data/datavalgen`, Dockerfile is in the `v6-validate/` subfolder):

   ```bash
   cd data/datavalgen
   docker build -f v6-validate/Dockerfile -t datavalgen-validate:local .
   ```

3. **Run it on the network** (from repo root):

   ```bash
   uv run python data/datavalgen/v6-validate/run_study.py
   ```

   Submits the validation task, waits for every node, and prints an OK/FAILED line per organization.

Notes:
- Only validates CSV databases (`db_type=csv` in the node spec) — it validates file structure/values, not SQL sources.
- To check against a different registered model, edit `MODEL` in `run_study.py` (must be an entry point under `datavalgen.models` in the `datavalgen-test` distribution, e.g. new models added to `data/datavalgen/src/datavalgen_test/model.py`).

## Notes

- Docker daemon must be available before running setup/test.
- `STRICT_DATA_CHECKS=true` enforces local CSV existence checks.
- UI can be disabled with `UI_ENABLED=false`.
- Recommended workflow is:
  1. validate the target algorithm repo in a fresh `/tmp` venv
  2. run a local container smoke for `RUN_CONTEXT_FILE`
  3. only then use this harness
