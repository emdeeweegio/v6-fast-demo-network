---
name: v6-infrastructure-skill
description: Use this skill when setting up or running vantage6 infrastructure tests for any algorithm repo, both locally and in GitHub Actions, with pinned infra SHA and repo-specific test configs/data.
---

# V6 Infrastructure Skill

Use this skill to run reusable vantage6 infra tests for any algorithm package without copying infra code into each repo.

## When to use

- You need to run infra-backed tests for an algorithm repo.
- The algorithm repo does not vendor `v6-infrastructure-sh`.
- You want reproducible CI with infra pinned to a commit SHA.

Use this skill only after local package/runtime validation is already green. It is not a substitute for import-safety checks or local `run_context` smoke tests.

## Required inputs

Keep these files in the algorithm repo:

- `tests/infra/config.env`
- `tests/infra/nodes.env`
- `tests/data/*`
- `tests/infra/run_algo_smoke.sh`
- `tests/infra/vars.env.example` (recommended placeholders for local env vars)

`infra.sh` has no bare default node spec file — even its own built-in demo dataset is just a profile (`nodes.lung1.env`). Copy your repo's `tests/infra/nodes.env` into the harness as your own named profile (e.g. `nodes.<algo>.env`) and pass `--nodes <algo>` on every lifecycle command instead of relying on a default filename.

`nodes.env` format:

```text
name|api_key|db_uri|db_type|db_label
```

Recommended placeholders for humans and LLM agents:

```bash
export PYTHON_INTERPRETER="${PYTHON_INTERPRETER:-python3.12}"
export REGISTRY_PORT="${REGISTRY_PORT:-5001}"  # macOS often prefers 5001 or 50000
export UI_ENABLED="${UI_ENABLED:-false}"
export VERSION_VANTAGE6="${VERSION_VANTAGE6:-4.14.0}"
export DOCKER_REGISTRY="${DOCKER_REGISTRY:-localhost:${REGISTRY_PORT}/v6infra}"
export V6_SERVER_SOURCE_IMAGE="${V6_SERVER_SOURCE_IMAGE:-ghcr.io/mdw-nl/vantage6/infrastructure/server-lite:4.14.0-rc8}"
export V6_NODE_SOURCE_IMAGE="${V6_NODE_SOURCE_IMAGE:-ghcr.io/mdw-nl/vantage6/infrastructure/node-lite:4.14.0-rc8}"
export V6_UI_SOURCE_IMAGE="${V6_UI_SOURCE_IMAGE:-ghcr.io/mdw-nl/vantage6/infrastructure/ui:4.14.0-rc8}"
```

When writing real node configs, prefer explicit image references inside the YAML instead of relying only on shell defaults. At minimum, keep these fields visible in the node config:

```yaml
images:
  node: ghcr.io/mdw-nl/vantage6/infrastructure/node-lite@sha256:<digest>

run_context_file: true
share_config: false
share_algorithm_logs: false
prometheus:
  enabled: false
```

For production-like configs, also keep `policies.allowed_algorithms` and `databases[*].mount_mode` explicit. This makes review easier and avoids hidden runtime drift between shell env and node behavior.

## Local workflow

Recommended execution order:

1. Clean temp venv: verify package-root imports and minimal unit tests.
2. Local container smoke: verify `RUN_CONTEXT_FILE` execution without infra.
3. Infra lane: only then run `v6-infrastructure-sh`.

If step 1 fails, do not continue into infra. Recent STRATA-FIT failures were usually caused by package-root imports pulling runtime-only code, not by the infra harness itself.

1. Clone infra harness and pin SHA.
2. Copy algorithm test configs into infra folder.
3. Run lifecycle commands with consistent env flags for `up`, `test`, and `down`.

```bash
# from algorithm repo root
ALG_ROOT="$(pwd)"
INFRA_DIR="$ALG_ROOT/tools/v6-infra"
git clone https://github.com/mdw-nl/v6-infrastructure-sh.git "$INFRA_DIR"
cd "$INFRA_DIR"
git checkout <INFRA_SHA>

cp "$ALG_ROOT/tests/infra/config.env" infrastructure/config.env
cp "$ALG_ROOT/tests/infra/nodes.env" "infrastructure/nodes.${ALG_NAME:-algo}.env"

cd infrastructure
PYTHON_INTERPRETER="${PYTHON_INTERPRETER:-python3.12}" ./infra.sh --nodes "${ALG_NAME:-algo}" preflight
PYTHON_INTERPRETER="${PYTHON_INTERPRETER:-python3.12}" ENVIRONMENT=CI UI_ENABLED="${UI_ENABLED:-false}" ./infra.sh --nodes "${ALG_NAME:-algo}" up
bash "$ALG_ROOT/tests/infra/run_algo_smoke.sh"
PYTHON_INTERPRETER="${PYTHON_INTERPRETER:-python3.12}" UI_ENABLED="${UI_ENABLED:-false}" ./infra.sh --nodes "${ALG_NAME:-algo}" test
PYTHON_INTERPRETER="${PYTHON_INTERPRETER:-python3.12}" ENVIRONMENT=CI UI_ENABLED="${UI_ENABLED:-false}" ./infra.sh --nodes "${ALG_NAME:-algo}" down
```

If permissions are stale or task creation fails unexpectedly, do a clean restart:

```bash
./infra.sh down
rm -rf "$HOME/.local/share/vantage6/server/demoserver"
rm -rf "$HOME/.local/share/vantage6/node/"*
PYTHON_INTERPRETER="${PYTHON_INTERPRETER:-python3.12}" ENVIRONMENT=CI UI_ENABLED="${UI_ENABLED:-false}" ./infra.sh up
```

When debugging task failures, attach to the master org node container first and inspect the traceback there before changing infra configuration. This is the fastest way to separate container/package breakage from harness breakage.

## Registry migration note

Harbor is no longer a safe or reliable default for Vantage6 infrastructure images.

- Vantage6 publicly reported unauthorized access to the Harbor registry on April 2, 2026.
- On May 11, 2026, Vantage6 announced that infrastructure images had moved to GitHub Container Registry and Harbor would be discontinued.
- For local infra runs, a practical pattern is:
  1. pull `server-lite` / `node-lite` / `ui` from GHCR
  2. retag and push them into a local disposable registry
  3. point `DOCKER_REGISTRY` at that local mirror for `infra.sh`

## Rolesets in `entities.yaml`

When using this harness-generated entities, explicit user roles are not set in
the YAML. On vantage6 `4.13.3` the import flow assigns organization-scoped
`super` roles to imported org users by default.

If users can authenticate but receive `You lack the permission to do that!`
when creating tasks, first suspect stale server state and recreate infra
(`infra.sh down` + clean local state + `infra.sh up`) before changing role
assumptions.

## Local registry for custom images

If tasks stay in `non-existing Docker image`, publish the image to a reachable
local registry and reference that image in task payloads:

```bash
REGISTRY_PORT="${REGISTRY_PORT:-5001}"  # macOS fallback: 50000
docker run -d --restart unless-stopped -p "${REGISTRY_PORT}:5000" --name v6-local-registry registry:2
docker tag <local-image>:<tag> localhost:${REGISTRY_PORT}/<local-image>:<tag>
docker push localhost:${REGISTRY_PORT}/<local-image>:<tag>
```

Then use `localhost:${REGISTRY_PORT}/<local-image>:<tag>` as the task `image`.

## Client API compatibility (vantage6 4.13.x)

When collecting results in smoke scripts:

- Use `client.result.from_task(task_id=<id>)` (not `client.result.list(...)`).
- Use `client.run.list(task=<id>)` (argument name is `task`, not `task_id`).

For portable scripts, avoid hardcoded host-specific paths; prefer env vars and placeholders.

## Dependency guardrails for algorithm repos

- Avoid `vantage6-tools` in requirements; use `vantage6-client` and `vantage6-algorithm-tools`.
- Keep smoke scripts non-interactive (`MPLBACKEND=Agg` if plots are produced).
- Harbor is deprecated for infra-image sourcing. Prefer GHCR-backed `server-lite` / `node-lite` / `ui` images or a local mirror of them.
- Keep the Python package version (`VERSION_VANTAGE6`, e.g. `4.14.0`) separate from the Docker image source tags you mirror (e.g. `4.14.0-rc8`).
- Treat amd64 CI as the authoritative signoff environment for infra-backed validation.
- On ARM developer machines, assume the replacement GHCR infra images may require `--platform linux/amd64` when mirroring them locally.
- If `infra.sh up` reports an architecture probe failure or `exec format error`, stop debugging the algorithm itself: the published infra images are not runnable on that host without working amd64 emulation or an amd64 runner.
- For standalone-runtime migrations, prefer deterministic install order:
  1. stable primitives first
  2. git/tar algorithm dependencies with `--no-deps`
  3. target repo install with `--no-deps`
- Do not assume the local workspace repo matches the published pin consumed in tests. Validate the exact installed artifact in a fresh temp venv.
- Avoid reusing a repo-local `.venv` after interrupted pip/self-upgrade failures; create a fresh temp venv instead.
- Do not give `infra.sh` a shared project virtualenv. It upgrades and installs into the environment it is given, so use a disposable `/tmp/...` venv for infra runs.
- If the harness is not checked out as a sibling repo, set `INFRA_DIR` explicitly instead of relying on relative-path defaults.
- If a runtime smoke reports duplicate `run_context` entrypoints after an editable install, check for local `.egg-info` plus installed metadata overlap before changing code.
- For Docker/container smoke, ensure the `RUN_CONTEXT_FILE` input URIs are valid inside the container namespace, not only on the host.

## GitHub Actions workflow pattern

Use `actions/checkout` twice: once for the algorithm repo and once for infra harness at pinned SHA.

```yaml
- uses: actions/checkout@v4

- name: Checkout infra harness
  uses: actions/checkout@v4
  with:
    repository: mdw-nl/v6-infrastructure-sh
    ref: <INFRA_SHA>
    path: tools/v6-infra

- name: Inject repo test config
  run: |
    cp tests/infra/config.env tools/v6-infra/infrastructure/config.env
    cp tests/infra/nodes.env "tools/v6-infra/infrastructure/nodes.${{ github.event.repository.name }}.env"

- name: Start infra
  run: PYTHON_INTERPRETER=python3.12 ENVIRONMENT=CI UI_ENABLED=false tools/v6-infra/infrastructure/infra.sh --nodes "${{ github.event.repository.name }}" up

- name: Algorithm smoke
  run: bash tests/infra/run_algo_smoke.sh

- name: Infra smoke
  run: PYTHON_INTERPRETER=python3.12 UI_ENABLED=false tools/v6-infra/infrastructure/infra.sh --nodes "${{ github.event.repository.name }}" test

- name: Shutdown infra
  if: always()
  run: PYTHON_INTERPRETER=python3.12 ENVIRONMENT=CI UI_ENABLED=false tools/v6-infra/infrastructure/infra.sh --nodes "${{ github.event.repository.name }}" down
```

## Guardrails

- Do not duplicate infra scripts into algorithm repos.
- Always pin infra by commit SHA in CI.
- Keep algorithm-specific data and expectations in the algorithm repo only.
- Keep smoke tests deterministic and non-interactive.
- If Docker daemon is unavailable, fail fast at `preflight`.
- Prefer path-agnostic env vars/placeholders over machine-specific absolute paths.
- Use the same infra env flags for `up`, `test`, and `down` to avoid false negatives.

## Failure triage

1. `preflight` fails: Docker/runtime or missing config paths.
2. Local import gate failed before infra: fix package-root imports and mixed runtime/library boundaries first.
3. Local container `run_context` smoke failed: fix the algorithm/container contract before infra.
4. `up` fails: server/node startup config mismatch.
5. `run_algo_smoke.sh` fails: algorithm/package/runtime issue. Inspect the master org node container traceback first.
6. `infra.sh test` fails: verify `UI_ENABLED`/`--nodes` match how infra was started.
7. Task status `non-existing Docker image`: use local registry with configurable port and retag image.
8. Harbor server/node image pull fails: Harbor is retired for this use; switch to GHCR `server-lite` / `node-lite` / `ui` images or a local mirror of them.
9. `down` fails: teardown residue; rerun and inspect remaining containers.
