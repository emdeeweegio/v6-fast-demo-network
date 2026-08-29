"""Federated datavalgen validation over a vantage6 network.

`central()` fans a `partial()` task out to every organization; each node
validates its own local CSV against a `datavalgen` pydantic model and returns
only a pass/fail verdict plus error counts — never the offending cell values
(same privacy stance as datavalgen's own `safe_validate`, see
data/datavalgen's README "Got: .." note). `central()` then reports which
organizations have correctly formatted data.

Assumes each node's requested database is a CSV (db_type=csv in its
nodes.<profile>.env row) — this validates file structure, not SQL sources.
"""
import os
from pathlib import Path

from datavalgen.plugins import get_model
from datavalgen.read_csv import read_csv_columns
from datavalgen.validate import check_column_names, check_csv_file

from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.decorators import RunMetaData, algorithm_client, metadata
from vantage6.algorithm.tools.util import info, warn

DEFAULT_MODEL = "example"


def _dataset_path() -> Path:
    label = os.environ["USER_REQUESTED_DATABASE_LABELS"].split(",")[0]
    db_type = os.environ.get(f"{label.upper()}_DATABASE_TYPE", "csv").lower()
    if db_type != "csv":
        raise ValueError(
            f"datavalgen validation only supports csv databases, got "
            f"'{db_type}' for label '{label}'"
        )
    return Path(os.environ[f"{label.upper()}_DATABASE_URI"])


@algorithm_client
def central(client: AlgorithmClient, model: str = DEFAULT_MODEL) -> dict:
    orgs = client.organization.list()
    org_names = {org["id"]: org["name"] for org in orgs}
    org_ids = list(org_names)
    info(f"Validating data on {len(org_ids)} node(s) against model '{model}'")

    task = client.task.create(
        input_={"method": "partial", "kwargs": {"model": model}},
        organizations=org_ids,
        name="datavalgen_validate_partial",
        description=f"Validate local CSV against datavalgen model '{model}'",
    )
    partials = client.wait_for_results(task["id"])

    nodes = {}
    for r in partials:
        name = org_names.get(r["organization_id"], str(r["organization_id"]))
        nodes[name] = {
            "ok": r["ok"],
            "num_errors": r["num_errors"],
            "column_errors": r["column_errors"],
            "column_warnings": r["column_warnings"],
        }
        status = "OK" if r["ok"] else f"FAILED ({r['num_errors']} invalid cell(s))"
        info(f"  {name}: {status}")

    all_ok = all(r["ok"] for r in partials)
    info("All nodes have correctly formatted data" if all_ok else "One or more nodes failed validation")
    return {"model": model, "all_ok": all_ok, "nodes": nodes}


@metadata
def partial(metadata: RunMetaData, model: str = DEFAULT_MODEL) -> dict:
    distribution = os.environ.get("DATAVALGEN_DISTRIBUTION")
    dataset_path = _dataset_path()
    model_cls = get_model(model, distribution=distribution)

    columns = read_csv_columns(dataset_path)
    column_check = check_column_names(columns, model_cls)
    if column_check.errors:
        warn(f"Column check failed: {'; '.join(column_check.errors)}")
        return {
            "organization_id": metadata.organization_id,
            "ok": False,
            "num_errors": None,
            "column_errors": list(column_check.errors),
            "column_warnings": list(column_check.warnings),
        }

    # max_errors=0 keeps this privacy-safe: check_csv_file still counts every
    # invalid cell but never stores/returns the offending values themselves.
    result = check_csv_file(dataset_path, model_cls, max_errors=0)
    if result.ok:
        info(f"{dataset_path.name}: OK")
    else:
        warn(f"{dataset_path.name}: {result.num_errors} invalid cell(s)")

    return {
        "organization_id": metadata.organization_id,
        "ok": result.ok,
        "num_errors": result.num_errors,
        "column_errors": [],
        "column_warnings": list(result.warnings),
    }
