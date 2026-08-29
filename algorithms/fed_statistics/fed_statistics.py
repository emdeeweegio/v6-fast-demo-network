from hashlib import blake2b
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2
from vantage6.algorithm.tools.util import error, info, warn
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.client import AlgorithmClient

VALID_STATISTICS = {
    "counts",
    "binned_counts",
    "minmax",
    "mean",
    "std",
    "quantiles",
    "nrows",
    "nans",
}

DEFAULT_OPTIONS = {
    "suppress_threshold": 5,
    "suppress_secondary": False,
    "suppress_key": None,
    "quantile_bootstrap_iterations": 1000,
}


# ── Secondary suppression (HMAC-based, deterministic) ────────────────────────


def _hmac_u32(secret: str, *parts: str) -> int:
    digest = blake2b((":".join((secret, *parts))).encode(), digest_size=4).digest()
    return int.from_bytes(digest, "little")


def pick_secondary_cat(secret: str, column: str, candidates: list) -> str:
    extra = "s.e.c.o.n.d.a.r.y"
    idx = _hmac_u32(secret, column, extra) % len(candidates)
    return candidates[idx]


def deterministic_offset(secret: str, column: str, window: int) -> int:
    return _hmac_u32(secret, column) % window


# ── Request normalization (replaces pydantic contracts) ──────────────────────


def _normalize_statistics(statistics: dict) -> dict:
    if not statistics:
        raise ValueError("statistics must include at least one column")

    normalized = {}
    for column, spec in statistics.items():
        if isinstance(spec, list):
            spec = {"stats": spec}
        elif not isinstance(spec, dict):
            raise TypeError(f"Statistic request for column '{column}' must be a list or dict")

        stats = spec.get("stats")
        if not stats:
            raise ValueError(f"Column '{column}' must specify at least one statistic")
        for name in stats:
            if name not in VALID_STATISTICS:
                raise ValueError(f"Unknown statistic '{name}' for column '{column}'")

        bins = spec.get("bins")
        if "binned_counts" in stats and (bins is None or len(bins) < 2):
            raise ValueError(f"binned_counts requires at least two bin edges for column '{column}'")

        normalized[column] = {
            "stats": list(stats),
            "bins": bins,
            "min_cap": spec.get("min_cap"),
            "max_cap": spec.get("max_cap"),
        }
    return normalized


def _normalize_options(options: dict = None) -> dict:
    resolved = dict(DEFAULT_OPTIONS)
    resolved.update(options or {})
    if resolved["suppress_secondary"] and not resolved["suppress_key"]:
        raise ValueError("suppress_key is required when suppress_secondary is enabled")
    return resolved


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    filtered = df
    for column, value in filters.items():
        if column not in filtered.columns:
            raise ValueError(f"Filter column '{column}' not found in dataset")
        if isinstance(value, list):
            filtered = filtered[filtered[column].isin(value)]
        else:
            filtered = filtered[filtered[column] == value]
    return filtered


# ── Local (per-node) statistics ───────────────────────────────────────────────


def _normalize_display_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if np.isnan(value):
            return "nan"
        if value.is_integer():
            return str(int(value))
        return f"{value:g}"
    return str(value)


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    source = df[column]
    converted = pd.to_numeric(source, errors="coerce")
    invalid_mask = source.notna() & converted.isna()
    if invalid_mask.any():
        raise TypeError(
            f"Column '{column}' contains non-numeric values and cannot be used for numeric statistics"
        )
    return converted


def _validate_column(df: pd.DataFrame, column: str) -> bool:
    if column not in df.columns:
        warn(f"Column '{column}' not found in dataset")
        return False
    return True


def compute_local_quantile_sampling_variance(values: np.ndarray, q: float, iterations: int) -> float:
    quantiles = []
    n = len(values)
    np.random.seed(0)
    for _ in range(iterations):
        sample = np.random.choice(values, size=n, replace=True)
        quantiles.append(np.quantile(sample, q))
    return float(np.var(quantiles))


def compute_local_quantiles(df: pd.DataFrame, column: str, options: dict):
    series = _numeric_series(df, column).dropna()
    if len(series) <= options["suppress_threshold"]:
        return None

    quantiles = {}
    q_levels = {1: 0.25, 2: 0.50, 3: 0.75}
    values = series.values
    for idx, q_value in q_levels.items():
        quantiles[f"Q{idx}"] = float(np.quantile(values, q_value))
        quantiles[f"variance_Q{idx}"] = compute_local_quantile_sampling_variance(
            values, q_value, options["quantile_bootstrap_iterations"]
        )
    quantiles["nrows"] = int(len(series))
    return quantiles


def compute_local_nans(df: pd.DataFrame, column: str, options: dict):
    count = int(df[column].isna().sum())
    return None if count <= options["suppress_threshold"] else count


def compute_local_nrows(df: pd.DataFrame, column: str, options: dict):
    count = int(df[column].dropna().shape[0])
    return None if count <= options["suppress_threshold"] else count


def compute_local_means(df: pd.DataFrame, column: str, options: dict):
    total_n = compute_local_nrows(df, column, options)
    if total_n is None or total_n <= options["suppress_threshold"]:
        return {"sum": None, "n": None}
    total_sum = _numeric_series(df, column).dropna().sum()
    if isinstance(total_sum, np.generic):
        total_sum = total_sum.item()
    if not isinstance(total_sum, (int, float)):
        raise TypeError(f"Local sum for column '{column}' is not numeric")
    return {"sum": total_sum, "n": total_n}


def compute_local_std_moments(df: pd.DataFrame, column: str, options: dict):
    local_mean = compute_local_means(df, column, options)
    total_n = local_mean["n"]
    total_sum = local_mean["sum"]
    if total_n is None or total_sum is None:
        return {"sumsq": None, "sum": None, "n": None}
    series = _numeric_series(df, column).dropna()
    total_sumsq = np.square(series.values).sum()
    if isinstance(total_sumsq, np.generic):
        total_sumsq = total_sumsq.item()
    if not isinstance(total_sumsq, (int, float)):
        raise TypeError(f"Local sumsq for column '{column}' is not numeric")
    return {"sumsq": total_sumsq, "sum": total_sum, "n": total_n}


def _maybe_apply_secondary_suppression(counts: dict, *, column: str, options: dict) -> None:
    if not options["suppress_secondary"]:
        return
    suppressed = [key for key, value in counts.items() if value["lower"] != value["upper"]]
    if len(suppressed) != 1 or len(counts) <= 1:
        return

    unsuppressed = [key for key in counts if key not in suppressed]
    secondary_key = pick_secondary_cat(options["suppress_key"] or "", column, sorted(unsuppressed))
    actual_value = counts[secondary_key]["lower"]
    offset = deterministic_offset(
        options["suppress_key"] or "", column, max(options["suppress_threshold"], 1)
    )
    lower = actual_value - offset
    counts[secondary_key] = {"lower": lower, "upper": lower + options["suppress_threshold"]}


def compute_local_counts(df: pd.DataFrame, column: str, options: dict) -> dict:
    series = df[column].dropna()
    value_counts = series.value_counts(dropna=False)
    counts = {}
    for raw_value, count in value_counts.items():
        label = _normalize_display_value(raw_value)
        if int(count) <= options["suppress_threshold"]:
            counts[label] = {"lower": 0, "upper": options["suppress_threshold"]}
        else:
            counts[label] = {"lower": int(count), "upper": int(count)}
    _maybe_apply_secondary_suppression(counts, column=column, options=options)
    return counts


def compute_local_binned_counts(df: pd.DataFrame, column: str, spec: dict, options: dict) -> dict:
    bins = spec.get("bins")
    if bins is None or len(bins) < 2:
        raise ValueError(f"binned_counts requires bins for column '{column}'")

    series = _numeric_series(df, column).dropna().astype(float)
    counts, edges = np.histogram(series.values, bins=np.asarray(sorted(bins), dtype=float))
    result = {}
    for count, left, right in zip(counts, edges[:-1], edges[1:]):
        if np.isneginf(left):
            label = f"< {right:g}"
        elif np.isposinf(right):
            label = f">= {left:g}"
        else:
            label = f"{left:g}-{right:g}"
        if int(count) <= options["suppress_threshold"]:
            result[label] = {"lower": 0, "upper": options["suppress_threshold"]}
        else:
            result[label] = {"lower": int(count), "upper": int(count)}
    _maybe_apply_secondary_suppression(result, column=column, options=options)
    return result


def compute_local_minmax(df: pd.DataFrame, column: str, spec: dict, options: dict):
    series = _numeric_series(df, column).dropna()
    if len(series) <= options["suppress_threshold"]:
        return {"min": None, "max": None}

    minimum = series.min()
    maximum = series.max()
    if isinstance(minimum, np.generic):
        minimum = minimum.item()
    if isinstance(maximum, np.generic):
        maximum = maximum.item()

    min_cap = spec.get("min_cap")
    max_cap = spec.get("max_cap")
    if min_cap is not None and minimum < min_cap:
        minimum = min_cap
    if max_cap is not None and maximum > max_cap:
        maximum = max_cap

    return {"min": minimum, "max": maximum}


def _compute_stats_for_frame(working: pd.DataFrame, statistics: dict, options: dict) -> dict:
    methods = {
        "counts": lambda column, spec: compute_local_counts(working, column, options),
        "binned_counts": lambda column, spec: compute_local_binned_counts(working, column, spec, options),
        "minmax": lambda column, spec: compute_local_minmax(working, column, spec, options),
        "mean": lambda column, spec: compute_local_means(working, column, options),
        "std": lambda column, spec: compute_local_std_moments(working, column, options),
        "quantiles": lambda column, spec: compute_local_quantiles(working, column, options),
        "nrows": lambda column, spec: compute_local_nrows(working, column, options),
        "nans": lambda column, spec: compute_local_nans(working, column, options),
    }

    local_stats = {}
    for column, spec in statistics.items():
        if not _validate_column(working, column):
            continue
        local_stats[column] = {}
        for statistic in spec["stats"]:
            local_stats[column][statistic] = methods[statistic](column, spec)
    return local_stats


def compute_local_stats_frame(df: pd.DataFrame, statistics: dict, filters: dict, options: dict) -> dict:
    working = apply_filters(df, filters)
    return _compute_stats_for_frame(working, statistics, options)


# ── Federated (central) aggregation ───────────────────────────────────────────


def compute_federated_counts(local_counts: list) -> dict:
    federated_counts = {}
    for local_count in local_counts:
        for category, bounds in local_count.items():
            if category in federated_counts:
                federated_counts[category]["lower"] += bounds["lower"]
                federated_counts[category]["upper"] += bounds["upper"]
            else:
                federated_counts[category] = {"lower": bounds["lower"], "upper": bounds["upper"]}
    return federated_counts


def compute_federated_minmax(local_minmaxes: list) -> dict:
    mins = [item["min"] for item in local_minmaxes if item.get("min") is not None]
    maxes = [item["max"] for item in local_minmaxes if item.get("max") is not None]
    return {"min": min(mins) if mins else None, "max": max(maxes) if maxes else None}


def compute_federated_mean(local_means: list):
    sums = [item["sum"] for item in local_means if item.get("sum") is not None]
    counts = [item["n"] for item in local_means if item.get("n") is not None]
    if len(sums) != len(counts):
        error("Length mismatch between local sums and n values")
    total_n = sum(int(value) for value in counts)
    return (sum(float(value) for value in sums) / total_n) if total_n > 0 else None


def test_cdf_homogeneity(local_quantiles: list) -> bool:
    results = {}
    for idx in range(1, 4):
        quantiles_i = []
        variances_i = []
        for item in local_quantiles:
            q_val = item.get(f"Q{idx}")
            var_val = item.get(f"variance_Q{idx}")
            if q_val is not None and var_val is not None and not (np.isnan(q_val) or np.isnan(var_val)):
                quantiles_i.append(q_val)
                variances_i.append(var_val)

        quantiles_np = np.array(quantiles_i)
        variances_np = np.array(variances_i)
        positive = variances_np[variances_np > 0]
        epsilon = max(0.05 * np.median(positive), 1e-2) if len(positive) > 0 else 1e-2
        variances_np = np.where(variances_np < epsilon, epsilon, variances_np)

        k = len(quantiles_np)
        if k < 2:
            results[f"Q{idx}"] = {"p_value": np.nan}
            continue

        weights = 1.0 / variances_np
        weighted_mean = np.sum(weights * quantiles_np) / np.sum(weights)
        q_stat = np.sum(weights * (quantiles_np - weighted_mean) ** 2)
        p_value = 1 - chi2.cdf(q_stat, k - 1)
        results[f"Q{idx}"] = {"p_value": float(p_value)}

    return any(
        results.get(key, {}).get("p_value") is not None and results.get(key, {}).get("p_value") < 0.05
        for key in ("Q1", "Q2", "Q3")
    )


def compute_federated_quantiles(local_quantiles: list) -> dict:
    if test_cdf_homogeneity(local_quantiles):
        info("Significant heterogeneity detected among local CDFs, quantiles will not be calculated")
        return {
            "Q1": None,
            "Q1_std_err": None,
            "Q2": None,
            "Q2_std_err": None,
            "Q3": None,
            "Q3_std_err": None,
        }

    federated_quantiles = {}
    for idx in range(1, 4):
        quantiles_i = np.array([item[f"Q{idx}"] for item in local_quantiles if not np.isnan(item[f"Q{idx}"])])
        variances_i = np.array(
            [item[f"variance_Q{idx}"] for item in local_quantiles if not np.isnan(item[f"variance_Q{idx}"])]
        )
        if len(quantiles_i) != len(variances_i):
            error("Length mismatch between quantiles and variances")

        omega_i0 = 1.0 / variances_i
        quantile_0 = np.sum(omega_i0 * quantiles_i) / np.sum(omega_i0)
        tau2_nom = np.sum(omega_i0 * (quantiles_i - quantile_0) ** 2) - (len(quantiles_i) - 1)
        tau2_den = np.sum(omega_i0) - np.sum(omega_i0**2) / np.sum(omega_i0)
        tau2 = np.max([0.0, tau2_nom / tau2_den]) if tau2_den else 0.0

        omega_i = 1.0 / (variances_i + tau2)
        federated_quantile = np.sum(quantiles_i * omega_i) / np.sum(omega_i)
        federated_error = np.sqrt(1.0 / np.sum(omega_i))
        federated_quantiles[f"Q{idx}"] = float(federated_quantile) if not np.isnan(federated_quantile) else None
        federated_quantiles[f"Q{idx}_std_err"] = (
            float(federated_error) if not np.isnan(federated_error) else None
        )
    return federated_quantiles


def compute_federated_nans(local_nans: list) -> int:
    return int(sum(value for value in local_nans if value is not None))


def compute_federated_nrows(local_nrows: list) -> int:
    return int(sum(value for value in local_nrows if value is not None))


def compute_federated_std(local_stds: list):
    complete_results = []
    for local_std in local_stds:
        values = [local_std.get("sumsq"), local_std.get("sum"), local_std.get("n")]
        if any(value is None for value in values):
            if not all(value is None for value in values):
                error("Inconsistent std payload from a site")
                return None
            continue
        complete_results.append(local_std)

    if not complete_results:
        return None

    total_n = sum(int(item["n"]) for item in complete_results)
    if total_n < 2:
        return None

    total_sum = sum(float(item["sum"]) for item in complete_results)
    total_sumsq = sum(float(item["sumsq"]) for item in complete_results)
    mean = total_sum / total_n
    variance_population = max(float((total_sumsq / total_n) - (mean**2)), 0.0)
    variance_sample = variance_population * (total_n / (total_n - 1))
    return float(np.sqrt(variance_sample))


def _collect_local_stat_inputs(local_results: list, column: str, statistic: str):
    usable_results = []
    missing_results = 0
    for local_result in local_results:
        try:
            stat_result = local_result[column][statistic]
        except (KeyError, TypeError):
            missing_results += 1
            continue
        if stat_result is None:
            missing_results += 1
            continue
        if statistic in {"mean", "quantiles", "minmax", "std"} and stat_result == {}:
            missing_results += 1
            continue
        usable_results.append(stat_result)
    return usable_results, missing_results


def aggregate_local_results(local_results: list, statistics: dict):
    methods = {
        "counts": compute_federated_counts,
        "binned_counts": compute_federated_counts,
        "minmax": compute_federated_minmax,
        "mean": compute_federated_mean,
        "std": compute_federated_std,
        "quantiles": compute_federated_quantiles,
        "nrows": compute_federated_nrows,
        "nans": compute_federated_nans,
    }
    warnings = []
    aggregated = {}
    for column, spec in statistics.items():
        aggregated[column] = {}
        for statistic in spec["stats"]:
            inputs, missing = _collect_local_stat_inputs(local_results, column, statistic)
            if missing:
                warnings.append(
                    f"Column '{column}' statistic '{statistic}' was missing or suppressed on {missing} site(s)."
                )
            if not inputs:
                aggregated[column][statistic] = None
                continue
            aggregated[column][statistic] = methods[statistic](inputs)

    if any("quantiles" in spec["stats"] for spec in statistics.values()):
        warnings.append("Quantile estimation uses local bootstrapping and can be slow on large datasets.")
    return aggregated, warnings


# ── vantage6 entry points ─────────────────────────────────────────────────────


@algorithm_client
def central(
    client: AlgorithmClient,
    statistics: dict,
    filters: dict = None,
    options: dict = None,
) -> dict:
    stats_spec = _normalize_statistics(statistics)
    resolved_options = _normalize_options(options)
    resolved_filters = filters or {}

    orgs = client.organization.list()
    org_ids = [org["id"] for org in orgs]

    info("=" * 60)
    info("FEDERATED DESCRIPTIVE STATISTICS")
    info("=" * 60)
    info(f"Organizations : {[org['name'] for org in orgs]}")
    info(f"Columns       : {list(stats_spec.keys())}")

    task = client.task.create(
        input_={
            "method": "compute_local_stats",
            "kwargs": {
                "statistics": stats_spec,
                "filters": resolved_filters,
                "options": resolved_options,
            },
        },
        organizations=org_ids,
        name="fed_statistics_local",
        description="Compute local descriptive statistics",
    )
    local_results = client.wait_for_results(task["id"])

    columns, warnings = aggregate_local_results(local_results, stats_spec)
    for warning in warnings:
        warn(warning)

    return {"columns": columns, "warnings": warnings}


@data(1)
def compute_local_stats(
    df: pd.DataFrame,
    statistics: dict,
    filters: dict = None,
    options: dict = None,
) -> dict:
    stats_spec = _normalize_statistics(statistics)
    resolved_options = _normalize_options(options)
    info(f"Computing local statistics for columns: {list(stats_spec.keys())}")
    return compute_local_stats_frame(df, stats_spec, filters or {}, resolved_options)
