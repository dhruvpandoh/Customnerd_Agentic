from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"patent_id", "claim", "litigated", "flagged"}
OPTIONAL_PHRASE_COLUMN = "phrase_within_claim"


@dataclass(frozen=True)
class CountMetrics:
    n1: int  # litigated and flagged by system
    n2: int  # flagged by system but not litigated
    n3: int  # litigated but not flagged by system
    precision: float
    recall: float
    f1: float


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def metrics_from_counts(n1: int, n2: int, n3: int) -> CountMetrics:
    precision = safe_divide(n1, n1 + n2)
    recall = safe_divide(n1, n1 + n3)
    f1 = safe_divide(2 * precision * recall, precision + recall)

    return CountMetrics(
        n1=int(n1),
        n2=int(n2),
        n3=int(n3),
        precision=precision,
        recall=recall,
        f1=f1,
    )


def validate_binary(series: pd.Series, column_name: str) -> pd.Series:
    numeric = pd.to_numeric(series, errors="raise").astype(int)
    invalid = ~numeric.isin([0, 1])

    if invalid.any():
        bad = sorted(numeric[invalid].unique().tolist())
        raise ValueError(f"{column_name} must contain only 0/1. Found: {bad}")

    return numeric


def load_system_csv(path: Path, system_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    df = df.copy()
    df["patent_id"] = df["patent_id"].astype(str).str.strip()
    df["claim"] = df["claim"].astype(str).str.strip()
    df["litigated"] = validate_binary(df["litigated"], "litigated")
    df["flagged"] = validate_binary(df["flagged"], "flagged")

    if OPTIONAL_PHRASE_COLUMN in df.columns:
        df[OPTIONAL_PHRASE_COLUMN] = (
            df[OPTIONAL_PHRASE_COLUMN].fillna("").astype(str).str.strip()
        )

    return df.rename(columns={"flagged": f"flagged_by_{system_name}"})


def merge_system_outputs(system_files: dict[str, Path]) -> pd.DataFrame:
    frames = {
        name: load_system_csv(path, name)
        for name, path in system_files.items()
    }

    use_phrase = any(OPTIONAL_PHRASE_COLUMN in df.columns for df in frames.values())
    keys = ["patent_id", "claim"] + ([OPTIONAL_PHRASE_COLUMN] if use_phrase else [])

    for df in frames.values():
        if use_phrase and OPTIONAL_PHRASE_COLUMN not in df.columns:
            df[OPTIONAL_PHRASE_COLUMN] = ""

    merged = None

    for name, df in frames.items():
        pred_col = f"flagged_by_{name}"
        part = df[keys + ["litigated", pred_col]].drop_duplicates(
            subset=keys,
            keep="last",
        )

        if merged is None:
            merged = part
            continue

        merged = merged.merge(
            part,
            on=keys,
            how="outer",
            suffixes=("", f"_{name}"),
        )

        duplicate_truth = f"litigated_{name}"
        if duplicate_truth in merged.columns:
            both_present = merged["litigated"].notna() & merged[duplicate_truth].notna()
            conflicts = both_present & (merged["litigated"] != merged[duplicate_truth])

            if conflicts.any():
                sample = merged.loc[conflicts, keys].head(5).to_dict("records")
                raise ValueError(
                    f"Conflicting litigated labels while merging {name}. "
                    f"Examples: {sample}"
                )

            merged["litigated"] = merged["litigated"].fillna(merged[duplicate_truth])
            merged = merged.drop(columns=[duplicate_truth])

    if merged is None:
        raise ValueError("No system files provided.")

    merged["litigated"] = validate_binary(merged["litigated"], "litigated")

    prediction_cols = [c for c in merged.columns if c.startswith("flagged_by_")]

    for col in prediction_cols:
        if merged[col].isna().any():
            count = int(merged[col].isna().sum())
            raise ValueError(
                f"{col} has {count} missing predictions after the union merge. "
                "Resolve missing rows before evaluation."
            )

        merged[col] = validate_binary(merged[col], col)

    return merged.sort_values(keys).reset_index(drop=True)


def patent_counts_for_system(df: pd.DataFrame, system_col: str) -> pd.DataFrame:
    """
    For each patent and system, compute:
    N1 = litigated and flagged by system
    N2 = flagged by system but not litigated
    N3 = litigated but not flagged by system
    """
    temp = df[["patent_id", "litigated", system_col]].copy()

    temp["n1"] = ((temp["litigated"] == 1) & (temp[system_col] == 1)).astype(int)
    temp["n2"] = ((temp["litigated"] == 0) & (temp[system_col] == 1)).astype(int)
    temp["n3"] = ((temp["litigated"] == 1) & (temp[system_col] == 0)).astype(int)

    out = (
        temp.groupby("patent_id", as_index=False)[["n1", "n2", "n3"]]
        .sum()
        .rename(
            columns={
                "n1": f"{system_col}_n1",
                "n2": f"{system_col}_n2",
                "n3": f"{system_col}_n3",
            }
        )
    )

    return out


def build_patent_level_counts(df: pd.DataFrame) -> pd.DataFrame:
    prediction_cols = [c for c in df.columns if c.startswith("flagged_by_")]

    patent_table = pd.DataFrame({"patent_id": sorted(df["patent_id"].unique())})

    for col in prediction_cols:
        counts = patent_counts_for_system(df, col)
        patent_table = patent_table.merge(counts, on="patent_id", how="left")

    count_cols = [c for c in patent_table.columns if c != "patent_id"]
    patent_table[count_cols] = patent_table[count_cols].fillna(0).astype(int)

    return patent_table


def system_metrics_from_patent_counts(patent_counts: pd.DataFrame) -> pd.DataFrame:
    systems = sorted(
        col.removeprefix("flagged_by_").removesuffix("_n1")
        for col in patent_counts.columns
        if col.startswith("flagged_by_") and col.endswith("_n1")
    )

    rows = []

    for system in systems:
        prefix = f"flagged_by_{system}"
        n1 = int(patent_counts[f"{prefix}_n1"].sum())
        n2 = int(patent_counts[f"{prefix}_n2"].sum())
        n3 = int(patent_counts[f"{prefix}_n3"].sum())

        m = metrics_from_counts(n1, n2, n3)

        rows.append(
            {
                "system": system,
                "n1_correctly_identified": m.n1,
                "n2_flagged_not_litigated": m.n2,
                "n3_litigated_not_flagged": m.n3,
                "precision": m.precision,
                "recall": m.recall,
                "f1": m.f1,
                "n_patents": len(patent_counts),
            }
        )

    return pd.DataFrame(rows)


def metric_from_patent_table(
    patent_counts: pd.DataFrame,
    system: str,
    metric: str,
) -> float:
    prefix = f"flagged_by_{system}"

    n1 = int(patent_counts[f"{prefix}_n1"].sum())
    n2 = int(patent_counts[f"{prefix}_n2"].sum())
    n3 = int(patent_counts[f"{prefix}_n3"].sum())

    return float(getattr(metrics_from_counts(n1, n2, n3), metric))


def swap_patent_counts(
    patent_counts: pd.DataFrame,
    system_a: str,
    system_b: str,
    swap_mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For paired randomization:
    each patent is one unit.
    With probability 0.5, swap the full N1/N2/N3 tuple between systems.
    """
    a_prefix = f"flagged_by_{system_a}"
    b_prefix = f"flagged_by_{system_b}"

    a_cols = [f"{a_prefix}_n1", f"{a_prefix}_n2", f"{a_prefix}_n3"]
    b_cols = [f"{b_prefix}_n1", f"{b_prefix}_n2", f"{b_prefix}_n3"]

    permuted = patent_counts.copy()

    for a_col, b_col in zip(a_cols, b_cols):
        a_values = patent_counts[a_col].to_numpy()
        b_values = patent_counts[b_col].to_numpy()

        permuted[a_col] = np.where(swap_mask, b_values, a_values)
        permuted[b_col] = np.where(swap_mask, a_values, b_values)

    return permuted, patent_counts


def paired_randomization_test_by_patent(
    patent_counts: pd.DataFrame,
    baseline_system: str,
    comparison_system: str,
    metric: str,
    iterations: int = 10_000,
    seed: int = 42,
) -> dict[str, float]:
    observed = (
        metric_from_patent_table(patent_counts, baseline_system, metric)
        - metric_from_patent_table(patent_counts, comparison_system, metric)
    )

    rng = np.random.default_rng(seed)
    extreme = 0
    n_patents = len(patent_counts)

    for _ in range(iterations):
        swap_mask = rng.random(n_patents) < 0.5

        permuted, _ = swap_patent_counts(
            patent_counts,
            baseline_system,
            comparison_system,
            swap_mask,
        )

        diff = (
            metric_from_patent_table(permuted, baseline_system, metric)
            - metric_from_patent_table(permuted, comparison_system, metric)
        )

        # Two-sided test. If Dennis wants one-sided, this can be changed.
        if abs(diff) >= abs(observed):
            extreme += 1

    return {
        "observed_difference": observed,
        "p_value": (extreme + 1) / (iterations + 1),
    }


def bootstrap_difference_by_patent(
    patent_counts: pd.DataFrame,
    baseline_system: str,
    comparison_system: str,
    metric: str,
    iterations: int = 10_000,
    lower_percentile: float = 10.0,
    upper_percentile: float = 90.0,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n_patents = len(patent_counts)

    if n_patents == 0:
        raise ValueError("Cannot bootstrap an empty patent table.")

    diffs = np.empty(iterations, dtype=float)

    for i in range(iterations):
        sample_indices = rng.integers(0, n_patents, size=n_patents)
        sample = patent_counts.iloc[sample_indices].reset_index(drop=True)

        diffs[i] = (
            metric_from_patent_table(sample, baseline_system, metric)
            - metric_from_patent_table(sample, comparison_system, metric)
        )

    return {
        "bootstrap_mean_difference": float(np.mean(diffs)),
        "ci_lower": float(np.percentile(diffs, lower_percentile)),
        "ci_upper": float(np.percentile(diffs, upper_percentile)),
    }


def pairwise_comparison_table(
    patent_counts: pd.DataFrame,
    baseline_system: str = "patentnerd",
    iterations: int = 10_000,
    seed: int = 42,
) -> pd.DataFrame:
    systems = sorted(
        col.removeprefix("flagged_by_").removesuffix("_n1")
        for col in patent_counts.columns
        if col.startswith("flagged_by_") and col.endswith("_n1")
    )

    if baseline_system not in systems:
        raise ValueError(
            f"Baseline system '{baseline_system}' not found. "
            f"Available systems: {systems}"
        )

    rows = []

    for comparison_system in systems:
        if comparison_system == baseline_system:
            continue

        for metric in ("precision", "recall", "f1"):
            paired = paired_randomization_test_by_patent(
                patent_counts=patent_counts,
                baseline_system=baseline_system,
                comparison_system=comparison_system,
                metric=metric,
                iterations=iterations,
                seed=seed,
            )

            boot = bootstrap_difference_by_patent(
                patent_counts=patent_counts,
                baseline_system=baseline_system,
                comparison_system=comparison_system,
                metric=metric,
                iterations=iterations,
                lower_percentile=10.0,
                upper_percentile=90.0,
                seed=seed,
            )

            rows.append(
                {
                    "baseline_system": baseline_system,
                    "comparison_system": comparison_system,
                    "metric": metric,
                    "observed_difference_baseline_minus_comparison": paired[
                        "observed_difference"
                    ],
                    "paired_randomization_p_value": paired["p_value"],
                    "bootstrap_mean_difference": boot["bootstrap_mean_difference"],
                    "bootstrap_ci_10th_percentile": boot["ci_lower"],
                    "bootstrap_ci_90th_percentile": boot["ci_upper"],
                    "iterations": iterations,
                    "unit_of_resampling": "patent",
                }
            )

    return pd.DataFrame(rows)


def parse_system_args(items: list[str]) -> dict[str, Path]:
    result = {}

    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --system '{item}'. Use NAME=/path/to/file.csv")

        name, raw_path = item.split("=", 1)
        name = name.strip().lower().replace(" ", "_")
        path = Path(raw_path).expanduser().resolve()

        if not path.exists():
            raise FileNotFoundError(path)

        result[name] = path

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PatentNerd and comparison systems using Professor Shasha's "
            "per-patent N1/N2/N3 method."
        )
    )

    parser.add_argument(
        "--system",
        action="append",
        required=True,
        help="NAME=/path/to/file.csv; repeat once per system",
    )
    parser.add_argument("--baseline", default="patentnerd")
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="evaluation_output")

    args = parser.parse_args()

    systems = parse_system_args(args.system)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    combined_rows = merge_system_outputs(systems)
    patent_counts = build_patent_level_counts(combined_rows)
    metrics = system_metrics_from_patent_counts(patent_counts)
    comparisons = pairwise_comparison_table(
        patent_counts,
        baseline_system=args.baseline,
        iterations=args.iterations,
        seed=args.seed,
    )

    combined_rows.to_csv(output_dir / "combined_union_table.csv", index=False)
    patent_counts.to_csv(output_dir / "patent_level_n1_n2_n3.csv", index=False)
    metrics.to_csv(output_dir / "system_metrics.csv", index=False)
    comparisons.to_csv(output_dir / "paired_and_bootstrap_results.csv", index=False)

    print(f"Wrote {output_dir / 'combined_union_table.csv'}")
    print(f"Wrote {output_dir / 'patent_level_n1_n2_n3.csv'}")
    print(f"Wrote {output_dir / 'system_metrics.csv'}")
    print(f"Wrote {output_dir / 'paired_and_bootstrap_results.csv'}")


if __name__ == "__main__":
    main()