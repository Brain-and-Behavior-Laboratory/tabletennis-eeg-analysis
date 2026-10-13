from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(data: object, path: Path) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(data, tmp, indent=2, sort_keys=True, default=str)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        frame.to_csv(tmp.name, index=False)
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)


def merge_csv_files(files: Iterable[Path], output: Path) -> pd.DataFrame:
    files = sorted(files)
    if not files:
        raise FileNotFoundError(f"No CSV files found for merge into {output}")
    frame = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    write_csv(frame, output)
    return frame


def safe_wilcoxon(machine: np.ndarray, human: np.ndarray) -> tuple[float, float]:
    difference = np.asarray(machine, dtype=float) - np.asarray(human, dtype=float)
    if difference.size == 0 or np.allclose(difference, 0.0):
        return math.nan, math.nan
    try:
        result = stats.wilcoxon(machine, human, alternative="two-sided")
        return float(result.statistic), float(result.pvalue)
    except ValueError:
        return math.nan, math.nan


def cohens_dz(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    standard_deviation = np.nanstd(differences, ddof=1)
    if not np.isfinite(standard_deviation) or standard_deviation == 0:
        return math.nan
    return float(np.nanmean(differences) / standard_deviation)


def paired_statistics(machine: np.ndarray, human: np.ndarray) -> dict[str, float | int]:
    machine = np.asarray(machine, dtype=float)
    human = np.asarray(human, dtype=float)
    valid = np.isfinite(machine) & np.isfinite(human)
    machine = machine[valid]
    human = human[valid]
    differences = machine - human
    if len(differences) < 2:
        raise ValueError("At least two finite paired observations are required.")
    t_result = stats.ttest_rel(machine, human, nan_policy="omit")
    wilcoxon_statistic, wilcoxon_p = safe_wilcoxon(machine, human)
    return {
        "n_participants": int(len(differences)),
        "human_mean": float(np.mean(human)),
        "machine_mean": float(np.mean(machine)),
        "machine_minus_human_mean": float(np.mean(differences)),
        "machine_minus_human_median": float(np.median(differences)),
        "difference_sd": float(np.std(differences, ddof=1)),
        "paired_t": float(t_result.statistic),
        "paired_p": float(t_result.pvalue),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p": wilcoxon_p,
        "cohens_dz": cohens_dz(differences),
        "n_machine_higher": int(np.sum(differences > 0)),
        "n_machine_lower": int(np.sum(differences < 0)),
        "n_equal": int(np.sum(differences == 0)),
        "fraction_machine_higher": float(np.mean(differences > 0)),
        "fraction_machine_lower": float(np.mean(differences < 0)),
    }


def add_fdr_columns(frame: pd.DataFrame, method: str = "fdr_bh") -> pd.DataFrame:
    frame = frame.copy()
    frame["paired_p_fdr"] = multipletests(frame["paired_p"].fillna(1.0), method=method)[1]
    frame["wilcoxon_p_fdr"] = multipletests(
        frame["wilcoxon_p"].fillna(1.0), method=method
    )[1]
    return frame


def bootstrap_paired_difference(
    differences: np.ndarray,
    n_resamples: int,
    seed: int,
) -> dict[str, float | int]:
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if len(differences) < 2:
        raise ValueError("At least two finite differences are required.")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(n_resamples, len(differences)))
    samples = differences[indices]
    means = samples.mean(axis=1)
    sds = samples.std(axis=1, ddof=1)
    dz = np.divide(means, sds, out=np.full_like(means, np.nan), where=sds > 0)
    return {
        "n_participants": int(len(differences)),
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "observed_mean_difference": float(np.mean(differences)),
        "mean_difference_ci_2_5": float(np.quantile(means, 0.025)),
        "mean_difference_ci_97_5": float(np.quantile(means, 0.975)),
        "observed_cohens_dz": cohens_dz(differences),
        "cohens_dz_ci_2_5": float(np.nanquantile(dz, 0.025)),
        "cohens_dz_ci_97_5": float(np.nanquantile(dz, 0.975)),
        "fraction_mean_positive": float(np.mean(means > 0)),
    }


def inclusive_sample_count(tmin: float, tmax: float, sfreq: float) -> int:
    return int(round((tmax - tmin) * sfreq)) + 1


def frequency_resolution(sfreq: float, n_fft: int) -> float:
    return float(sfreq / n_fft)
