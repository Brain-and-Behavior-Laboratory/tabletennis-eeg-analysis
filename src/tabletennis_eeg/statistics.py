from __future__ import annotations

import numpy as np
import pandas as pd

from .config import available_model_ids
from .utils import add_fdr_columns, paired_statistics, write_csv


def compute_model_statistics(paths, config: dict, model_id: str):
    model_root = paths.output_root / "specparam" / model_id
    input_file = model_root / "participant_context_features.csv"
    if not input_file.exists():
        raise FileNotFoundError(input_file)
    data = pd.read_csv(input_file)
    data = data[data["fit_success"].astype(bool)].copy()
    rows = []
    for feature in ["aperiodic_exponent", "aperiodic_offset"]:
        feature_rows = []
        for (window, region), group in data.groupby(["window", "region"]):
            wide = group.pivot(index="subject", columns="context", values=feature)
            if not {"human_opponent", "ball_machine"}.issubset(wide.columns):
                continue
            wide = wide.dropna(subset=["human_opponent", "ball_machine"])
            if len(wide) < config["analysis"]["statistics"]["minimum_paired_participants"]:
                continue
            row = {"model_id": model_id, "feature": feature, "window": window, "region": region}
            row.update(paired_statistics(wide["ball_machine"].to_numpy(), wide["human_opponent"].to_numpy()))
            feature_rows.append(row)
        adjusted = add_fdr_columns(
            pd.DataFrame(feature_rows), config["analysis"]["statistics"]["fdr_method"]
        )
        rows.append(adjusted)
    statistics = pd.concat(rows, ignore_index=True).sort_values(
        ["feature", "paired_p_fdr", "window", "region"]
    )
    write_csv(statistics, model_root / "aperiodic_statistics.csv")

    quality_rows = []
    groupings = {
        "overall": [],
        "window": ["window"],
        "window_context": ["window", "context"],
        "window_region": ["window", "region"],
        "window_context_region": ["window", "context", "region"],
    }
    for grouping, columns in groupings.items():
        iterator = [((), data)] if not columns else data.groupby(columns, dropna=False)
        for keys, group in iterator:
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = {"model_id": model_id, "grouping": grouping, "n_cells": len(group)}
            row.update(dict(zip(columns, keys)))
            row.update(
                {
                    "r_squared_mean": group["r_squared"].mean(),
                    "r_squared_median": group["r_squared"].median(),
                    "r_squared_q25": group["r_squared"].quantile(0.25),
                    "r_squared_min": group["r_squared"].min(),
                    "error_mean": group["error"].mean(),
                    "error_median": group["error"].median(),
                    "error_q75": group["error"].quantile(0.75),
                    "error_max": group["error"].max(),
                    "n_r_squared_below_0_8": int((group["r_squared"] < 0.8).sum()),
                    "fraction_r_squared_below_0_8": float((group["r_squared"] < 0.8).mean()),
                }
            )
            quality_rows.append(row)
    quality = pd.DataFrame(quality_rows)
    write_csv(quality, model_root / "fit_quality_summary.csv")

    peak_rows = []
    for (window, context, region), group in data.groupby(["window", "context", "region"]):
        for band in config["analysis"]["frequency_bands_hz"]:
            column = f"{band}_peak_present"
            peak_rows.append(
                {
                    "model_id": model_id,
                    "window": window,
                    "context": context,
                    "region": region,
                    "band": band,
                    "n_model_cells": len(group),
                    "n_peaks_present": int(group[column].astype(bool).sum()),
                    "detection_rate": float(group[column].astype(bool).mean()),
                }
            )
    peaks = pd.DataFrame(peak_rows)
    write_csv(peaks, model_root / "peak_detection_summary.csv")
    return statistics, quality, peaks


def compute_all_model_statistics(paths, config: dict, models: list[str] | None = None):
    models = models or available_model_ids(paths.root)
    outputs = {}
    for model_id in models:
        outputs[model_id] = compute_model_statistics(paths, config, model_id)
    return outputs
