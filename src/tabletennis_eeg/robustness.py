from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .utils import (
    add_fdr_columns,
    bootstrap_paired_difference,
    cohens_dz,
    paired_statistics,
    write_csv,
)

SHORT_MODELS = ["initial_short_window", "resolution_aware_max6", "resolution_aware_max3"]
ALL_FOCAL_MODELS = SHORT_MODELS + ["two_second_max6", "two_second_aperiodic_only"]


def _features(paths, model_id: str) -> pd.DataFrame:
    path = paths.output_root / "specparam" / model_id / "participant_context_features.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    data = pd.read_csv(path)
    return data[data["fit_success"].astype(bool)].copy()


def _subject_differences(data: pd.DataFrame, feature: str, window: str, region: str) -> pd.Series:
    selected = data[(data["window"] == window) & (data["region"] == region)]
    wide = selected.pivot(index="subject", columns="context", values=feature)
    wide = wide.dropna(subset=["human_opponent", "ball_machine"])
    return (wide["ball_machine"] - wide["human_opponent"]).sort_index()


def _family_statistics(data: pd.DataFrame, feature: str, fdr_method: str) -> pd.DataFrame:
    rows = []
    for (window, region), group in data.groupby(["window", "region"]):
        wide = group.pivot(index="subject", columns="context", values=feature)
        if not {"human_opponent", "ball_machine"}.issubset(wide.columns):
            continue
        wide = wide.dropna(subset=["human_opponent", "ball_machine"])
        row = {"window": window, "region": region}
        row.update(paired_statistics(wide["ball_machine"].to_numpy(), wide["human_opponent"].to_numpy()))
        rows.append(row)
    return add_fdr_columns(pd.DataFrame(rows), fdr_method)


def _concordance(left: pd.Series, right: pd.Series, left_model: str, right_model: str) -> dict:
    joined = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    difference = joined["right"] - joined["left"]
    return {
        "left_model": left_model,
        "right_model": right_model,
        "n_participants": len(joined),
        "pearson_r": stats.pearsonr(joined["left"], joined["right"]).statistic,
        "spearman_rho": stats.spearmanr(joined["left"], joined["right"]).statistic,
        "mean_difference_right_minus_left": difference.mean(),
        "median_difference_right_minus_left": difference.median(),
        "mean_absolute_difference": difference.abs().mean(),
        "maximum_absolute_difference": difference.abs().max(),
        "n_direction_changes": int((np.sign(joined["left"]) != np.sign(joined["right"])).sum()),
        "direction_agreement_fraction": float((np.sign(joined["left"]) == np.sign(joined["right"])).mean()),
    }


def _permutation_spearman(x: np.ndarray, y: np.ndarray, n_resamples: int, seed: int):
    x_rank = stats.rankdata(x)
    y_rank = stats.rankdata(y)
    x_centered = x_rank - x_rank.mean()
    y_centered = y_rank - y_rank.mean()
    denominator = np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2))
    observed = float(np.sum(x_centered * y_centered) / denominator)
    rng = np.random.default_rng(seed)
    indices = np.array([rng.permutation(len(y)) for _ in range(n_resamples)])
    permuted = y_centered[indices]
    correlations = (permuted @ x_centered) / denominator
    p_value = float((np.sum(np.abs(correlations) >= abs(observed)) + 1) / (n_resamples + 1))
    return observed, p_value


def _bootstrap_intercept(x: np.ndarray, y: np.ndarray, n_resamples: int, seed: int):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(x), size=(n_resamples, len(x)))
    xb, yb = x[indices], y[indices]
    x_mean, y_mean = xb.mean(axis=1), yb.mean(axis=1)
    x_centered = xb - x_mean[:, None]
    denominator = np.sum(x_centered**2, axis=1)
    slopes = np.divide(
        np.sum(x_centered * (yb - y_mean[:, None]), axis=1),
        denominator,
        out=np.full(n_resamples, np.nan),
        where=denominator > 0,
    )
    intercepts = y_mean - slopes * x_mean
    return {
        "intercept_at_zero_ci_2_5": float(np.nanquantile(intercepts, 0.025)),
        "intercept_at_zero_ci_97_5": float(np.nanquantile(intercepts, 0.975)),
        "bootstrap_fraction_intercept_positive": float(np.nanmean(intercepts > 0)),
    }


def compute_robustness(paths, config: dict):
    settings = config["analysis"]["statistics"]
    short_target = config["analysis"]["focal_robustness_target"]
    broad_target = config["analysis"]["broad_window_target"]
    output_root = paths.output_root / "robustness"
    output_root.mkdir(parents=True, exist_ok=True)

    model_data = {model: _features(paths, model) for model in ALL_FOCAL_MODELS}
    target_by_model = {
        model: short_target if model in SHORT_MODELS else broad_target for model in ALL_FOCAL_MODELS
    }
    focal_rows, bootstrap_rows, subject_frame = [], [], None

    differences_by_model = {}
    for index, model in enumerate(ALL_FOCAL_MODELS):
        target = target_by_model[model]
        data = model_data[model]
        family = _family_statistics(data, target["feature"], settings["fdr_method"])
        focal = family[(family["window"] == target["window"]) & (family["region"] == target["region"])].iloc[0].to_dict()
        focal.update({"model_id": model, "feature": target["feature"]})
        focal_rows.append(focal)
        differences = _subject_differences(data, target["feature"], target["window"], target["region"])
        differences_by_model[model] = differences
        boot = bootstrap_paired_difference(
            differences.to_numpy(), settings["bootstrap_resamples"], settings["random_seed"] + index
        )
        boot.update({"model_id": model, "window": target["window"], "region": target["region"]})
        bootstrap_rows.append(boot)
        column = differences.rename(f"{model}_difference")
        subject_frame = column.to_frame() if subject_frame is None else subject_frame.join(column, how="outer")

    write_csv(pd.DataFrame(focal_rows), output_root / "focal_model_comparison.csv")
    write_csv(pd.DataFrame(bootstrap_rows), output_root / "bootstrap_summary.csv")
    subject_frame = subject_frame.reset_index().rename(columns={"index": "subject"})
    write_csv(subject_frame, output_root / "subject_differences.csv")

    loso_rows = []
    for model in ALL_FOCAL_MODELS:
        target = target_by_model[model]
        data = model_data[model]
        for omitted in sorted(data["subject"].unique()):
            reduced = data[data["subject"] != omitted]
            family = _family_statistics(reduced, target["feature"], settings["fdr_method"])
            focal = family[(family["window"] == target["window"]) & (family["region"] == target["region"])].iloc[0].to_dict()
            focal.update({"model_id": model, "omitted_subject": omitted, "feature": target["feature"]})
            loso_rows.append(focal)
    write_csv(pd.DataFrame(loso_rows), output_root / "leave_one_participant_out.csv")

    concordance_pairs = [
        ("initial_short_window", "resolution_aware_max6"),
        ("resolution_aware_max6", "resolution_aware_max3"),
        ("two_second_max6", "two_second_aperiodic_only"),
    ]
    concordance = pd.DataFrame(
        [_concordance(differences_by_model[left], differences_by_model[right], left, right) for left, right in concordance_pairs]
    )
    write_csv(concordance, output_root / "model_concordance.csv")

    max6 = model_data["resolution_aware_max6"]
    max3 = model_data["resolution_aware_max3"]
    keys = ["subject", "context", "window", "region"]
    comparison = max6.merge(max3, on=keys, suffixes=("_max6", "_max3"), validate="one_to_one")
    comparison["delta_bic_max3_minus_max6"] = comparison["residual_bic_max3"] - comparison["residual_bic_max6"]
    comparison["delta_r_squared_max3_minus_max6"] = comparison["r_squared_max3"] - comparison["r_squared_max6"]
    comparison["delta_error_max3_minus_max6"] = comparison["error_max3"] - comparison["error_max6"]
    comparison["delta_n_peaks_max3_minus_max6"] = comparison["n_peaks_max3"] - comparison["n_peaks_max6"]
    write_csv(comparison, output_root / "max6_max3_cell_comparison.csv")

    tolerance = settings["bic_tie_tolerance"]
    bic_rows = []
    for grouping, columns in {"overall": [], "window": ["window"]}.items():
        iterator = [((), comparison)] if not columns else comparison.groupby(columns)
        for keys_value, group in iterator:
            if not isinstance(keys_value, tuple):
                keys_value = (keys_value,)
            delta = group["delta_bic_max3_minus_max6"]
            row = {
                "grouping": grouping,
                "n_cells": len(group),
                "mean_delta_bic_max3_minus_max6": delta.mean(),
                "median_delta_bic_max3_minus_max6": delta.median(),
                "q25_delta_bic_max3_minus_max6": delta.quantile(0.25),
                "q75_delta_bic_max3_minus_max6": delta.quantile(0.75),
                "n_max3_lower_bic": int((delta < -tolerance).sum()),
                "n_max6_lower_bic": int((delta > tolerance).sum()),
                "n_bic_ties": int((delta.abs() <= tolerance).sum()),
                "mean_delta_r_squared_max3_minus_max6": group["delta_r_squared_max3_minus_max6"].mean(),
                "mean_delta_error_max3_minus_max6": group["delta_error_max3_minus_max6"].mean(),
                "mean_delta_n_peaks_max3_minus_max6": group["delta_n_peaks_max3_minus_max6"].mean(),
            }
            row.update(dict(zip(columns, keys_value)))
            bic_rows.append(row)
    write_csv(pd.DataFrame(bic_rows), output_root / "bic_summary.csv")

    broad = model_data["two_second_max6"]
    target = broad_target
    selected = broad[(broad["window"] == target["window"]) & (broad["region"] == target["region"])]
    wide_effect = selected.pivot(index="subject", columns="context", values=target["feature"])
    wide_r2 = selected.pivot(index="subject", columns="context", values="r_squared")
    wide_error = selected.pivot(index="subject", columns="context", values="error")
    influence_rows, diagnostic_rows, intercept_rows = [], [], []
    effect = (wide_effect["ball_machine"] - wide_effect["human_opponent"]).sort_index()
    for predictor_index, (predictor, wide_quality) in enumerate([("delta_r_squared", wide_r2), ("delta_error", wide_error)]):
        quality = (wide_quality["ball_machine"] - wide_quality["human_opponent"]).reindex(effect.index)
        valid = effect.notna() & quality.notna()
        x, y = quality[valid].to_numpy(), effect[valid].to_numpy()
        rho, permutation_p = _permutation_spearman(x, y, settings["permutation_resamples"], settings["random_seed"] + 100 + predictor_index)
        design = sm.add_constant(x)
        fit = sm.OLS(y, design).fit(cov_type="HC3")
        ordinary_fit = sm.OLS(y, design).fit()
        influence = ordinary_fit.get_influence()
        cooks = influence.cooks_distance[0]
        threshold = 4.0 / len(y)
        theil = stats.theilslopes(y, x, alpha=0.95)
        bootstrap_intercept = _bootstrap_intercept(x, y, settings["bootstrap_resamples"], settings["random_seed"] + 200 + predictor_index)
        row = {
            "predictor": predictor,
            "n_participants": len(y),
            "spearman_rho": rho,
            "spearman_permutation_p": permutation_p,
            "ols_intercept_at_zero": fit.params[0],
            "ols_intercept_hc3_p": fit.pvalues[0],
            "ols_intercept_hc3_ci_lower": fit.conf_int()[0, 0],
            "ols_intercept_hc3_ci_upper": fit.conf_int()[0, 1],
            "ols_slope": fit.params[1],
            "ols_slope_hc3_p": fit.pvalues[1],
            "ols_slope_hc3_ci_lower": fit.conf_int()[1, 0],
            "ols_slope_hc3_ci_upper": fit.conf_int()[1, 1],
            "ols_r_squared": ordinary_fit.rsquared,
            "theil_sen_intercept": theil.intercept,
            "theil_sen_slope": theil.slope,
            "theil_sen_slope_ci_lower": theil.low_slope,
            "theil_sen_slope_ci_upper": theil.high_slope,
            "maximum_cooks_distance": cooks.max(),
            "n_cooks_distance_above_4_over_n": int((cooks > threshold).sum()),
            "cooks_distance_threshold_4_over_n": threshold,
        }
        influence_rows.append(row)
        intercept_rows.append({"predictor": predictor, **bootstrap_intercept})
        for subject, xv, yv, cook in zip(effect[valid].index, x, y, cooks):
            diagnostic_rows.append(
                {
                    "predictor": predictor,
                    "subject": subject,
                    "quality_difference": xv,
                    "effect_difference": yv,
                    "cooks_distance": cook,
                    "cooks_distance_above_4_over_n": bool(cook > threshold),
                }
            )
    write_csv(pd.DataFrame(influence_rows), output_root / "fit_quality_influence_summary.csv")
    write_csv(pd.DataFrame(diagnostic_rows), output_root / "fit_quality_influence_subject_diagnostics.csv")
    write_csv(pd.DataFrame(intercept_rows), output_root / "quality_neutral_intercept_bootstrap.csv")

    max6_broad = model_data["two_second_max6"]
    ap_only = model_data["two_second_aperiodic_only"]
    broad_comparison = max6_broad.merge(ap_only, on=keys, suffixes=("_max6", "_aperiodic_only"), validate="one_to_one")
    broad_comparison["delta_aperiodic_only_minus_max6_r_squared"] = broad_comparison["r_squared_aperiodic_only"] - broad_comparison["r_squared_max6"]
    broad_comparison["delta_aperiodic_only_minus_max6_error"] = broad_comparison["error_aperiodic_only"] - broad_comparison["error_max6"]
    write_csv(broad_comparison, output_root / "aperiodic_only_fit_quality_changes.csv")
    return {
        "focal": pd.DataFrame(focal_rows),
        "bootstrap": pd.DataFrame(bootstrap_rows),
        "loso": pd.DataFrame(loso_rows),
        "concordance": concordance,
    }
