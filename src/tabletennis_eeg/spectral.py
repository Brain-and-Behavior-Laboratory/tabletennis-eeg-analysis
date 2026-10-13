from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .audit import subject_ids
from .channels import sensor_region
from .config import load_model_configuration
from .utils import ensure_dir, frequency_resolution, merge_csv_files, write_csv


def _as_scalar(value, default=math.nan) -> float:
    if isinstance(value, dict):
        if len(value) == 1:
            value = next(iter(value.values()))
        else:
            return float(default)
    array = np.asarray(value)
    if array.size == 0:
        return float(default)
    return float(array.reshape(-1)[0])


def _metric(model, category: str, measure: str, fallback_keys: tuple[str, ...]) -> float:
    try:
        return _as_scalar(model.get_metrics(category, measure))
    except Exception:
        try:
            table = model.to_df()
            if isinstance(table, pd.DataFrame):
                series = table.iloc[0]
            else:
                series = table
            for key in fallback_keys:
                if key in series:
                    return float(series[key])
        except Exception:
            pass
    return math.nan


def _periodic_array(model) -> np.ndarray:
    try:
        peaks = np.asarray(model.get_params("periodic"), dtype=float)
    except Exception:
        return np.empty((0, 3), dtype=float)
    if peaks.ndim == 1:
        if peaks.size >= 3 and np.all(np.isfinite(peaks[:3])):
            peaks = peaks.reshape(1, -1)
        else:
            return np.empty((0, 3), dtype=float)
    return peaks[:, :3] if peaks.ndim == 2 and peaks.shape[1] >= 3 else np.empty((0, 3))


def _residual_bic(model) -> tuple[float, float, int, int]:
    try:
        observed = np.asarray(model.data.power_spectrum, dtype=float)
        fitted = np.asarray(model.results.model.modeled_spectrum, dtype=float)
        residuals = observed - fitted
        rss = float(np.sum(residuals**2))
        n_points = int(len(residuals))
        n_parameters = int(model.results.n_params)
        if rss <= 0 or n_points <= n_parameters:
            return math.nan, rss, n_points, n_parameters
        bic = n_points * np.log(rss / n_points) + n_parameters * np.log(n_points)
        return float(bic), rss, n_points, n_parameters
    except Exception:
        return math.nan, math.nan, 0, 0


def fit_spectrum(frequencies, power, model_settings: dict, analysis_settings: dict, window: str):
    try:
        from specparam import SpectralModel
    except ImportError as exc:
        raise RuntimeError("specparam 2.0.0rc7 is required for spectral fitting.") from exc

    valid = np.isfinite(frequencies) & np.isfinite(power) & (power > 0)
    frequencies = np.asarray(frequencies)[valid]
    power = np.asarray(power)[valid]
    if len(frequencies) < 10:
        return {"fit_success": False, "fit_error": "too_few_frequency_bins"}

    spectral = analysis_settings["spectral_parameterization"]
    model = SpectralModel(
        peak_width_limits=(
            model_settings["minimum_peak_width_hz"][window],
            spectral["maximum_peak_width_hz"],
        ),
        max_n_peaks=model_settings["maximum_peaks"],
        min_peak_height=spectral["minimum_peak_height"],
        peak_threshold=spectral["peak_threshold_sd"],
        aperiodic_mode=spectral["aperiodic_mode"],
        verbose=False,
    )
    row = {"fit_success": False, "fit_error": ""}
    try:
        model.fit(frequencies, power, spectral["fit_frequency_range_hz"])
        aperiodic = np.asarray(model.get_params("aperiodic"), dtype=float).reshape(-1)
        row["aperiodic_offset"] = float(aperiodic[0])
        row["aperiodic_exponent"] = float(aperiodic[-1])
        row["r_squared"] = _metric(model, "gof", "rsquared", ("gof_rsquared", "r_squared"))
        row["error"] = _metric(model, "error", "mae", ("error_mae", "error"))
        peaks = _periodic_array(model)
        row["n_peaks"] = int(len(peaks))
        for band_name, bounds in analysis_settings["frequency_bands_hz"].items():
            in_band = peaks[(peaks[:, 0] >= bounds[0]) & (peaks[:, 0] <= bounds[1])]
            if len(in_band):
                selected = in_band[np.argmax(in_band[:, 1])]
                row[f"{band_name}_peak_cf"] = float(selected[0])
                row[f"{band_name}_peak_power"] = float(selected[1])
                row[f"{band_name}_peak_bw"] = float(selected[2])
                row[f"{band_name}_peak_present"] = True
            else:
                row[f"{band_name}_peak_cf"] = math.nan
                row[f"{band_name}_peak_power"] = math.nan
                row[f"{band_name}_peak_bw"] = math.nan
                row[f"{band_name}_peak_present"] = False
        bic, rss, n_points, n_parameters = _residual_bic(model)
        row.update(
            {
                "residual_bic": bic,
                "residual_rss": rss,
                "n_frequency_points_fit": n_points,
                "n_model_parameters": n_parameters,
                "fit_success": True,
            }
        )
    except Exception as exc:
        row["fit_error"] = repr(exc)
        for name in ["aperiodic_offset", "aperiodic_exponent", "r_squared", "error", "residual_bic", "residual_rss"]:
            row[name] = math.nan
        row["n_peaks"] = 0
        row["n_frequency_points_fit"] = 0
        row["n_model_parameters"] = 0
        for band_name in analysis_settings["frequency_bands_hz"]:
            row[f"{band_name}_peak_cf"] = math.nan
            row[f"{band_name}_peak_power"] = math.nan
            row[f"{band_name}_peak_bw"] = math.nan
            row[f"{band_name}_peak_present"] = False
    return row


def extract_model_subject(paths, config: dict, model_id: str, subject: str, overwrite: bool = False):
    import mne

    model_settings = load_model_configuration(paths.root, model_id)
    analysis = config["analysis"]
    epoch_file = paths.derivative_root / "epochs" / subject / f"{subject}_task-TableTennis_desc-subjecthit_epo.fif"
    if not epoch_file.exists():
        raise FileNotFoundError(epoch_file)
    output_dir = ensure_dir(paths.output_root / "specparam" / model_id / "subjects")
    output_file = output_dir / f"{subject}_{model_id}_features.csv"
    if output_file.exists() and not overwrite:
        return output_file

    epochs = mne.read_epochs(epoch_file, preload=True, verbose="WARNING")
    if epochs.metadata is None:
        raise RuntimeError(f"Epoch metadata missing for {subject}")
    metadata = epochs.metadata.reset_index(drop=True)
    eeg = epochs.copy().pick("eeg")
    region_order = config["sensor_regions"]["ordered_regions"]
    region_labels = np.array([sensor_region(name, config) for name in eeg.ch_names])
    rows = []

    for window_name in model_settings["windows"]:
        window = analysis["windows"][window_name]
        cropped = eeg.copy().crop(window["tmin_seconds"], window["tmax_seconds"])
        n_times = len(cropped.times)
        n_overlap = int(n_times * model_settings["welch_overlap_fraction"])
        spectrum = cropped.compute_psd(
            method="welch",
            fmin=analysis["spectral_parameterization"]["psd_fmin_hz"],
            fmax=analysis["spectral_parameterization"]["psd_fmax_hz"],
            n_fft=n_times,
            n_per_seg=n_times,
            n_overlap=n_overlap,
            window=analysis["spectral_parameterization"]["welch_window"],
            verbose="WARNING",
        )
        powers, frequencies = spectrum.get_data(return_freqs=True)
        for context in ["human_opponent", "ball_machine"]:
            epoch_mask = metadata["context"].to_numpy() == context
            if int(epoch_mask.sum()) < 5:
                continue
            for region in region_order:
                channel_mask = region_labels == region
                if not np.any(channel_mask):
                    continue
                average_power = powers[epoch_mask, :, :][:, channel_mask, :].mean(axis=(0, 1))
                fit = fit_spectrum(frequencies, average_power, model_settings, analysis, window_name)
                row = {
                    "model_id": model_id,
                    "model_label": model_settings["label"],
                    "subject": subject,
                    "context": context,
                    "window": window_name,
                    "region": region,
                    "n_epochs": int(epoch_mask.sum()),
                    "n_channels": int(channel_mask.sum()),
                    "samples": n_times,
                    "n_fft": n_times,
                    "n_per_seg": n_times,
                    "n_overlap": n_overlap,
                    "frequency_resolution_hz": frequency_resolution(float(cropped.info["sfreq"]), n_times),
                    "minimum_peak_width_hz": model_settings["minimum_peak_width_hz"][window_name],
                    "maximum_peak_width_hz": analysis["spectral_parameterization"]["maximum_peak_width_hz"],
                    "maximum_peaks": model_settings["maximum_peaks"],
                    "input_power_scale": "linear",
                }
                row.update(fit)
                rows.append(row)
    write_csv(pd.DataFrame(rows), output_file)
    return output_file


def fit_model_all(paths, config: dict, model_id: str, subjects: list[str] | None = None, **kwargs):
    subjects = subjects or subject_ids(paths.raw_root)
    for subject in subjects:
        extract_model_subject(paths, config, model_id, subject, **kwargs)
    model_root = paths.output_root / "specparam" / model_id
    features = merge_csv_files(model_root.glob("subjects/sub-*.csv"), model_root / "participant_context_features.csv")
    settings_columns = [
        "model_id", "model_label", "window", "samples", "n_fft", "n_per_seg", "n_overlap",
        "frequency_resolution_hz", "minimum_peak_width_hz", "maximum_peak_width_hz", "maximum_peaks", "input_power_scale",
    ]
    settings = features[settings_columns].drop_duplicates().sort_values("window")
    write_csv(settings, model_root / "model_settings.csv")
    return features
