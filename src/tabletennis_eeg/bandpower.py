from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .audit import subject_ids
from .channels import classify_channel_names, sensor_region
from .utils import add_fdr_columns, ensure_dir, merge_csv_files, paired_statistics, write_csv


def rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(data)))) if data.size else np.nan


def extract_subject_features(paths, config: dict, subject: str, overwrite: bool = False):
    import mne

    epoch_file = (
        paths.derivative_root
        / "epochs"
        / subject
        / f"{subject}_task-TableTennis_desc-subjecthit_epo.fif"
    )
    if not epoch_file.exists():
        raise FileNotFoundError(epoch_file)
    output_dir = ensure_dir(paths.output_root / "bandpower" / "trial_level")
    band_path = output_dir / f"{subject}_bandpower_features.csv"
    nuisance_path = output_dir / f"{subject}_nuisance_features.csv"
    if band_path.exists() and nuisance_path.exists() and not overwrite:
        return band_path, nuisance_path

    settings = config["analysis"]
    epochs = mne.read_epochs(epoch_file, preload=True, verbose="WARNING")
    if epochs.metadata is None:
        raise RuntimeError(f"Epoch metadata are missing for {subject}")
    metadata = epochs.metadata.reset_index(drop=True).copy()
    groups = classify_channel_names(epochs.ch_names, config)
    nuisance_rows, band_rows = [], []

    for window_name in ["pre_hit", "impact", "post_hit"]:
        window = settings["windows"][window_name]
        cropped = epochs.copy().crop(window["tmin_seconds"], window["tmax_seconds"])
        data = cropped.get_data(copy=True)
        index = {name: position for position, name in enumerate(cropped.ch_names)}
        group_indices = {
            group: [index[name] for name in names if name in index]
            for group, names in groups.items()
        }
        for epoch_index in range(len(cropped)):
            nuisance_rows.append(
                {
                    "subject": subject,
                    "epoch_index": epoch_index,
                    "window": window_name,
                    "context": metadata.loc[epoch_index, "context"],
                    "trial_type": metadata.loc[epoch_index, "trial_type"],
                    "trial_id": metadata.loc[epoch_index, "trial_id"],
                    "noise_rms": rms(data[epoch_index, group_indices["noise"], :]),
                    "emg_rms": rms(data[epoch_index, group_indices["emg"], :]),
                    "motion_rms": rms(data[epoch_index, group_indices["motion"], :]),
                }
            )

        eeg = cropped.copy().pick("eeg")
        n_times = len(eeg.times)
        n_fft = min(settings["bandpower"]["maximum_fft_length_samples"], n_times)
        spectrum = eeg.compute_psd(
            method="welch",
            fmin=settings["bandpower"]["psd_fmin_hz"],
            fmax=settings["bandpower"]["psd_fmax_hz"],
            n_fft=n_fft,
            n_per_seg=n_fft,
            n_overlap=int(n_fft * settings["bandpower"]["overlap_fraction"]),
            window=settings["bandpower"]["welch_window"],
            verbose="WARNING",
        )
        powers, frequencies = spectrum.get_data(return_freqs=True)
        log_power = np.log10(powers + np.finfo(float).eps)
        region_labels = np.array([sensor_region(name, config) for name in eeg.ch_names])

        for epoch_index in range(len(eeg)):
            for band_name, bounds in settings["frequency_bands_hz"].items():
                frequency_mask = (frequencies >= bounds[0]) & (frequencies <= bounds[1])
                for region in settings["sensor_regions_order"]:
                    channel_mask = region_labels == region
                    if not np.any(channel_mask):
                        continue
                    band_rows.append(
                        {
                            "subject": subject,
                            "epoch_index": epoch_index,
                            "window": window_name,
                            "context": metadata.loc[epoch_index, "context"],
                            "trial_type": metadata.loc[epoch_index, "trial_type"],
                            "trial_id": metadata.loc[epoch_index, "trial_id"],
                            "band": band_name,
                            "region": region,
                            "log_power": float(
                                log_power[epoch_index, channel_mask, :][:, frequency_mask].mean()
                            ),
                        }
                    )

    write_csv(pd.DataFrame(band_rows), band_path)
    write_csv(pd.DataFrame(nuisance_rows), nuisance_path)
    return band_path, nuisance_path


def _with_region_order(config: dict) -> dict:
    analysis = config["analysis"]
    analysis["sensor_regions_order"] = config["sensor_regions"]["ordered_regions"]
    return config


def extract_all_features(paths, config: dict, subjects: list[str] | None = None, **kwargs):
    config = _with_region_order(config)
    subjects = subjects or subject_ids(paths.raw_root)
    for subject in subjects:
        extract_subject_features(paths, config, subject, **kwargs)
    root = paths.output_root / "bandpower" / "trial_level"
    bands = merge_csv_files(root.glob("sub-*_bandpower_features.csv"), paths.output_root / "bandpower" / "trial_level_bandpower_features.csv")
    nuisance = merge_csv_files(root.glob("sub-*_nuisance_features.csv"), paths.output_root / "bandpower" / "trial_level_nuisance_features.csv")
    return bands, nuisance


def residualize_and_test(paths, config: dict):
    output_root = ensure_dir(paths.output_root / "bandpower")
    band = pd.read_csv(output_root / "trial_level_bandpower_features.csv")
    nuisance = pd.read_csv(output_root / "trial_level_nuisance_features.csv")
    keys = ["subject", "epoch_index", "window", "context", "trial_type", "trial_id"]
    data = band.merge(nuisance, on=keys, how="left", validate="many_to_one")
    for column in ["noise_rms", "emg_rms", "motion_rms"]:
        data[f"{column}_z"] = (data[column] - data[column].mean()) / data[column].std(ddof=1)

    residualized = []
    for (window, band_name, region), group in data.groupby(["window", "band", "region"]):
        group = group.copy()
        valid = group[["log_power", "noise_rms_z", "emg_rms_z", "motion_rms_z"]].dropna().index
        group["residualized_log_power"] = np.nan
        design = sm.add_constant(group.loc[valid, ["noise_rms_z", "emg_rms_z", "motion_rms_z"]])
        model = sm.OLS(group.loc[valid, "log_power"], design).fit()
        group.loc[valid, "residualized_log_power"] = model.resid
        residualized.append(group)
    residualized_data = pd.concat(residualized, ignore_index=True)

    participant_means = (
        residualized_data.groupby(["subject", "context", "window", "band", "region"], as_index=False)
        .agg(
            residualized_log_power_mean=("residualized_log_power", "mean"),
            raw_log_power_mean=("log_power", "mean"),
            n_epochs=("residualized_log_power", "count"),
            noise_rms_mean=("noise_rms", "mean"),
            emg_rms_mean=("emg_rms", "mean"),
            motion_rms_mean=("motion_rms", "mean"),
        )
    )
    write_csv(participant_means, output_root / "participant_context_means.csv")
    if config["analysis"]["bandpower"].get("save_trial_residuals", False):
        write_csv(residualized_data, output_root / "trial_level_residualized_bandpower.csv")

    rows = []
    for (window, band_name, region), group in participant_means.groupby(["window", "band", "region"]):
        wide = group.pivot(index="subject", columns="context", values="residualized_log_power_mean")
        if not {"human_opponent", "ball_machine"}.issubset(wide.columns):
            continue
        wide = wide.dropna(subset=["human_opponent", "ball_machine"])
        row = {"window": window, "band": band_name, "region": region}
        row.update(paired_statistics(wide["ball_machine"].to_numpy(), wide["human_opponent"].to_numpy()))
        rows.append(row)
    statistics = add_fdr_columns(
        pd.DataFrame(rows), config["analysis"]["statistics"]["fdr_method"]
    ).sort_values(["paired_p_fdr", "window", "band", "region"])
    write_csv(statistics, output_root / "residualized_bandpower_statistics.csv")
    return participant_means, statistics
