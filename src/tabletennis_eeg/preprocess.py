from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import pandas as pd

from .audit import subject_eeg_paths, subject_ids
from .channels import apply_channel_types, channel_group_frame
from .utils import ensure_dir, write_csv, write_json


def _add_boundary_annotations(raw, padding_seconds: float) -> int:
    import mne

    if raw.annotations is None or len(raw.annotations) == 0:
        return 0
    max_time = raw.times[-1]
    onsets, durations = [], []
    for onset, duration, description in zip(
        raw.annotations.onset, raw.annotations.duration, raw.annotations.description
    ):
        if "boundary" not in str(description).lower():
            continue
        start = max(0.0, float(onset) - padding_seconds)
        stop = min(max_time, float(onset) + max(float(duration), 0.0) + padding_seconds)
        onsets.append(start)
        durations.append(max(0.0, stop - start))
    if not onsets:
        return 0
    additions = mne.Annotations(
        onset=onsets,
        duration=durations,
        description=["BAD_boundary"] * len(onsets),
        orig_time=raw.annotations.orig_time,
    )
    raw.set_annotations(raw.annotations + additions)
    return len(onsets)


def _powerline_frequency(eeg_json: Path) -> float:
    if not eeg_json.exists():
        return 60.0
    with eeg_json.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    value = metadata.get("PowerLineFrequency", 60.0)
    return float(value) if isinstance(value, (int, float)) else 60.0


def _derivative_description(paths, config: dict) -> None:
    target = paths.derivative_root / "dataset_description.json"
    if target.exists():
        return
    write_json(
        {
            "Name": "Table-tennis EEG computational derivatives",
            "BIDSVersion": "1.10.0",
            "DatasetType": "derivative",
            "GeneratedBy": [
                {
                    "Name": "tabletennis-eeg",
                    "Version": "1.0.0",
                    "Description": "Conservative preprocessing and participant-hit epoching pipeline",
                }
            ],
            "SourceDatasets": [
                {
                    "DOI": config["dataset"]["doi"],
                    "Version": config["dataset"]["snapshot_version"],
                }
            ],
        },
        target,
    )


def preprocess_subject(paths, config: dict, subject: str, overwrite: bool = False, n_jobs: int = 1):
    import mne

    settings = config["analysis"]["preprocessing"]
    source = subject_eeg_paths(paths.raw_root, subject)
    output_dir = ensure_dir(paths.derivative_root / "preprocessed" / subject)
    output_raw = output_dir / f"{subject}_task-TableTennis_desc-preprocessed_raw.fif"
    summary_path = output_dir / f"{subject}_preprocessing_summary.csv"
    groups_path = output_dir / f"{subject}_channel_groups.csv"
    if output_raw.exists() and not overwrite:
        return pd.read_csv(summary_path).iloc[0].to_dict()

    raw = mne.io.read_raw_eeglab(
        source["set"], preload=True, montage_units="auto", verbose="WARNING"
    )
    raw, groups = apply_channel_types(raw, config)
    if len(groups["scalp"]) != 120 or len(groups["noise"]) != 120:
        warnings.warn(
            f"{subject}: expected 120 scalp and 120 noise channels; observed "
            f"{len(groups['scalp'])} and {len(groups['noise'])}."
        )

    n_bad = _add_boundary_annotations(raw, settings["boundary_padding_seconds"])
    filter_channels = groups["scalp"] + groups["noise"]
    raw.filter(
        l_freq=settings["l_freq_hz"],
        h_freq=settings["h_freq_hz"],
        picks=filter_channels,
        n_jobs=n_jobs,
        skip_by_annotation=("BAD", "bad", "edge", "bad_acq_skip"),
        verbose="WARNING",
    )
    line_frequency = _powerline_frequency(source["eeg_json"])
    if 0 < line_frequency < settings["h_freq_hz"]:
        raw.notch_filter(
            freqs=[line_frequency],
            picks=filter_channels,
            n_jobs=n_jobs,
            skip_by_annotation=("BAD", "bad", "edge", "bad_acq_skip"),
            verbose="WARNING",
        )
    raw.set_eeg_reference(ref_channels="average", projection=False, verbose="WARNING")
    raw.save(output_raw, overwrite=True)

    annotations = Counter(str(item) for item in raw.annotations.description)
    summary = {
        "subject": subject,
        "sampling_frequency_hz": float(raw.info["sfreq"]),
        "duration_seconds": float(raw.n_times / raw.info["sfreq"]),
        "n_channels": int(raw.info["nchan"]),
        "n_scalp": len(groups["scalp"]),
        "n_noise": len(groups["noise"]),
        "n_emg": len(groups["emg"]),
        "n_motion_misc": len(groups["motion"]),
        "l_freq_hz": settings["l_freq_hz"],
        "h_freq_hz": settings["h_freq_hz"],
        "boundary_padding_seconds": settings["boundary_padding_seconds"],
        "n_bad_boundary_annotations_added": n_bad,
        "n_bad_boundary_annotations_total": annotations.get("BAD_boundary", 0),
        "output_relative_path": paths.relative(output_raw),
    }
    write_csv(pd.DataFrame([summary]), summary_path)
    write_csv(channel_group_frame(subject, groups), groups_path)
    return summary


def preprocess_all(paths, config: dict, subjects: list[str] | None = None, **kwargs):
    _derivative_description(paths, config)
    subjects = subjects or subject_ids(paths.raw_root)
    rows = [preprocess_subject(paths, config, subject, **kwargs) for subject in subjects]
    frame = pd.DataFrame(rows)
    write_csv(frame, paths.output_root / "preprocessing" / "preprocessing_summary.csv")
    return frame
