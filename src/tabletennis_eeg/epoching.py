from __future__ import annotations

import numpy as np
import pandas as pd

from .audit import subject_ids
from .utils import ensure_dir, write_csv


def create_subject_epochs(paths, config: dict, subject: str, overwrite: bool = False):
    import mne

    epoch_settings = config["analysis"]["saved_epoch"]
    raw_file = (
        paths.derivative_root
        / "preprocessed"
        / subject
        / f"{subject}_task-TableTennis_desc-preprocessed_raw.fif"
    )
    event_file = paths.output_root / "events" / "all_subjects_trial_events.csv"
    if not raw_file.exists():
        raise FileNotFoundError(raw_file)
    if not event_file.exists():
        raise FileNotFoundError(event_file)

    output_dir = ensure_dir(paths.derivative_root / "epochs" / subject)
    output_epochs = output_dir / f"{subject}_task-TableTennis_desc-subjecthit_epo.fif"
    output_metadata = output_dir / f"{subject}_subject_hit_epoch_metadata.csv"
    output_summary = output_dir / f"{subject}_subject_hit_epoch_summary.csv"
    if output_epochs.exists() and not overwrite:
        return pd.read_csv(output_summary).iloc[0].to_dict()

    raw = mne.io.read_raw_fif(raw_file, preload=False, verbose="WARNING")
    trials = pd.read_csv(event_file)
    selected = trials[
        (trials["subject"] == subject)
        & (trials["event_role"] == "subject_hit")
        & (trials["context"].isin(["human_opponent", "ball_machine"]))
    ].sort_values("sample")
    samples = np.round(selected["sample"].to_numpy()).astype(int)
    _, unique_indices = np.unique(samples, return_index=True)
    unique_indices = np.sort(unique_indices)
    selected = selected.iloc[unique_indices].reset_index(drop=True)
    samples = samples[unique_indices]
    codes = np.where(selected["context"].to_numpy() == "human_opponent", 1, 2)
    events = np.column_stack([samples, np.zeros(len(samples), dtype=int), codes])
    metadata_columns = [
        "subject",
        "trial_id",
        "onset",
        "sample",
        "context",
        "trial_type",
        "event_role",
        "prev_event_role",
        "next_event_role",
        "time_since_prev_event_seconds",
        "time_to_next_event_seconds",
    ]
    metadata = selected[metadata_columns].copy()
    epochs = mne.Epochs(
        raw,
        events=events,
        event_id={"human_opponent/subject_hit": 1, "ball_machine/subject_hit": 2},
        tmin=epoch_settings["tmin_seconds"],
        tmax=epoch_settings["tmax_seconds"],
        baseline=None,
        preload=True,
        metadata=metadata,
        reject_by_annotation=True,
        verbose="WARNING",
    )
    epochs.save(output_epochs, overwrite=True)
    epochs.metadata.to_csv(output_metadata, index=False)
    summary = {
        "subject": subject,
        "n_candidate_subject_hit_events": int(len(selected)),
        "n_epochs_retained": int(len(epochs)),
        "retention_fraction": float(len(epochs) / len(selected)),
        "n_human_opponent_epochs": int((epochs.metadata["context"] == "human_opponent").sum()),
        "n_ball_machine_epochs": int((epochs.metadata["context"] == "ball_machine").sum()),
        "tmin_seconds": epoch_settings["tmin_seconds"],
        "tmax_seconds": epoch_settings["tmax_seconds"],
        "output_relative_path": paths.relative(output_epochs),
    }
    write_csv(pd.DataFrame([summary]), output_summary)
    return summary


def create_all_epochs(paths, config: dict, subjects: list[str] | None = None, **kwargs):
    subjects = subjects or subject_ids(paths.raw_root)
    rows = [create_subject_epochs(paths, config, subject, **kwargs) for subject in subjects]
    frame = pd.DataFrame(rows)
    write_csv(frame, paths.output_root / "epoching" / "participant_epoch_summary.csv")
    return frame
