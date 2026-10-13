from __future__ import annotations

from pathlib import Path

import pandas as pd

from .channels import classify_channel_names
from .utils import sha256_file, write_csv


def subject_ids(raw_root: Path) -> list[str]:
    return sorted(path.name for path in raw_root.glob("sub-*") if path.is_dir())


def subject_eeg_paths(raw_root: Path, subject: str) -> dict[str, Path]:
    eeg_dir = raw_root / subject / "eeg"
    prefix = f"{subject}_task-TableTennis"
    return {
        "eeg_dir": eeg_dir,
        "set": eeg_dir / f"{prefix}_eeg.set",
        "fdt": eeg_dir / f"{prefix}_eeg.fdt",
        "eeg_json": eeg_dir / f"{prefix}_eeg.json",
        "channels": eeg_dir / f"{prefix}_channels.tsv",
        "events": eeg_dir / f"{prefix}_events.tsv",
        "events_json": eeg_dir / f"{prefix}_events.json",
    }


def audit_dataset(paths, config: dict):
    import mne

    subjects = subject_ids(paths.raw_root)
    if not subjects:
        raise FileNotFoundError(f"No subject directories found below {paths.raw_root.name}")

    output_root = paths.output_root / "audit"
    manifest_rows = []
    channel_rows = []
    event_rows = []

    for path in sorted(paths.raw_root.rglob("*")):
        if path.is_file():
            row = {
                "relative_path": paths.relative(path),
                "size_bytes": path.stat().st_size,
                "sha256": "",
            }
            if path.name in {"participants.tsv", "dataset_description.json"} or path.suffix in {
                ".tsv",
                ".json",
            }:
                row["sha256"] = sha256_file(path)
            manifest_rows.append(row)

    for subject in subjects:
        files = subject_eeg_paths(paths.raw_root, subject)
        for required in ["set", "fdt", "events"]:
            if not files[required].exists():
                raise FileNotFoundError(f"Missing {required} file for {subject}")
        raw = mne.io.read_raw_eeglab(
            files["set"], preload=False, montage_units="auto", verbose="ERROR"
        )
        groups = classify_channel_names(raw.ch_names, config)
        channel_rows.append(
            {
                "subject": subject,
                "sampling_frequency_hz": float(raw.info["sfreq"]),
                "n_channels": int(raw.info["nchan"]),
                "n_scalp": len(groups["scalp"]),
                "n_noise": len(groups["noise"]),
                "n_emg": len(groups["emg"]),
                "n_motion_misc": len(groups["motion"]),
            }
        )
        events = pd.read_csv(files["events"], sep="\t")
        event_rows.append(
            {
                "subject": subject,
                "n_events": len(events),
                "columns": ",".join(events.columns),
                "sample_min": pd.to_numeric(events.get("sample"), errors="coerce").min(),
                "sample_max": pd.to_numeric(events.get("sample"), errors="coerce").max(),
                "onset_min": pd.to_numeric(events.get("onset"), errors="coerce").min(),
                "onset_max": pd.to_numeric(events.get("onset"), errors="coerce").max(),
            }
        )

    participants_path = paths.raw_root / "participants.tsv"
    participants = pd.read_csv(participants_path, sep="\t")
    write_csv(pd.DataFrame(manifest_rows), output_root / "dataset_manifest.csv")
    write_csv(participants, output_root / "participant_summary.csv")
    write_csv(pd.DataFrame(channel_rows), output_root / "channel_summary.csv")
    write_csv(pd.DataFrame(event_rows), output_root / "event_file_summary.csv")
    return {
        "subjects": subjects,
        "manifest": pd.DataFrame(manifest_rows),
        "participants": participants,
        "channels": pd.DataFrame(channel_rows),
        "events": pd.DataFrame(event_rows),
    }
