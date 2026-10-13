from __future__ import annotations

import numpy as np
import pandas as pd

from .audit import subject_eeg_paths, subject_ids
from .utils import write_csv


def context_from_trial_type(trial_type: str, config: dict) -> str:
    contexts = config["analysis"]["contexts"]
    if trial_type in contexts["human_opponent_trial_types"]:
        return "human_opponent"
    if trial_type in contexts["ball_machine_trial_types"]:
        return "ball_machine"
    return "unknown"


def build_trial_table(paths, config: dict):
    contexts = config["analysis"]["contexts"]
    hit_value = contexts["event_values"]["subject_hit"]
    receive_value = contexts["event_values"]["subject_receive"]
    rows = []

    for subject in subject_ids(paths.raw_root):
        event_path = subject_eeg_paths(paths.raw_root, subject)["events"]
        events = pd.read_csv(event_path, sep="\t")
        required = {"trial_type", "value", "sample", "onset"}
        missing = required - set(events.columns)
        if missing:
            raise RuntimeError(f"{subject} event file is missing columns: {sorted(missing)}")
        events["trial_type"] = events["trial_type"].astype(str).str.strip()
        events["value"] = events["value"].astype(str).str.strip()
        selected = events[events["value"].isin([hit_value, receive_value])].copy()
        selected = selected[~selected["trial_type"].isin(["STATUS", "nan"])].copy()
        selected["context"] = selected["trial_type"].map(
            lambda value: context_from_trial_type(value, config)
        )
        selected["subject"] = subject
        selected["event_index_original"] = selected.index
        selected["event_role"] = selected["value"].map(
            {hit_value: "subject_hit", receive_value: "subject_receive"}
        )
        selected["sample"] = pd.to_numeric(selected["sample"], errors="coerce")
        selected["onset"] = pd.to_numeric(selected["onset"], errors="coerce")
        selected = selected[selected["sample"] >= 0].sort_values("sample").reset_index(drop=True)
        selected["prev_event_role"] = selected["event_role"].shift(1)
        selected["next_event_role"] = selected["event_role"].shift(-1)
        selected["prev_trial_type"] = selected["trial_type"].shift(1)
        selected["next_trial_type"] = selected["trial_type"].shift(-1)
        selected["time_since_prev_event_seconds"] = selected["onset"].diff()
        selected["time_to_next_event_seconds"] = selected["onset"].shift(-1) - selected["onset"]
        selected["trial_id"] = np.arange(len(selected), dtype=int)
        rows.append(selected)

    table = pd.concat(rows, ignore_index=True)
    output_root = paths.output_root / "events"
    write_csv(table, output_root / "all_subjects_trial_events.csv")
    summary = (
        table.groupby(["subject", "context", "trial_type", "event_role"], dropna=False)
        .size()
        .reset_index(name="n_events")
    )
    write_csv(summary, output_root / "trial_event_summary_by_subject.csv")
    return table
