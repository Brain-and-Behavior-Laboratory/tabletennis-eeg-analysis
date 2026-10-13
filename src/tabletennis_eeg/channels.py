from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd


def classify_channel_names(ch_names: Iterable[str], config: dict) -> dict[str, list[str]]:
    channel_config = config["channels"]
    noise_prefix = channel_config["noise_prefix"]
    emg_exact = set(channel_config["emg_exact_labels"])
    emg_substrings = [item.lower() for item in channel_config["emg_substrings"]]
    motion_substrings = [item.lower() for item in channel_config["motion_substrings"]]

    groups = {"scalp": [], "noise": [], "motion": [], "emg": []}
    for name in ch_names:
        lowered = name.lower()
        if name.startswith(noise_prefix):
            groups["noise"].append(name)
        elif name in emg_exact or any(token in lowered for token in emg_substrings):
            groups["emg"].append(name)
        elif any(token in lowered for token in motion_substrings):
            groups["motion"].append(name)
        else:
            groups["scalp"].append(name)
    return groups


def apply_channel_types(raw, config: dict):
    groups = classify_channel_names(raw.ch_names, config)
    mapping = {}
    mapping.update({name: "eeg" for name in groups["scalp"]})
    mapping.update({name: "misc" for name in groups["noise"]})
    mapping.update({name: "misc" for name in groups["motion"]})
    mapping.update({name: "emg" for name in groups["emg"]})
    raw.set_channel_types(mapping, verbose="WARNING")
    return raw, groups


def sensor_region(channel_name: str, config: dict) -> str:
    cleaned = channel_name.removeprefix(config["channels"]["noise_prefix"])
    for region, pattern in config["sensor_regions"]["patterns"].items():
        if re.match(pattern, cleaned):
            return region
    return "other"


def channel_group_frame(subject: str, groups: dict[str, list[str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject": subject,
                "channel_group": group,
                "n_channels": len(names),
                "channels": ",".join(names),
            }
            for group, names in groups.items()
        ]
    )
