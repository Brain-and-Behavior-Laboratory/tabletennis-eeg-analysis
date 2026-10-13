from __future__ import annotations

import argparse
from pathlib import Path

from .audit import audit_dataset, subject_ids
from .bandpower import extract_all_features, residualize_and_test
from .config import available_model_ids, load_configuration
from .download import download_dataset
from .epoching import create_all_epochs
from .events import build_trial_table
from .paths import ProjectPaths, find_repository_root
from .preprocess import preprocess_all
from .robustness import compute_robustness
from .spectral import fit_model_all
from .statistics import compute_all_model_statistics


def _context(root_argument: str | None):
    root = Path(root_argument).resolve() if root_argument else find_repository_root()
    config = load_configuration(root)
    config["analysis"]["sensor_regions_order"] = config["sensor_regions"]["ordered_regions"]
    paths = ProjectPaths.create(root, config["dataset"]["accession"])
    paths.ensure_directories()
    return paths, config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tabletennis-eeg")
    parser.add_argument("--root", help="Repository root; normally detected automatically.")
    subcommands = parser.add_subparsers(dest="command", required=True)

    download = subcommands.add_parser("download", help="Download configured OpenNeuro files.")
    download.add_argument("--overwrite", action="store_true")
    download.add_argument("--metadata-only", action="store_true")

    subcommands.add_parser("audit", help="Inspect downloaded participants, channels, and events.")

    preprocess = subcommands.add_parser("preprocess", help="Preprocess continuous recordings.")
    preprocess.add_argument("--subjects", nargs="*")
    preprocess.add_argument("--overwrite", action="store_true")
    preprocess.add_argument("--n-jobs", type=int, default=1)

    subcommands.add_parser("events", help="Build the cleaned event table.")

    epochs = subcommands.add_parser("epochs", help="Create participant-hit epochs.")
    epochs.add_argument("--subjects", nargs="*")
    epochs.add_argument("--overwrite", action="store_true")

    bandpower = subcommands.add_parser("bandpower", help="Extract bandpower and nuisance features.")
    bandpower.add_argument("--subjects", nargs="*")
    bandpower.add_argument("--overwrite", action="store_true")

    subcommands.add_parser("bandpower-stats", help="Residualize and compare bandpower.")

    spectral = subcommands.add_parser("spectral", help="Fit configured spectral models.")
    spectral.add_argument("--models", nargs="*")
    spectral.add_argument("--subjects", nargs="*")
    spectral.add_argument("--overwrite", action="store_true")

    spectral_stats = subcommands.add_parser(
        "spectral-stats", help="Compute participant-level spectral statistics."
    )
    spectral_stats.add_argument("--models", nargs="*")

    subcommands.add_parser("robustness", help="Compute configured sensitivity analyses.")
    subcommands.add_parser("run-all", help="Run the complete sequence.")
    return parser


def main(argv: list[str] | None = None):
    args = build_parser().parse_args(argv)
    paths, config = _context(args.root)

    if args.command == "download":
        download_dataset(paths, config, overwrite=args.overwrite, metadata_only=args.metadata_only)
    elif args.command == "audit":
        audit_dataset(paths, config)
    elif args.command == "preprocess":
        preprocess_all(
            paths,
            config,
            subjects=args.subjects,
            overwrite=args.overwrite,
            n_jobs=args.n_jobs,
        )
    elif args.command == "events":
        build_trial_table(paths, config)
    elif args.command == "epochs":
        create_all_epochs(paths, config, subjects=args.subjects, overwrite=args.overwrite)
    elif args.command == "bandpower":
        extract_all_features(paths, config, subjects=args.subjects, overwrite=args.overwrite)
    elif args.command == "bandpower-stats":
        residualize_and_test(paths, config)
    elif args.command == "spectral":
        models = args.models or available_model_ids(paths.root)
        for model_id in models:
            fit_model_all(
                paths,
                config,
                model_id,
                subjects=args.subjects,
                overwrite=args.overwrite,
            )
    elif args.command == "spectral-stats":
        compute_all_model_statistics(paths, config, args.models)
    elif args.command == "robustness":
        compute_robustness(paths, config)
    elif args.command == "run-all":
        download_dataset(paths, config)
        audit_dataset(paths, config)
        subjects = subject_ids(paths.raw_root)
        preprocess_all(paths, config, subjects=subjects)
        build_trial_table(paths, config)
        create_all_epochs(paths, config, subjects=subjects)
        extract_all_features(paths, config, subjects=subjects)
        residualize_and_test(paths, config)
        models = available_model_ids(paths.root)
        for model_id in models:
            fit_model_all(paths, config, model_id, subjects=subjects)
        compute_all_model_statistics(paths, config, models)
        compute_robustness(paths, config)
