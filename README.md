# Table-tennis EEG analysis

This repository provides a Python implementation of the EEG analysis workflow described in the following research work:

Arunim Guchait, Chiao-Yun Chen, and Neil G. Muggleton. **Resolution-aware spectral parameterization reveals robust context-dependent aperiodic EEG dynamics during real-world table tennis**. *bioRxiv*, 2026.10.01.755924, 2026. https://doi.org/10.64898/2026.10.01.755924.

The implementation analyses the public Real World Table Tennis EEG dataset from OpenNeuro (`ds004505`). It provides a configurable workflow for EEG preprocessing, event construction, participant-hit epoching, spectral parameterization, nuisance-informed bandpower analysis, participant-level statistical comparisons, and robustness evaluation. The analysis examines differences in periodic and aperiodic EEG activity between human-opponent and ball-machine table-tennis contexts.

## This repository structure

```text
<tabletennis-eeg-analysis>/
├── config/                    # Dataset and analysis settings
│   ├── models/                # Spectral-model configurations
│   ├── analysis.yaml
│   ├── channels.yaml
│   ├── dataset.yaml
│   └── sensor_regions.yaml
├── scripts/                   # Ordered analysis entry points
├── src/
│   └── tabletennis_eeg/       # Reusable analysis functions
├── environment.yml
├── pyproject.toml
├── LICENSE
└── README.md
```

## Recommended local folder layout

Keep the cloned repository and the analysis workspace as separate sibling directories. A recommended layout is:

```text
<analysis-parent>/
├── tabletennis-eeg-analysis/          # Cloned code repository
└── tabletennis-eeg-workspace/         # Created locally when the analysis runs
    ├── data/
    │   ├── raw/
    │   │   └── ds004505/              # Downloaded OpenNeuro dataset
    │   └── derivatives/
    │       └── tabletennis-eeg/
    │           ├── preprocessed/       # Preprocessed continuous recordings
    │           └── epochs/             # Participant-hit epoch files
    └── outputs/
        └── csv/
            ├── audit/
            ├── preprocessing/
            ├── events/
            ├── epoching/
            ├── bandpower/
            ├── specparam/
            └── robustness/
```

The scripts create the required workspace directories automatically. The workspace does not need to exist before the first command is run.

By default, the workspace is created beside the cloned repository and is named `tabletennis-eeg-workspace`. A different location can be selected with the `TABLETENNIS_EEG_WORKSPACE` environment variable.

## Installation

From the cloned repository directory, create and activate the Conda environment:

```bash
conda env create -f environment.yml
conda activate tabletennis-eeg
python -m pip install -e .
```

The editable installation makes the package available while allowing the source code to be modified directly.

## Computing environment used for the analysis

The analysis workflow was run on a workstation with the following configuration:

```text
Operating system: Ubuntu 24.04.3 LTS
Processor:        12th Gen Intel® Core™ i9-12900 × 24
Memory:           128 GiB
```

These specifications describe the computing environment used for the analysis; they are not strict hardware requirements. The workflow may be run on other systems that provide sufficient memory, storage, and compatible software dependencies. Runtime and resource requirements may vary with the number of participants processed in parallel and the analysis settings.

## Select the analysis workspace

### Linux or macOS

```bash
export TABLETENNIS_EEG_WORKSPACE=/path/to/tabletennis-eeg-workspace
```

### Windows PowerShell

```powershell
$env:TABLETENNIS_EEG_WORKSPACE = "D:\path\to\tabletennis-eeg-workspace"
```

These are generic examples. Replace the placeholder with any suitable local directory. When the variable is not set, the code uses a sibling directory named `tabletennis-eeg-workspace`.

For advanced storage arrangements, the data and CSV locations may be overridden separately:

```bash
export TABLETENNIS_EEG_DATA_ROOT=/path/to/eeg-data
export TABLETENNIS_EEG_OUTPUT_ROOT=/path/to/csv-output
```

The single workspace variable is sufficient for most users.

## Recommended analysis approach

The analysis should be approached in stages rather than running the complete dataset immediately. First, review the configuration files in `config/`. These files define the dataset version, sampling rate, filtering, event windows, context mapping, frequency bands, channel classification, sensor regions, spectral settings, statistical procedures, and sensitivity models. Scientific changes should normally be made in these YAML files before modifying the reusable Python functions.

Next, download the dataset and audit its structure:

```bash
python scripts/00_download_data.py
python scripts/01_audit_dataset.py
```

Then test the participant-level stages on one participant:

```bash
python scripts/02_preprocess.py --subjects sub-01
python scripts/03_build_events.py
python scripts/04_create_epochs.py --subjects sub-01
python scripts/05_extract_bandpower.py --subjects sub-01
python scripts/07_fit_spectral_models.py \
    --subjects sub-01 \
    --models resolution_aware_max6
```

Before scaling up, inspect the generated CSV files and verify that the following items are appropriate for the intended analysis:

- channel classification;
- event-to-context mapping;
- sampling frequency;
- retained epoch counts;
- short-window sample counts;
- spectral frequency spacing;
- sensor-region assignments;
- spectral-model settings.

After the single-participant check, run the complete sequence:

```bash
python scripts/00_download_data.py
python scripts/01_audit_dataset.py
python scripts/02_preprocess.py
python scripts/03_build_events.py
python scripts/04_create_epochs.py
python scripts/05_extract_bandpower.py
python scripts/06_compute_bandpower_statistics.py
python scripts/07_fit_spectral_models.py
python scripts/08_compute_spectral_statistics.py
python scripts/09_compute_robustness.py
```

The same sequence can be started with:

```bash
python scripts/10_run_all.py
```

Running the stages separately is recommended when modifying the analysis because it makes it easier to inspect each intermediate step and rerun only the affected computations.

## What each script does

| Script | Purpose |
|---|---|
| `00_download_data.py` | Downloads the configured public OpenNeuro files. |
| `01_audit_dataset.py` | Summarizes participants, channels, events, sampling rates, and available files. |
| `02_preprocess.py` | Classifies channels, filters the configured signals, applies scalp referencing, and saves preprocessed recordings. |
| `03_build_events.py` | Cleans event labels and maps trials to human-opponent and ball-machine contexts. |
| `04_create_epochs.py` | Creates participant-hit epochs and stores epoch metadata. |
| `05_extract_bandpower.py` | Computes trial-level bandpower and nuisance features. |
| `06_compute_bandpower_statistics.py` | Residualizes bandpower and computes participant-level paired comparisons. |
| `07_fit_spectral_models.py` | Estimates spectra and fits the selected spectral-parameterization configurations. |
| `08_compute_spectral_statistics.py` | Computes participant-level statistics for the fitted spectral features. |
| `09_compute_robustness.py` | Runs the configured bootstrap, leave-one-participant-out, concordance, BIC, and quality-sensitivity analyses. |
| `10_run_all.py` | Runs the complete sequence using the current configuration. |

Each script accepts `--help`. Subject-level scripts accept `--subjects`, and stages that write participant-level files accept `--overwrite` when existing files need to be replaced.

## Modifying the analysis

The repository separates scientific settings from reusable computation code.

Use `config/analysis.yaml` to change preprocessing parameters, event windows, frequency bands, PSD settings, statistical settings, and robustness targets. Use `config/channels.yaml` to adjust auxiliary-channel classification rules. Use `config/sensor_regions.yaml` to change scalp-region mappings. Use the files in `config/models/` to change the spectral window, peak-width limits, maximum number of peaks, overlap, or aperiodic-only setting.

A new spectral specification can be added by copying one of the files in `config/models/`, assigning a unique `model_id`, and changing the required parameters. The shared fitting script detects model configuration files automatically.

Modify functions under `src/tabletennis_eeg/` when the required change affects the computational method itself rather than a parameter. The files under `scripts/` are intentionally short entry points and should generally not contain duplicated analysis logic.

## Restarting or rerunning stages

The computational order matters because later stages read files created earlier. If a preprocessing setting changes, rerun preprocessing and every downstream stage. If only a statistical setting changes, the preprocessed recordings, epochs, and extracted features can usually be retained while the relevant statistical scripts are rerun. If a spectral-model YAML file changes, rerun that spectral model and its downstream statistics and sensitivity analyses.

Use `--overwrite` only when existing participant-level files should be replaced. Keeping separate workspace directories for substantially different analysis variants is often safer than repeatedly overwriting one workspace.

