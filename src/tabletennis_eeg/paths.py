from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def find_repository_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "config").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not locate the repository root containing pyproject.toml and config/."
    )


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved paths for the code repository and a separate analysis workspace."""

    root: Path
    workspace_root: Path
    data_root: Path
    raw_root: Path
    derivative_root: Path
    output_root: Path

    @classmethod
    def create(cls, root: Path, accession: str) -> "ProjectPaths":
        root = root.resolve()

        # Keep downloaded data, intermediate files, and CSV outputs outside the
        # code repository by default. Users can place the workspace anywhere by
        # setting TABLETENNIS_EEG_WORKSPACE.
        default_workspace = root.parent / "tabletennis-eeg-workspace"
        workspace_root = Path(
            os.environ.get("TABLETENNIS_EEG_WORKSPACE", str(default_workspace))
        ).expanduser().resolve()

        # Optional advanced overrides retain flexibility for large storage volumes.
        data_root = Path(
            os.environ.get("TABLETENNIS_EEG_DATA_ROOT", str(workspace_root / "data"))
        ).expanduser().resolve()
        output_root = Path(
            os.environ.get(
                "TABLETENNIS_EEG_OUTPUT_ROOT",
                str(workspace_root / "outputs" / "csv"),
            )
        ).expanduser().resolve()

        return cls(
            root=root,
            workspace_root=workspace_root,
            data_root=data_root,
            raw_root=data_root / "raw" / accession,
            derivative_root=data_root / "derivatives" / "tabletennis-eeg",
            output_root=output_root,
        )

    def ensure_directories(self) -> None:
        for path in [
            self.workspace_root,
            self.data_root,
            self.raw_root,
            self.derivative_root,
            self.output_root,
        ]:
            path.mkdir(parents=True, exist_ok=True)

    def relative(self, path: Path) -> str:
        resolved = path.resolve()
        for base in (self.workspace_root, self.root):
            try:
                return resolved.relative_to(base).as_posix()
            except ValueError:
                continue
        return path.name
