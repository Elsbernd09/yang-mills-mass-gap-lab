"""
Reproducibility and experiment-manifest infrastructure.

The module records immutable run specifications and execution metadata for
finite-lattice experiments.

A run manifest can include:

1. experiment name,
2. validated configuration,
3. deterministic configuration hash,
4. UTC creation timestamp,
5. Python version,
6. package versions,
7. operating-system and machine information,
8. Git commit identifier when available,
9. Git working-tree cleanliness when available,
10. runtime duration,
11. generated output files.

The manifest is intended to make numerical runs inspectable and repeatable.

A manifest does not prove scientific correctness. It records the computational
conditions under which an experiment was executed.
"""

from __future__ import annotations

from dataclasses import (
    asdict,
    dataclass,
    field,
)
from datetime import (
    datetime,
    timezone,
)
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib
import numpy as np
import scipy


@dataclass(frozen=True)
class RuntimeEnvironment:
    """Version and platform information for one experiment run."""

    python_version: str
    numpy_version: str
    scipy_version: str
    matplotlib_version: str
    platform_system: str
    platform_release: str
    platform_machine: str


@dataclass(frozen=True)
class GitMetadata:
    """Best-effort Git repository metadata."""

    commit: str | None
    dirty: bool | None
    available: bool


@dataclass
class RunManifest:
    """Serializable experiment-run manifest."""

    experiment_name: str
    configuration: dict[str, Any]
    configuration_hash: str
    created_at_utc: str
    environment: RuntimeEnvironment
    git: GitMetadata
    runtime_seconds: float | None = None
    output_files: list[str] = field(
        default_factory=list
    )
    status: str = "created"
    notes: list[str] = field(
        default_factory=list
    )

    def to_dict(
        self,
    ) -> dict[str, Any]:
        """Convert the manifest into JSON-serializable data."""
        return asdict(
            self
        )


def canonical_json(
    value: Any,
) -> str:
    """
    Return deterministic compact JSON.

    Dictionary keys are sorted and non-ASCII characters are retained.
    """
    return json.dumps(
        value,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
        ensure_ascii=False,
    )


def configuration_hash(
    configuration: dict[str, Any],
    length: int = 16,
) -> str:
    """
    Return a deterministic SHA-256 prefix for a configuration dictionary.
    """
    if length < 8:
        raise ValueError(
            "Configuration hash length must be at least eight."
        )

    payload = canonical_json(
        configuration
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        payload
    ).hexdigest()

    return digest[
        :length
    ]


def current_runtime_environment() -> RuntimeEnvironment:
    """Return package and platform versions for the active Python process."""
    return RuntimeEnvironment(
        python_version=sys.version.split()[0],
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
        matplotlib_version=matplotlib.__version__,
        platform_system=platform.system(),
        platform_release=platform.release(),
        platform_machine=platform.machine(),
    )


def _run_git_command(
    arguments: list[str],
    repository_root: Path,
) -> subprocess.CompletedProcess[str] | None:
    """
    Run a Git command without raising when Git metadata is unavailable.
    """
    try:
        return subprocess.run(
            [
                "git",
                *arguments,
            ],
            cwd=repository_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except (
        OSError,
        subprocess.SubprocessError,
    ):
        return None


def git_metadata(
    repository_root: str | Path = ".",
) -> GitMetadata:
    """Return best-effort Git commit and dirty-tree metadata."""
    root = Path(
        repository_root
    ).resolve()

    commit_result = _run_git_command(
        [
            "rev-parse",
            "HEAD",
        ],
        repository_root=root,
    )

    if (
        commit_result is None
        or commit_result.returncode != 0
    ):
        return GitMetadata(
            commit=None,
            dirty=None,
            available=False,
        )

    status_result = _run_git_command(
        [
            "status",
            "--porcelain",
        ],
        repository_root=root,
    )

    dirty = (
        None
        if (
            status_result is None
            or status_result.returncode != 0
        )
        else bool(
            status_result.stdout.strip()
        )
    )

    return GitMetadata(
        commit=commit_result.stdout.strip(),
        dirty=dirty,
        available=True,
    )


def create_run_manifest(
    experiment_name: str,
    configuration: dict[str, Any],
    repository_root: str | Path = ".",
) -> RunManifest:
    """Create a new run manifest before experiment execution."""
    experiment_name = str(
        experiment_name
    ).strip()

    if not experiment_name:
        raise ValueError(
            "experiment_name cannot be empty."
        )

    if not isinstance(
        configuration,
        dict,
    ):
        raise TypeError(
            "configuration must be a dictionary."
        )

    normalized_configuration = json.loads(
        canonical_json(
            configuration
        )
    )

    created_at = datetime.now(
        timezone.utc
    ).isoformat()

    return RunManifest(
        experiment_name=experiment_name,
        configuration=normalized_configuration,
        configuration_hash=configuration_hash(
            normalized_configuration
        ),
        created_at_utc=created_at,
        environment=current_runtime_environment(),
        git=git_metadata(
            repository_root=repository_root
        ),
    )


def default_run_directory(
    manifest: RunManifest,
    root: str | Path = "results/runs",
) -> Path:
    """
    Construct a deterministic-style run directory name.

    The timestamp prevents collisions between repeated executions of the same
    configuration. The configuration hash preserves configuration identity.
    """
    timestamp = (
        manifest.created_at_utc
        .replace(
            ":",
            "",
        )
        .replace(
            "-",
            "",
        )
        .replace(
            "+0000",
            "Z",
        )
        .replace(
            ".",
            "",
        )
    )

    safe_name = "".join(
        character
        if (
            character.isalnum()
            or character in (
                "-",
                "_",
            )
        )
        else "-"
        for character in manifest.experiment_name
    )

    return Path(
        root
    ) / (
        f"{safe_name}-"
        f"{manifest.configuration_hash}-"
        f"{timestamp}"
    )


def write_json(
    path: str | Path,
    value: Any,
) -> Path:
    """Write pretty deterministic JSON and return the output path."""
    output_path = Path(
        path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        + "\n"
    )

    return output_path


def write_manifest(
    manifest: RunManifest,
    path: str | Path,
) -> Path:
    """Serialize one manifest to JSON."""
    return write_json(
        path=path,
        value=manifest.to_dict(),
    )


def register_output_file(
    manifest: RunManifest,
    path: str | Path,
    run_directory: str | Path | None = None,
) -> None:
    """
    Register one generated output file.

    When run_directory is supplied, paths under that directory are recorded
    relative to the run directory.
    """
    output_path = Path(
        path
    )

    if run_directory is None:
        recorded = str(
            output_path
        )
    else:
        root = Path(
            run_directory
        ).resolve()

        try:
            recorded = str(
                output_path.resolve().relative_to(
                    root
                )
            )
        except ValueError:
            recorded = str(
                output_path
            )

    if recorded not in manifest.output_files:
        manifest.output_files.append(
            recorded
        )


def finalize_manifest(
    manifest: RunManifest,
    runtime_seconds: float,
    status: str = "completed",
    notes: list[str] | None = None,
) -> None:
    """Finalize runtime and status metadata after execution."""
    runtime_seconds = float(
        runtime_seconds
    )

    if (
        not np.isfinite(
            runtime_seconds
        )
        or runtime_seconds < 0.0
    ):
        raise ValueError(
            "runtime_seconds must be finite and nonnegative."
        )

    status = str(
        status
    ).strip()

    if not status:
        raise ValueError(
            "status cannot be empty."
        )

    manifest.runtime_seconds = runtime_seconds
    manifest.status = status

    if notes is not None:
        manifest.notes.extend(
            str(
                note
            )
            for note in notes
        )


def load_manifest(
    path: str | Path,
) -> dict[str, Any]:
    """Load a serialized run manifest as a dictionary."""
    return json.loads(
        Path(
            path
        ).read_text()
    )
