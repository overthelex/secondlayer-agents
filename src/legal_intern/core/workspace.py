"""Workspace management -- git-versioned workspace for each consultation."""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path


class WorkspaceManager:
    """Manages a git-versioned workspace directory for a consultation run."""

    def __init__(self, config) -> None:
        self.config = config
        self.root: Path = Path(".")
        self.logs_dir: Path = Path(".")

    def init(self, problem_title: str) -> Path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = problem_title[:40].replace(" ", "_").replace("/", "_")
        self.root = Path(self.config.workspace_dir) / f"{ts}_{slug}"
        self.root.mkdir(parents=True, exist_ok=True)

        self.logs_dir = self.root / "logs"
        self.logs_dir.mkdir(exist_ok=True)

        (self.root / "evidence").mkdir(exist_ok=True)
        (self.root / "derivations").mkdir(exist_ok=True)

        subprocess.run(["git", "init"], cwd=self.root, capture_output=True)
        return self.root

    def snapshot(self, message: str) -> None:
        """Create a git commit snapshot of the current workspace state."""
        subprocess.run(["git", "add", "-A"], cwd=self.root, capture_output=True)
        subprocess.run(
            ["git", "commit", "-m", message, "--allow-empty"],
            cwd=self.root,
            capture_output=True,
        )
