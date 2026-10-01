from __future__ import annotations

import re
from pathlib import Path

import actium


def test_runtime_and_project_versions_match() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    assert match is not None
    assert actium.__version__ == match.group(1)
