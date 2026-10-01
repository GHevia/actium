from __future__ import annotations

import ast
from pathlib import Path


def test_installable_package_has_no_octavian_imports() -> None:
    package_root = Path(__file__).parents[1] / "src" / "actium"
    for path in package_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert all(not name.startswith("octavian") for name in names), path


def test_v01_surface_does_not_contain_out_of_scope_modules() -> None:
    package_root = Path(__file__).parents[1] / "src" / "actium"
    module_names = {path.stem for path in package_root.glob("*.py")}
    forbidden = {
        "drag",
        "events",
        "forces",
        "maneuvers",
        "measurements",
        "od",
        "srp",
        "third_bodies",
    }
    assert module_names.isdisjoint(forbidden)
