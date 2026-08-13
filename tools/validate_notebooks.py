"""Validate structure and clean-kernel execution of Learning Labs notebooks."""

from __future__ import annotations

import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]


def notebook_links(notebook: nbformat.NotebookNode) -> tuple[set[str], set[str]]:
    """Collect local contents links and explicitly declared HTML anchors."""
    markdown = "\n".join(
        cell.source for cell in notebook.cells if cell.cell_type == "markdown"
    )
    links = set(re.findall(r"\]\(#([a-zA-Z0-9_-]+)\)", markdown))
    anchors = set(re.findall(r'<a\s+id="([a-zA-Z0-9_-]+)"\s*></a>', markdown))
    return links, anchors


def validate(path: Path, execute: bool) -> list[str]:
    """Validate one notebook and return human-readable failures."""
    notebook = nbformat.read(path, as_version=4)
    failures: list[str] = []

    links, anchors = notebook_links(notebook)
    missing = sorted(links - anchors)
    if missing:
        failures.append(f"missing contents anchors: {', '.join(missing)}")

    if not any("Contents" in cell.source for cell in notebook.cells if cell.cell_type == "markdown"):
        failures.append("missing table of contents")

    if execute:
        try:
            client = NotebookClient(
                notebook,
                timeout=120,
                kernel_name="python3",
                resources={"metadata": {"path": str(path.parent)}},
            )
            client.execute()
        except Exception as exc:  # Execution exceptions require their full type.
            failures.append(f"clean-kernel execution failed: {type(exc).__name__}: {exc}")

    return failures


def main() -> int:
    """Validate the reusable template structurally and execute published labs."""
    targets = [
        (ROOT / "templates" / "CH3RRY PI3 Learning Lab Template.ipynb", False),
        (
            ROOT
            / "notebooks"
            / "01-foundations"
            / "gradient-descent"
            / "01-gradient-descent-from-intuition-to-implementation.ipynb",
            True,
        ),
    ]

    failed = False
    for path, execute in targets:
        failures = validate(path, execute)
        if failures:
            failed = True
            print(f"FAIL: {path.relative_to(ROOT)}")
            for failure in failures:
                print(f"  - {failure}")
        else:
            suffix = "structure and execution" if execute else "structure"
            print(f"PASS: {path.relative_to(ROOT)} ({suffix})")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

