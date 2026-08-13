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

    unresolved_markdown_tokens = ("readable_table(", "DIRECTION_TABLE", "full_width_table(")
    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "markdown":
            continue
        token = next((item for item in unresolved_markdown_tokens if item in cell.source), None)
        if token:
            failures.append(
                f"unresolved generator token {token!r} in Markdown cell {cell_index}"
            )

    # Eight leading spaces turn ordinary Markdown into a literal code block.
    # The public notebooks use fenced blocks if code is ever needed in prose,
    # so this reliably catches accidental generator indentation.
    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type != "markdown":
            continue
        for line_number, line in enumerate(cell.source.splitlines(), start=1):
            if line.startswith("        "):
                failures.append(
                    "unintended Markdown code-block indentation in "
                    f"cell {cell_index}, line {line_number}"
                )
                break

    if execute:
        try:
            client = NotebookClient(
                notebook,
                timeout=120,
                kernel_name="python3",
                resources={"metadata": {"path": str(path.parent)}},
            )
            client.execute()
            # Preserve only outputs produced by a successful clean-kernel run.
            # GitHub readers can therefore inspect the complete verified lab.
            nbformat.write(notebook, path)
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
