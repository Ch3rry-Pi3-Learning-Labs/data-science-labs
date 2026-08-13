"""Validate structure and clean-kernel execution of Learning Labs notebooks."""

from __future__ import annotations

import base64
import io
import re
import sys
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
# Measured in GitHub's live notebook renderer at both 894 px and 1264 px:
# rich image output begins at x=95 px, while code and Markdown begin at
# approximately x=112 px. Seventeen transparent pixels align the visible plot.
PLOT_LEFT_OFFSET_PX = 17
# At the repository page's 894 px iframe width, Markdown runs from x=112 px
# to x=836 px. A 724 px visible plot plus the 17 px left offset therefore
# renders natively without GitHub responsively shrinking the alignment offset.
PLOT_VISIBLE_WIDTH_PX = 724
# GitHub renders stream and plain-text output in a monospace font at roughly
# 8.5 px per character. Two preserved spaces therefore provide approximately
# the same 17 px inset as the transparent padding used for PNG figures.
TEXT_OUTPUT_LEFT_INDENT = "  "


def align_png_outputs(notebook: nbformat.NotebookNode) -> None:
    """Align the visible figure canvas with the notebook content column.

    GitHub positions rich output slightly to the left of Markdown and code.
    A small transparent left offset leaves the plot itself unchanged while
    moving its visible white canvas onto the same left boundary. No matching
    right padding is added because the right edge is already satisfactory.
    """
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            encoded = output.get("data", {}).get("image/png")
            if not encoded:
                continue
            raw = base64.b64decode(encoded)
            with Image.open(io.BytesIO(raw)).convert("RGBA") as image:
                if image.width > PLOT_VISIBLE_WIDTH_PX:
                    scale = PLOT_VISIBLE_WIDTH_PX / image.width
                    image = image.resize(
                        (PLOT_VISIBLE_WIDTH_PX, round(image.height * scale)),
                        Image.Resampling.LANCZOS,
                    )
                canvas = Image.new(
                    "RGBA",
                    (image.width + PLOT_LEFT_OFFSET_PX, image.height),
                    (0, 0, 0, 0),
                )
                canvas.paste(image, (PLOT_LEFT_OFFSET_PX, 0))
                buffer = io.BytesIO()
                canvas.save(buffer, format="PNG")
            output.data["image/png"] = base64.b64encode(buffer.getvalue()).decode("ascii")


def indent_text_output(text: str) -> str:
    """Inset each visible line of native text output by approximately 17 px."""
    return "".join(
        f"{TEXT_OUTPUT_LEFT_INDENT}{line}" if line.strip("\r\n") else line
        for line in text.splitlines(keepends=True)
    )


def align_text_outputs(notebook: nbformat.NotebookNode) -> None:
    """Align native text output with the surrounding notebook content.

    Stream output stores text directly, while execute-result and display-data
    outputs expose a ``text/plain`` representation. PNG-backed display data
    retain their text fallback unchanged because the visible figure receives
    the exact pixel alignment in :func:`align_png_outputs`.
    """
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        for output in cell.get("outputs", []):
            if output.output_type == "stream":
                text = output.get("text", "")
                if isinstance(text, list):
                    text = "".join(text)
                output["text"] = indent_text_output(text)
                continue

            data = output.get("data", {})
            if "image/png" in data or "text/plain" not in data:
                continue
            text = data["text/plain"]
            if isinstance(text, list):
                text = "".join(text)
            data["text/plain"] = indent_text_output(text)


def align_code_outputs(notebook: nbformat.NotebookNode) -> None:
    """Apply the standard publication alignment to every visible output."""
    align_png_outputs(notebook)
    align_text_outputs(notebook)


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

    for cell_index, cell in enumerate(notebook.cells):
        if cell.cell_type == "code" and cell.source.endswith("\n"):
            failures.append(f"trailing blank line in code cell {cell_index}")

    if execute:
        try:
            client = NotebookClient(
                notebook,
                timeout=120,
                kernel_name="python3",
                resources={"metadata": {"path": str(path.parent)}},
            )
            client.execute()
            align_code_outputs(notebook)
            # Preserve only outputs produced by a successful clean-kernel run.
            # GitHub readers can therefore inspect the complete verified lab.
            nbformat.write(notebook, path)
        except Exception as exc:  # Execution exceptions require their full type.
            failures.append(f"clean-kernel execution failed: {type(exc).__name__}: {exc}")

    return failures


def main() -> int:
    """Validate the reusable template structurally and execute published labs."""
    targets = [(ROOT / "templates" / "CH3RRY PI3 Learning Lab Template.ipynb", False)]
    targets.extend(
        (path, True) for path in sorted((ROOT / "notebooks").rglob("*.ipynb"))
    )

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
