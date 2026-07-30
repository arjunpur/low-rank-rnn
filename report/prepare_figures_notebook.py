"""Prepare a temporary notebook for reproducible report figure exports."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PANEL_EXPORT_HELPER = """\
def save_report_figure(figure: plt.Figure, filename: str) -> Path:
    \"\"\"Label data panels and save a vector PDF for the LaTeX report.\"\"\"
    if filename in {
        "rank1-input-output-examples.pdf",
        "rank1-training-input-output.pdf",
    }:
        figure.subplots_adjust(top=0.88)

    data_axes = [
        axis
        for axis in figure.axes
        if axis.get_visible() and axis.get_label() != "<colorbar>"
    ]
    if len(data_axes) > 1:
        panel_label_size = max(12, 1.1 * figure.get_figwidth())
        for index, axis in enumerate(data_axes):
            panel_label = axis.annotate(
                f"({chr(ord('a') + index)})",
                xy=(0, 1),
                xycoords="axes fraction",
                xytext=(-8, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
                fontsize=panel_label_size,
                fontweight="bold",
                annotation_clip=False,
            )
            panel_label.set_clip_on(False)

    REPORT_FIGURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    output_path = REPORT_FIGURE_DIRECTORY / filename
    with plt.rc_context({"savefig.bbox": None}):
        figure.savefig(output_path, format="pdf")
    return output_path
"""

GAUSSIAN_ASSIGNMENT = """\
(
    rank1_gaussian_loading_figure,
    rank1_gaussian_accuracy_figure,
    _,
    _,
    _,
) = plotting.plot_gaussian_sampling_summary(
"""

GAUSSIAN_EXPORTS = """\
save_report_figure(
    rank1_gaussian_loading_figure,
    "rank1-gaussian-loading-samples.pdf",
)
save_report_figure(
    rank1_gaussian_accuracy_figure,
    "rank1-gaussian-network-accuracy.pdf",
)
"""


def code_cell(source: str, cell_id: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def prepare_notebook(source_path: Path, output_path: Path) -> None:
    notebook = json.loads(source_path.read_text())
    repository_directory = source_path.resolve().parent

    first_code_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if cell["cell_type"] == "code"
    )
    notebook["cells"].insert(
        first_code_index,
        code_cell(
            "import os\n"
            f"os.chdir({str(repository_directory)!r})\n",
            "report-build-directory",
        ),
    )

    setup_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "def save_report_figure" in "".join(cell.get("source", []))
    )
    notebook["cells"].insert(
        setup_index + 1,
        code_cell(PANEL_EXPORT_HELPER, "report-panel-export"),
    )

    gaussian_cell = next(
        cell
        for cell in notebook["cells"]
        if "plotting.plot_gaussian_sampling_summary"
        in "".join(cell.get("source", []))
    )
    gaussian_source = "".join(gaussian_cell["source"])
    if "rank1_gaussian_loading_figure" not in gaussian_source:
        gaussian_source = gaussian_source.replace(
            "plotting.plot_gaussian_sampling_summary(\n",
            GAUSSIAN_ASSIGNMENT,
            1,
        )
        gaussian_source = gaussian_source.replace(
            "plt.show()\n",
            f"{GAUSSIAN_EXPORTS}plt.show()\n",
            1,
        )
        gaussian_cell["source"] = gaussian_source.splitlines(keepends=True)

    output_path.write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    prepare_notebook(Path(sys.argv[1]), Path(sys.argv[2]))
