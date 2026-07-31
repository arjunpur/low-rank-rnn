"""Prepare a temporary notebook for reproducible report figure exports."""

from __future__ import annotations

import json
import sys
from pathlib import Path


REPORT_EXPORT_ALIAS = "save_report_figure = plotting.save_report_figure\n"

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

TRAINING_INPUT_OUTPUT_EXPORT = """\
rank1_training_examples = np.concatenate((positive_examples, negative_examples))
rank1_training_input_output_figure, _ = plotting.plot_trial_outputs(
    rank1_test_inputs[rank1_training_examples],
    rank1_test_labels[rank1_training_examples],
    rank1_outputs[rank1_training_examples],
    decision_steps=RANK1_DECISION_STEPS,
)
save_report_figure(
    rank1_training_input_output_figure,
    "rank1-training-input-output.pdf",
)
plt.close(rank1_training_input_output_figure)
"""

PLOT_EXPORTS = (
    (
        "plotting.plot_reduced_system_trajectories(\n",
        "rank1_reduced_trajectories_figure",
        "rank1-reduced-trajectories.pdf",
    ),
    (
        "plotting.plot_fixed_points(\n",
        "rank1_fixed_points_figure",
        "rank1-fixed-points.pdf",
    ),
    (
        "plotting.plot_memory_trials(\n",
        "rank2_memory_trials_figure",
        "rank2-memory-trials.pdf",
    ),
    (
        "plotting.plot_memory_behavior(\n",
        "rank2_memory_behavior_figure",
        "rank2-memory-behavior.pdf",
    ),
    (
        "plotting.plot_latent_sweeps(\n",
        "rank2_latent_sweeps_figure",
        "rank2-latent-sweeps.pdf",
    ),
    (
        "plotting.plot_latent_plane(\n",
        "rank2_latent_plane_figure",
        "rank2-latent-plane.pdf",
    ),
    (
        "plotting.plot_delay_mse(\n",
        "rank2_delay_generalization_figure",
        "rank2-delay-generalization.pdf",
    ),
    (
        "plotting.plot_connectivity_covariance(\n",
        "rank2_loading_covariance_figure",
        "rank2-loading-covariance.pdf",
    ),
    (
        "plotting.plot_mse_comparison(\n",
        "rank2_gaussian_resampling_figure",
        "rank2-gaussian-pipeline.pdf",
    ),
    (
        "plotting.plot_circuit_mse_comparison(\n",
        "rank2_circuit_performance_figure",
        "rank2-circuit-performance-comparison.pdf",
    ),
)


def code_cell(source: str, cell_id: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def assign_and_export_plot(
    notebook: dict[str, object],
    call_prefix: str,
    variable_name: str,
    filename: str,
    *,
    occurrence: int = 0,
) -> None:
    """Assign one plotting call and export its returned notebook figure."""
    plot_cells = [
        cell
        for cell in notebook["cells"]
        if call_prefix in "".join(cell.get("source", []))
    ]
    plot_cell = plot_cells[occurrence]
    source = "".join(plot_cell["source"])
    call_index = source.index(call_prefix)
    source = (
        source[:call_index]
        + f"{variable_name}, _ = "
        + source[call_index:]
    )
    show_index = source.index("plt.show()", call_index)
    export = (
        f'save_report_figure(\n'
        f"    {variable_name},\n"
        f'    "{filename}",\n'
        f")\n"
    )
    source = source[:show_index] + export + source[show_index:]
    plot_cell["source"] = source.splitlines(keepends=True)


def export_existing_figure(
    notebook: dict[str, object],
    marker: str,
    variable_name: str,
    filename: str,
) -> None:
    """Export a figure that the notebook already stores in a variable."""
    plot_cell = next(
        cell
        for cell in notebook["cells"]
        if marker in "".join(cell.get("source", []))
    )
    source = "".join(plot_cell["source"])
    marker_index = source.index(marker)
    show_index = source.index("plt.show()", marker_index)
    export = (
        f'save_report_figure(\n'
        f"    {variable_name},\n"
        f'    "{filename}",\n'
        f")\n"
    )
    source = source[:show_index] + export + source[show_index:]
    plot_cell["source"] = source.splitlines(keepends=True)


def prepare_notebook(source_path: Path, output_path: Path) -> None:
    notebook = json.loads(source_path.read_text())
    repository_directory = source_path.resolve().parent

    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []

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
        if "plotting.set_plot_style()" in "".join(cell.get("source", []))
    )
    notebook["cells"].insert(
        setup_index + 1,
        code_cell(REPORT_EXPORT_ALIAS, "report-export-alias"),
    )

    trial_plot_index = next(
        index
        for index, cell in enumerate(notebook["cells"])
        if "rank1_examples =" in "".join(cell.get("source", []))
    )
    notebook["cells"].insert(
        trial_plot_index + 1,
        code_cell(
            TRAINING_INPUT_OUTPUT_EXPORT,
            "report-training-input-output",
        ),
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

    for call_prefix, variable_name, filename in PLOT_EXPORTS:
        assign_and_export_plot(
            notebook,
            call_prefix,
            variable_name,
            filename,
        )

    export_existing_figure(
        notebook,
        "rank1_reduced_summary_figure =",
        "rank1_reduced_summary_figure",
        "rank1-reduced-summary.pdf",
    )
    export_existing_figure(
        notebook,
        "paper_mode_time_course_figure,",
        "paper_mode_time_course_figure",
        "rank2-paper-latent-sweeps.pdf",
    )
    assign_and_export_plot(
        notebook,
        "plotting.plot_delay_mse(\n",
        "rank2_paper_delay_generalization_figure",
        "rank2-paper-delay-generalization.pdf",
        occurrence=1,
    )

    output_path.write_text(json.dumps(notebook, indent=1) + "\n")


if __name__ == "__main__":
    prepare_notebook(Path(sys.argv[1]), Path(sys.argv[2]))
