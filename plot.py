from scripts.args import get_plot_args
from scripts.visualization import plot, plot_experiment


def plot_tool(
        cell_type: list[str],
        lineage: list[str],
        pathway: list[str],
        all: bool,
        output: str,
    ):
    if not pathway:
        plot(output, all=all)

    for p in pathway:
        for c in cell_type:
            plot_experiment(output, c, p, 'cell_types', 'cell_type_classification', 'cell_types')
        for l in lineage:
            plot_experiment(output, l, p, 'pseudotime', 'pseudotime_regression', 'pseudotime')


if __name__ == '__main__':
    args = get_plot_args()
    plot_tool(**vars(args))
