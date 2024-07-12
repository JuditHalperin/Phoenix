from scripts.args import get_plot_args
from scripts.visualization import plot, plot_experiment


def plot_tool(
        cell_type: str,
        lineage: str,
        pathway: str,
        all_plots: bool,
        output: str,
    ):
    if all_plots:
        plot(output)

    if cell_type:
        plot_experiment(output, cell_type, pathway, 'cell_types')

    if lineage:
        plot_experiment(output, lineage, pathway, 'pseudotime')


if __name__ == '__main__':
    args = get_plot_args()
    plot_tool(**args)
