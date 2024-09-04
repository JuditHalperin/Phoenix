## Overview

This tool is designed for pathway analysis in single-cell RNA sequencing data, focusing on identifying important pathways that distinguish between cell-types and across pseudo-time.
It evaluates biological pathways for their predictive power in two key areas: predicting cell-types (discrete values) and estimating pseudo-time (continuous values), using classification and regression models, respectively.
To assess the significance of the identified pathways, the tool compares the performance of each gene set against a random set of genes of equivalent size.
Gene annotations that significantly outperform random gene sets are considered particularly relevant within the specific context of the data.


## Installation

To install the tool, clone this repository:

```
git clone https://github.com/NachmaniLab/pathway-analysis.git
cd pathway-analysis
```

Then, install the required dependencies using either `venv` or `conda`:

### Using `venv`

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Using `conda`

```
conda create prefix .venv python=3.11
conda activate ./.venv
conda install --file requirements.txt
```


## Usage

To run the tool, execute the `run.py` script with the relevant parameters.

### Basic arguments

Provide input data:

* `expression`: Path to single-cell raw expression data where rows represent cells and columns represent gene symbols (CSV file).
* `reduction`: Path to dimensionality reduction coordinates where rows represent cells and columns include names of the first two components (CSV file), or a dimensionality reduction method: `pca`, `umap` or `tsne`. Default: `umap`.

Provide at least one target values:

* `cell_types`: Path to cell-type annotations where rows represent cells and first column presents cell-types (CSV file).
* `pseudotime`: Path to pseudo-time values where rows represent cells and columns include names of different trajectories (CSV file).

Specify a known pathway database for a specific organism, or provide a custom gene set list:

* `organism`: Organism name for pathway annotations.
* `pathway_database`: Known pathway database: `go`, `kegg` or `msigdb`.
* `custom_pathways`: Path to custom gene sets where columns represent set names and rows include gene symbols (CSV file).

Provide output path:

* `output`: Path to output directory.

### Additional arguments

Customize preprocessing parameters:

* `preprocessed`: Whether expression data are log-normalized. Default: `False`.
* `exclude_cell_types`: Cell-types to exclude from the analysis.
* `exclude_lineages`: Lineages to exclude from the analysis.

Customize feature selection parameters:

* `feature_selection`: Feature selection method applied to each gene set: `ANOVA` or `RF`. Default: `ANOVA`.
* `set_fraction`: Fraction of genes to select from each gene set. Default: `0.75`.
* `min_set_size`: Minimum number of genes to select from each gene set. Default: `5`.

Customize prediction model parameters:

* `classifier`: Classification model: `Reg`, `KNN`, `SVM`, `DTree`, `RF`, `LGBM`, `XGB`, `GradBoost` or `MLP`. Default: `RF`.
* `regressor`: Regression model: `Reg`, `KNN`, `SVM`, `DTree`, `RF`, `LGBM`, `XGB`, `GradBoost` or `MLP`. Default: `RF`.
* `classification_metric`: Classification score: `accuracy`, `accuracy_balanced`, `f1`, `f1_weighted`, `f1_macro`, `f1_micro`, `f1_weighted_icf` or `recall_weighted_icf`. Default: `f1_weighted_icf`.
* `regression_metric`: Regression score: `neg_mean_absolute_error`, `neg_mean_squared_error` or `neg_root_mean_squared_error`. Default: `neg_root_mean_squared_error`.
* `cross_validation`: Number of cross-validation folds. Default: `10`.
* `seed`: Seed for reproducibility. Default: `3407`.

Customize background distribution parameters:

* `repeats`: Size of background distribution. Default: `200`.
* `distribution`: Type of background distribution: `gamma` or `normal`. Default: `gamma`.

Include parameters relevant for parallelization on a high-computing cluster, which is highly recommended for large pathway databases:

* `processes`: Number of processes to run in parallel. Default: `0`.

For a full list of available parameters, run:

```
python run.py --help
```

### Basic example

```
python run.py \
    --expression my_experiment/input/expression.csv \
    --reduction my_experiment/input/reduction.csv \
    --cell_types my_experiment/input/cell_types.csv \
    --pseudotime my_experiment/input/pseudotime.csv \
    --custom_pathways my_experiment/input/gene_sets.csv \
    --output my_experiment/output
```

### Advanced example

```
python run.py \
    --expression my_experiment/input/expression.csv \
    --cell_types my_experiment/input/cell_types.csv \
    --pseudotime my_experiment/input/pseudotime.csv \
    --output my_experiment/output \
    --reduction tsne \
    --organism human \
    --pathway_database msigdb \
    --set_fraction 0.5 \
    --processes 20
```

### Visualization

For additional visualization of specific pathways, run the `plot.py` script with parameters:

* `output`: Path to the output directory containing all tool results.
* `pathway`: Pathway name to plot.
* `cell_type`: Cell-type target column to plot.
* `lineage`: Trajectory target column to plot.
* `all`: Whether to plot all pathways for all cell-type and lineage targets. Default: `False`.

For example:

```
python plot.py \
    --output my_experiment/output \
    --pathway HALLMARK_HEME_METABOLISM
```


## Tests

To run all tests, use the `test.py` script:

```
python test.py
```

Alternatively, use the `unittest` module:

```
python -m unittest discover tests
```


