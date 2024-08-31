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

#### Using `venv`:

```
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Using `conda`:

```
conda create --prefix .venv python=3.11
conda activate ./.venv
conda install --file requirements.txt
```


## Usage

To run the tool, execute the `run.py` script with the relevant parameters.

#### Required arguments:


#### Optional arguments:


For additional visualization of specific pathways, run the `plot.py` script:


## Tests

To run all tests, use the `test.py` script:

```
python test.py
```

Alternatively, use the `unittest` module:

```
python -m unittest discover tests
```


