# Rocket Money

### Index

- `data` - directory contains the small sample data set.
- `model_bert.py` - script for testing the BERTopic model.
- `metrics.py` - module contains methods for calculating performance metrics.

### Run Docker Image

Run the following terminal commands within the `rocket-money` dir:

1. `docker build -t rocket-money -f ./Dockerfile ./`
2. `docker run -it rocket-money`

### Models Tested

`BERTopic.UMAP.01` - This baseline BERTopic model with UMAP dimensionality
reduction and a 0.01 min distance parameter.
`BERTopic.UMAP.1` - This is the same as UMAP but with a 0.1 min distance param.
`BERTopic.MPNET.01` - This is the the baseline BERTopic UMAP model. However, the 
original MiniLM "all-MiniLM-L6-v2" has been swapped for the MPNET
"all-mpnet-base-v2" embedding model, which should have better performance. It
uses a 0.01 min distance parameter for dimensionality reduction.
`BERTopic.MPNET.1` - This is the same MPNET model but it uses a 0.01 minimum
distance parameter for dimensionality reduction.

### Setup Git Hooks

The files `.flake8`, `pyproject.toml`, and `.pre-commit-config.yaml` are used, 
in addition to the black code formatter, to autoformat code that meets code
style guidelines. e.g. line length must be <= 80. In order to use this, follow 
these steps:

1. Within the project repo, run `pre-commit install`.
2. Then run `pre-commit autoupdate`.
3. To run pre-commit git hooks for flake8 and black run use 
`pre-commit run --all-files`.