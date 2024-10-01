# Rocket Money

### Info

- `data` - directory contains the small sample data set.
- `model_bert.py` - script for testing the BERTopic model.

### Setup Git Hooks

The files `.flake8`, `pyproject.toml`, and `.pre-commit-config.yaml` are used, 
in addition to the black code formatter, to autoformat code that meets code
style guidelines. e.g. line length must be <= 80. In order to use this, follow 
these steps:

1. Within the project repo, run `pre-commit install`.
2. Then run `pre-commit autoupdate`.
3. To run pre-commit git hooks for flake8 and black run use 
`pre-commit run --all-files`.