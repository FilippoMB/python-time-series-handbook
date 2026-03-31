# Contributing

This repository intentionally uses two environments with different audiences:

- Users use [environment.yml](environment.yml) with Conda to run the notebooks.
- Maintainers and CI use [pyproject.toml](pyproject.toml) plus [uv.lock](uv.lock).

The goal is to keep the user workflow simple without giving up the speed and reproducibility of `uv` for development and automation.

## Python version

The repository targets Python 3.12 for notebook development, CI, and the generated Conda environment.

## Maintainer setup

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Sync the maintainer environment:

   ```bash
   uv sync --extra notebooks --group docs --group deploy
   ```

3. Run Jupyter Notebook if needed:

   ```bash
   uv run jupyter notebook
   ```

Classic slideshow support depends on `rise`, which still imports `pkg_resources`, so keep the `setuptools<81` constraint until RISE drops that dependency.

On macOS, notebook `12 - Time series classification and clustering` may require `libomp`:

```bash
brew install libomp
```

## Dependency maintenance

When you change direct dependencies, keep the environments in sync in this order:

1. Edit [pyproject.toml](pyproject.toml).
2. Refresh the lockfile:

   ```bash
   uv lock
   ```

3. Regenerate the user Conda environment:

   ```bash
   uv run --no-sync python scripts/sync_user_environment.py
   ```

4. Validate the result:

   ```bash
   uv run --no-sync python scripts/sync_user_environment.py --check
   uv sync --extra notebooks --group docs
   uv run --extra notebooks --group docs jupyter-book build . --builder html
   ```

CI checks both the `uv` workflow and the user Conda environment.

## Notebook diffs

Keep notebook changes as small as possible.

- Avoid committing execution timestamps or unrelated output churn when only markdown or source cells changed.
- If a notebook must be re-executed, prefer changes that are reproducible and explain stability-related parameter changes in the markdown next to the code.
