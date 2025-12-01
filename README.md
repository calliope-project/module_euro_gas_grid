# Module Euro Gas Grid

A module to disaggregate gas networks for European countries.

A modular `snakemake` workflow built for [`clio`](https://clio.readthedocs.io/) data modules.

> [!CAUTION]
> This module is in development, with no official release.
> Based on the work of [N. Ortiz Torres](https://repository.tudelft.nl/record/uuid:8ce12f5f-a3c6-49f7-bf00-2a6dc685d303) and [PyPSA-Eur](https://github.com/PyPSA/pypsa-eur).

## Using this module

This module can be imported directly into any `snakemake` workflow.
Please consult the integration example in `tests/integration/Snakefile` for more information.

## Development

We use [`pixi`](https://pixi.sh/) as our package manager for development.
Once installed, run the following to clone this repo and install all dependencies.

```shell
git clone git@github.com:calliope-project/module_euro_gas_grid.git
cd module_euro_gas_grid
pixi install --all
```

For testing, simply run:

```shell
pixi run test-integration
```

To view the documentation locally, use:

```shell
pixi run serve-docs
```

To test a minimal example of a workflow using this module:

```shell
pixi shell    # activate this project's environment
cd tests/integration/  # navigate to the integration example
snakemake --use-conda --cores 2  # run the workflow!
```
