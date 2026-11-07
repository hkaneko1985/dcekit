# Add this research directory to DCEKit

## Destination and name

Host repository: [hkaneko1985/dcekit](https://github.com/hkaneko1985/dcekit).
Add the complete **`interaction_descriptors/`** folder at the repository root,
beside the existing `dcekit/` Python package and the other research directories.
The ZIP has exactly that one top-level folder. Extract it into a separate location,
then copy the folder into your DCEKit checkout.

The name covers both whole-composition and complete-pair descriptions and is
not tied to one neural architecture or to a fixed number of components. The
directory name and Python import name are both `interaction_descriptors`;
the distribution/command name is `interaction-descriptors`.

| Name considered | Assessment |
|---|---|
| `interaction_descriptors` | Recommended: short and consistent with the existing Python import |
| `learned_interaction_descriptors` | Makes learned origin explicit but lengthens commands and paths |
| `composition_interaction_descriptors` | Makes the composition scope explicit but is longer |

## Relationship to the existing DCEKit package

This is a self-contained research package with `pyproject.toml` and a `src/`
layout. The outer research directory and `src/` have no `__init__.py`.
DCEKit's currently observed root `setup.py` uses `setuptools.find_packages()`;
it does not traverse those two non-package directories to discover this package.
The release review checked that package-discovery behavior with the actual
package-marker layout. It did not clone and rebuild every DCEKit component.

No modification to DCEKit's root `setup.py`, root version, package imports,
license files, or dependencies is part of this addition. Do not place the folder
inside the inner `dcekit/` package, and do not add `__init__.py` to the outer
research folder or its `src/` directory. `pip install dcekit` alone does not
install this research package.

## Local installation and quick check

From the DCEKit repository root, using the Python environment in which you want
to run the research code:

```bash
python -m pip install -e ./interaction_descriptors
cd interaction_descriptors
python -m interaction_descriptors synthetic --output example_output
python scripts/verify_synthetic.py --output example_output/synthetic_validation.json
python scripts/reproduce.py tables --output-dir example_output/main_tables
python scripts/reproduce.py regressor-tables --output-dir example_output/regressor_tables
```

The core requires Python 3.10+ and NumPy. The synthetic example uses artificial
compositions and a toy model and needs no materials dataset or trained weights.
Table export uses the included aggregates and the Python standard library.
The paths above must be new or compatible with the documented command's output
policy; use separate output directories for separate executions.

For Figure 8, install the plotting dependencies in the same environment:

```bash
python -m pip install -r requirements-plot.txt
python scripts/reproduce.py plot-comparison --summary experiments/regressor_comparison/frozen_reference/summary.csv --output-dir example_output/figure8
```

The complete historical study needs separately obtained source/target inputs and
model assets. [PIPELINE.md](PIPELINE.md) gives the verified routes and exact scope.
The ZIP itself is sufficient for the synthetic example and aggregate table/Figure 8
exports, once their declared dependencies are installed.

## Optional link from the DCEKit root README

Add a link to the existing root README in a suitable research-code section:

```markdown
- [Interaction-aware descriptors](interaction_descriptors/README.md): frozen composition descriptors, complete-pair boundary extensions, and research reproduction commands. [日本語](interaction_descriptors/README_JA.md)
```

Keep this folder's `CITATION.cff` and `LICENSE` inside the folder. The citation is
for this research software and should not replace a citation for DCEKit as a whole.
The author-code MIT license does not change the separate distribution conditions
of third-party observations or model-weight assets.

## Publication record

The prepared ZIP does not attest that the new directory is already online. No
remote write is performed by these instructions. Once the directory has been
uploaded and its quick check has passed, record the actual DCEKit commit SHA.
Use that commit-specific directory URL in the manuscript and software citation
when a fixed code snapshot is needed. Do not invent a DOI or a release tag.

The internal Python package version is `0.1.0`; the v53 archive label records
this preparation/review stage and is not a DCEKit release number. A repository-wide
`v0.1.0` tag should not be created solely for this subdirectory.

Current hosting structure, package discovery, and MIT status were inspected on
7 November 2026 from the official repository:

- <https://github.com/hkaneko1985/dcekit>
- <https://github.com/hkaneko1985/dcekit/blob/master/setup.py>
- <https://github.com/hkaneko1985/dcekit/blob/master/LICENSE.txt>
