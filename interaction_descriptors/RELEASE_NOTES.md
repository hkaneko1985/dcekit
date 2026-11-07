# DCEKit publication preparation — v53

Date: 7 November 2026. Author: Hiromasa Kaneko.
Software version: `0.1.0` (initial public candidate).
Host: the `interaction_descriptors/` research directory in DCEKit.
The v53 label identifies this preparation stage and does not change DCEKit's
own software version. No remote commit or public release was created here.

## Scope of this review

The v52 public candidate was reviewed for the inference/input contracts,
scientific numerical boundaries, reproduction commands, external-input
requirements, DCEKit placement, and distributable contents. Original experiment
routines, source weights, split indices, candidate grids, and the complete
144-cell scientific summary remain unchanged. No new material prediction
experiment or source fit was performed during this publication review.

## Corrections to the reusable core

### Preserve a positive trace fraction in normalized pairs

Both pair fractions are now evaluated directly as `xi/(xi+xj)` and
`xj/(xi+xj)`. Previously the second was computed as `1-t`, which could round to
zero when the first fraction rounded to one. The direct fractions are used in
both the pair input and the model-endpoint reference.

The concrete regression case uses `x=[1, 1e-20]` and the continuous quadratic
function `g(x)=1e20*x[0]*x[1]`. Under the accepted floating-point simplex
tolerance, the whole value is 1. The earlier boundary value was 0; the corrected
boundary value is 1. The test also reverses the component ordering. This fixes
the binary identity for the concrete numerical edge case without dropping the
trace constituent. Ordinary four-row toy outputs changed by at most
`5.551115123125783e-17` in the boundary result; whole outputs and missingness
positions were unchanged.

### Validate component and output-coordinate names consistently

The standalone `boundary_extension` function now uses the same explicit name
validator as input alignment. A bare string such as `components="Fe"` is rejected
instead of being interpreted as two names, `"F"` and `"e"`. Bare strings for
`coordinate_names` are rejected too. Pass sequences, for example `["Fe", "Ni"]`.

The original 14 core tests and the two specific new regression tests all pass:
**16 tests in total**. This count concerns the reusable API tests, not predictive
experiments. `verification/synthetic_validation_v52.json` retains the earlier
14-test record; `verification/synthetic_validation.json` records the current
16-test execution. `verification/publication_review53.json` records the current
release review and its separately executed checks.

## Reproduction-command corrections

- External input requirements are checked before an isolated work directory is
  populated, with concise errors for missing assets.
- Figure 8 generation respects output-directory protection.
- Synthetic validation accepts an explicit output path, so a user's check can
  write its own record instead of replacing archived verification evidence.
- Source-fit property-output diagnostics explicitly report zero applicable target
  rows as not applicable, with null differences instead of an empty-array error
  or a fabricated zero difference. Mask and latent-output checks remain separate.
  Six NumPy-only tests in `tests_pipeline/` verify this final diagnostic patch
  without fitting source or target models.
- Scientific training/prediction routines remain separate from these input and
  output checks. The exact paths and executed review results are recorded in
  `verification/publication_review53.json`.

## Documentation and integration

- The top-level folder is `interaction_descriptors/`, intended for the DCEKit
  repository root beside its existing research directories.
- English and Japanese guides explain explicit installation, a source-free
  example, aggregate table/Figure 8 exports, and external-asset requirements.
- `DCEKIT_INTEGRATION.md` provides the placement instructions, a root-README
  link suggestion, and commit-specific citation guidance.
- `CITATION.cff` is retained inside this research directory. Its repository URL
  names the real DCEKit host; no nonexistent subdirectory URL, DOI, commit, or
  release tag is presented as already published.
- Author code is under MIT. Third-party row-level inputs and model weights
  have separate provenance and are not included in this public ZIP.

## Scientific reproduction already established

The v52 pipeline records one exact original source fit, the corresponding
143-composition output parity, Figures 1–6 with byte-identical images, Figure 7
as a data-equivalent redraw, Figure 8 from the complete aggregate, and all
16 main/supplement table units. The full source bank and every historical
fitting branch were not rerun. `PIPELINE.md` and `pipeline_status.json` retain
those scope distinctions. New public-core fixes are not silently applied to
the archived routines used to obtain the manuscript's reported results.

## Remaining publication actions

1. Add this folder to DCEKit and check the source-free quick start at the real
   location.
2. Record the actual public commit and use its directory URL in the manuscript.
3. Specify how separately distributed data/model assets can be obtained under
   their own conditions.
4. Complete the journal-specific submission format and final author declarations.

The sole author and KAKENHI numbers remain Hiromasa Kaneko and
24K08152/24K01234, respectively.
