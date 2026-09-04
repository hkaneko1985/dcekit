# Reviewer-driven revision notes

Date: 2026-09-02

This package is the revised computational companion to the manuscript. The
following changes address the methodological issues identified during internal
peer review.

## Analysis changes

1. **Independent unit for Bayesian optimization.** Inferential comparisons now
   reduce the five paired repetitions to one median difference per engineering
   system. Exact sign enumeration implements the two-sided Wilcoxon signed-rank
   test after zero differences are discarded. Whole-dataset bootstrap intervals
   accompany the tests; repetition-level counts are descriptive only.
2. **D5 failure audit.** The methanol process retained 104/125 requested
   training simulations and 42/65 requested external simulations. D5 is
   excluded from primary forward, inverse, and Bayesian-optimization inference
   and is retained only as an explicitly labeled exploratory analysis.
3. **Stronger forward baselines.** Training-shell-tuned RBF-GPR, RBF-SVR, and
   Ridge baselines were added. The external-test response standard deviation is
   the primary reporting scale; training-SD and test-range scales remain in the
   archive as sensitivity analyses.
4. **Common inverse feasibility rule.** Absolute-GMR, Transition-GMR, and
   EGC-GMR all discard out-of-bounds candidates. No method clips a candidate to
   a physical bound. Primary decision utility includes every requested target,
   treating rejection or failed verification as failure rather than selecting
   only EGC-accepted targets.
5. **D7 leakage correction.** Concrete inverse bounds use training inputs only.
6. **Uncertainty reporting.** Supported 90% coverage is reported with interval
   width and the proper interval score. Selective support is reported
   separately.

## Implementation and reproducibility changes

- The manuscript's anchor equation now matches the implemented segment search.
- GMR calibration equations now state the absolute and transition evidence
  definitions, anchor-count term, response shell, direction count, quantile,
  and scale.
- Dataset regeneration can record attempted inputs for every failed or
  non-finite simulation before exclusion.
- `test_revision_invariants.py` checks the corrected inferential unit, exact
  signed-rank calculation, D7 bounds, no-clipping rule, interval score, and D5
  manifest.
- `verify_reference_results.py` checks archived-result consistency without
  presenting that check as an independent refit.
- `Dockerfile`, pinned environment files, data-license notes, and SHA-256 file
  checksums were added.

## Revised headline interpretation

The revised results show descriptive improvements over fixed and tuned RBF
baselines on several systems, but no corrected comparison establishes universal
or statistically significant superiority. EGC-GMR is competitive with
Transition-GMR on all-target inverse utility. Evidence-gated Bayesian
optimization has uncertain attainment benefit at the six-dataset level, while
its measurable contribution is explicit support control, improved conditional
interval calibration, and greater batch separation. The revised manuscript is
written to reflect these limits.
