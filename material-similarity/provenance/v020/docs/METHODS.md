# Version 0.2.0 methods

`MaterialInstance`, `DocumentedValue`, and `ProcessStep` retain reported, unreported-or-unset, and structurally not-applicable states. "Absent from a documented sequence" does not establish omission from the actual experiment. Source IDs and provenance are excluded from distance features.

The NanoMine adapter reparses original strings using `normalization.py`. Quantity type and canonical unit jointly identify scales; mass and volume fractions remain distinct. Numeric prefixes in chemical/trade names are categorical. Unsupported or missing dimensional units produce unknown comparison values, with the original evidence quarantined in the audit. A partially quarantined multivalue group remains unknown rather than being treated as complete.

Repeated numeric settings are unordered multisets when zone identity is absent. Their reported distance is minimum assignment cost; the upper correspondence cost is maximum assignment. Both add unit cost for each unmatched value and divide by the greater cardinality. Repetition is retained. Physical zone order is not recovered.

`MaterialSimilarityEngine(..., feature_policy="active")` excludes jointly unknown entries within an active facet. `feature_policy="fixed_schema"` retains the reference variable schema and treats jointly unknown entries as [0,1]. Joint structural absence is excluded under both. A wholly inactive facet has bounds [0,1] and an undefined reported distance. Reported composites renormalize over defined facets; lower/upper composites retain all positive facet weights. Bounds are deterministic schema-conditional ranges, not confidence intervals. Excluded weight is an accounting quantity, not generally a normalized fraction.

`settings_alignment="occurrence"` matches operation type and occurrence. The `"sequence"` sensitivity instead performs token-only unit-edit alignment with symmetric deterministic tie handling, and treats unlike-token substitutions as unmatched groups. Neither mode guarantees a physical stage mapping.

Primary numeric scales use reported values from the full source resource; query-retrieval scales exclude the query article group. Scale references must be held fixed for cross-resource comparisons. An absent fitted quantity falls back to a scale of one; the reference scale file documents the fitted cases. The underlying dissimilarity need not satisfy a metric or yield a positive-semidefinite kernel.

The held-out metadata and masked-record tasks, same-information field-pooled/equal-weight baselines, bag/set sequence ablations, and bootstrap settings are fixed in `config/revision_evaluation.json`. The field-pooled comparator extends Gower-style available-variable averaging to the same set/edit comparators; it is not described as canonical Gower distance on raw sequences. Counterexamples demonstrate behavior and are not counted as experimental validation. No task chooses an optimal weight.
