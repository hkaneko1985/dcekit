# Version 0.2.2 methods

`MaterialInstance`, `DocumentedValue`, and `ProcessStep` retain reported, unreported-or-unset, and structurally not-applicable states. "Absent from a documented sequence" does not establish omission from the actual experiment. Source IDs and provenance are excluded from distance features.

The NanoMine adapter reparses original strings using `normalization.py`. Quantity type and canonical unit jointly identify scales; mass and volume fractions remain distinct. Numeric prefixes in chemical/trade names are categorical. Unsupported or missing dimensional units produce unknown comparison values, with the original evidence quarantined in the audit. A partially quarantined multivalue group remains unknown rather than being treated as complete.

Repeated numeric settings are unordered multisets when zone identity is absent. Their reported distance is minimum assignment cost; the upper correspondence cost is maximum assignment. Both add unit cost for each unmatched value and divide by the greater cardinality. Repetition is retained. Physical zone order is not recovered.

`MaterialSimilarityEngine(..., feature_policy="active")` excludes jointly unknown entries within an active facet. `feature_policy="fixed_schema"` retains the reference variable schema and treats jointly unknown entries as [0,1]. Joint structural absence is excluded under both. A wholly inactive facet has bounds [0,1] and an undefined reported distance. Reported composites renormalize over defined facets; lower/upper composites retain all positive facet weights. Bounds are deterministic schema-conditional ranges, not confidence intervals. Excluded weight is an accounting quantity, not generally a normalized fraction.

`settings_alignment="occurrence"` matches operation type and occurrence. The `"sequence"` sensitivity instead performs token-only unit-edit alignment with symmetric deterministic tie handling, and treats unlike-token substitutions as unmatched groups. Neither mode guarantees a physical stage mapping.

Primary numeric scales use reported values from the full source resource; query-retrieval scales exclude the query article group. Scale references must be held fixed for cross-resource comparisons. An absent fitted quantity falls back to a scale of one; the reference scale file documents the fitted cases. The underlying dissimilarity need not satisfy a metric or yield a positive-semidefinite kernel.

The held-out metadata and masked-record tasks, same-information field-pooled/equal-weight baselines, bag/set sequence ablations, and bootstrap settings are fixed in `config/revision_evaluation.json`. The field-pooled comparator extends Gower-style available-variable averaging to the same set/edit comparators; it is not described as canonical Gower distance on raw sequences. Counterexamples demonstrate behavior and are not counted as experimental validation. No task chooses an optimal weight.

## Canonical experimental masking

`process_step_keys` is shared by the engine and evaluation. `step_setting_locations` maps canonical keys back to recorded positions. Bilateral masking samples independently with probability 0.5 from the intersection of jointly reported keys, never from matching absolute positions. Unilateral masks sample the left reported keys; article-block masks withhold half the available normalized setting names per source group. The same 300 source pairs as version 0.2.0 are retained. Separate seeded streams for masking and scale bootstrapping prevent accidental coupling. All 300 pairs and the subset with at least one actually hidden field are both summarized; empty masks are not silently removed.

The missingness audit reports pair IDs, replicate, candidate intersection, and actual keys on each side. Synthetic tests cover interleaved/repeated operations, normalization, absent values, and preservation of order/source provenance.

## Supplementary interpretation of close relations

The unchanged 120-record cohort supplies queries and galleries. The original and all other same-article-group candidates are excluded. Query-group-excluded numeric scales and a fixed reference schema are reused before and after masking. Three existing Table 2 profiles (including their strict/lexical comparator choices) are evaluated without optimization. Only reported process settings are hidden, at fractions 0.25/0.50/0.75 with three seeded masks each. Ceiling counts are used. No missing field is filled.

For thresholds 0.1/0.2/0.3/0.4/0.5, accept a documented close relation by (i) reported distance at or below the threshold, (ii) that rule plus common coverage at least 0.50 or 0.75, or (iii) upper distance at or below the threshold. Undefined pre-mask reported comparisons are excluded from the reference evaluation. A contradiction means an accepted pair whose pre-mask reported distance exceeds the threshold; it is not a physical false positive. Retention is accepted/evaluable pairs; reference-close recall is accepted reference-close pairs/all reference-close pairs.

Counts pool directed query/gallery comparisons over repeats. Percentile intervals resample article-group blocks 1,000 times and recompute pooled ratios, conditional on the gallery. They differ from the article-macro held-out-method statistic. Masks and all threshold results are saved. Interval containment is an algebraic consequence of the fixed universe and unchanged remaining values; its empirical contribution is quantifying how many relationships can still be retained. No threshold or weight is selected as optimal.

## Unmasked documentation baseline

The R3 descriptive baseline runs once on unmodified query/candidate records with the same cohort, source-excluded scales, fixed schema, views and thresholds as Figure 7. It shares the article-block bootstrap summarizer and reports a separate set of 60 conditions; it does not insert a new masking rate into the old RNG stream. Denominators are 14,222 comparisons at baseline versus 42,666 across the three masked repeats. Baseline contradiction is necessarily zero for accepted documented-reference relations and undefined for empty acceptance. Results do not establish cluster membership or stability. The declared composition weight implies a constituent upper-distance floor of 0.40 and coverage ceiling of 0.60; other incomplete fields can make these restrictions stronger.
