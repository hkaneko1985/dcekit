# Current results

Historical resource count: 8,827 (5,001 Starrydata; 1,891 HTEM; 832 NanoMine; 1,103 PNCExtract). This is not a common-API corpus.

Common HTEM reported composition/settings ARI against composition: 0.938980.

Same-information metadata benchmarks (article-group macro means):

| Task | Method | Metric | Mean | 95% group interval |
|---|---|---|---:|---|
| heldout_method | balanced_strict | agreement_at_5 | 0.406951 | 0.356094–0.454771 |
| heldout_method | equal_facets | agreement_at_5 | 0.423961 | 0.369290–0.476222 |
| heldout_method | gower_field_pool | agreement_at_5 | 0.419176 | 0.361635–0.474238 |
| heldout_method | random_gallery | agreement_at_5 | 0.326198 | 0.325703–0.326690 |
| heldout_method | sequence_as_bag | agreement_at_5 | 0.417704 | 0.366071–0.463329 |
| heldout_method | sequence_as_set | agreement_at_5 | 0.419241 | 0.372539–0.465675 |
| masked_record | balanced_strict | mrr | 0.992521 | 0.985043–0.998932 |
| masked_record | equal_facets | mrr | 0.992521 | 0.985043–0.998932 |
| masked_record | gower_field_pool | mrr | 0.992521 | 0.985043–0.998932 |
| masked_record | sequence_as_bag | mrr | 0.992521 | 0.985043–0.998932 |
| masked_record | sequence_as_set | mrr | 0.992521 | 0.985043–0.998932 |

No single best weighting is established. The original-record retrieval task is near a ceiling. Source labels provide only contextual support. These results do not establish property equivalence or transfer learning performance.

The 39 software/artifact checks are separate from these scientific evaluations.

## Re-review masking and interpretation

The same 300 source pairs were retained. Corrected canonical bilateral masking yields pair-active interval containment 57.3% overall, or 58/186 (31.2%) among nonempty masks; fixed-reference containment is 100% in both denominators. There are 335 actually hidden common keys, with zero bilateral key mismatches. Unilateral and source-block all-pair containment is 87.0% and 89.7% for pair-active intervals.

In the original-excluded half-settings-mask experiment, protocol-view point acceptance at threshold 0.5 retains 1,350/42,666 directed comparisons, with 306 contradictions (22.7%) against pre-mask documented distance. Upper-distance acceptance retains 292, with no contradiction and 18.9% reference-close recall. Coverage filtering helps in some conditions and worsens conditional contradiction rates in others. Empty sets have undefined contradiction rate. These quantify the cost of conservative documentation interpretation, not chemical accuracy. All 180 aggregate conditions are supplied; none is chosen as an optimum.

## Baseline documentation before added masks

Table S15 supplies all 60 baseline conditions from one pass over 120 queries / 93 article groups (14,222 comparisons). Protocol upper-distance acceptance at threshold 0.5 is 132/14,222 = 0.928%, versus 292/42,666 = 0.684% under 50% masking, a 0.244-percentage-point decrease. Constituent point acceptance at threshold 0.3 is 832/14,222 versus 2,496/42,666: both 5.850%. Observed baseline upper-distance minima are 0.657 (constituent) and 0.420 (balanced); their maximum common coverages are 0.514 and 0.607. One record has no reported settings and contributes nine empty masks to the unchanged original experiment. These quantities concern documented pairwise decisions only.
