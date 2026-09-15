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

The 35 software/artifact checks are separate from these scientific evaluations.
