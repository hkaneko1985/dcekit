# Draft inquiry about NanoMine input access

Unsent draft for the author to review. This document is not permission, a deposit, or a confirmed acquisition route. Provider contact page: https://materialsmine.org/nm/contact .

Subject: Reproducible access to the NanoMine input for a material-similarity study

Dear MaterialsMine / NanoMine data stewards,

We are preparing a Journal of Chemical Information and Modeling manuscript, “Material Similarity from Incomplete Composition and Synthesis Metadata.” The study compares experimental materials by documented composition, constituents and synthesis metadata, using fixed similarity profiles and unsupervised organization. It uses no response-property targets and does not impute settings.

The current computations use an author-held normalized NanoMine input of 832 records in 119 article groups, together with retained public-API cache exports used in normalization. The normalized file is `nanomine_phase3_records.jsonl.gz` (138,028 bytes; SHA-256 `209168324fcbbcf9d308a2d6358d0caacf7a885b19b2c8e3de951bbff704833a`). Its normalization timestamp is 2026-09-10T00:59:08.885129+00:00. We can provide the complete source-ID list, per-record hashes, original API request parameters and proposed file inventory for checking. Display subsets contain 183, 72 and 120 records, but numerical scale fitting requires all 832 records.

Could you identify the applicable data-use terms and a permitted route for independent reproduction? Specifically, please clarify:

1. Whether the normalized input and required process/constituent metadata may be supplied to anonymous journal reviewers, and through which channel.
2. Whether this exact input may be deposited with a durable identifier for readers after publication, under which license, attribution and any restrictions. If only confidential review is permitted, what reproducible access route would remain after publication?
3. If redistribution is unavailable, whether an official versioned download or documented extraction can supply the same 832 records and settings used for scale fitting. Our search and SPARQL checks returned HTTP 503 on 15 September 2026.
4. Whether original API caches or detailed value-audit extracts need separate handling, and which provider/source citations should accompany the normalized data.

The MaterialsMine software license distinguishes API data from code, so we have not assumed that its license grants data redistribution rights. We will preserve source-specific terms and document any mismatch between newly obtained data and the frozen input before claiming an identical rerun.

Thank you for helping us establish a reproducible route consistent with the applicable terms.

Hiromasa Kaneko

## Follow-up record

After an actual reply, record the applicable terms or permission text, permitted files, recipient/access scope, publication access, citation requirements, persistent location/version, and the independently tested retrieval result. Do not mark this review item complete on the basis of this unsent draft.
