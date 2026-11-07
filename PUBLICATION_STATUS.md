# Publication status and file integrity

## Verified initial public baseline

The research code is public in the [DCEKit interaction descriptors directory](https://github.com/hkaneko1985/dcekit/tree/master/interaction_descriptors).
The baseline checked on 7 November 2026 is commit
`0d8d31e9d061208f85d3ef4d8de9dc587c43ca89`, committed on
7 November 2026 at 02:31:32 UTC.

- [Immutable directory](https://github.com/hkaneko1985/dcekit/tree/0d8d31e9d061208f85d3ef4d8de9dc587c43ca89/interaction_descriptors)
- [Commit](https://github.com/hkaneko1985/dcekit/commit/0d8d31e9d061208f85d3ef4d8de9dc587c43ca89)
- [Machine-readable publication audit](verification/publication_audit54.json)

The official Git tree contains all 143 files from the reviewed v53 archive,
with no missing or additional files. Of these, 119 are byte-identical,
including all 40 Python files. The remaining 24 are CSV files whose only
change is CRLF-to-LF line-ending normalization. Every resulting Git blob
identity and file size was checked against the pinned public tree.
The README, checksum manifest and main comparison summary were also downloaded
directly from their commit-pinned public raw URLs and matched those identities.

## File checksum correction

The first public commit retained the archive checksum manifest, which describes
the original CRLF CSV bytes. Its 24 affected entries consequently do not match
the LF bytes stored by Git. This is a packaging metadata discrepancy; no numeric
value, source code, fitted model or scientific result changed.

This correction adds an LF policy in `.gitattributes`, records the public
baseline, and regenerates `MANIFEST_SHA256.json` from the complete candidate
tree. The manifest covers every payload file and excludes itself. It describes
canonical LF files, so an existing checkout whose text files were already
converted to CRLF may need to be refreshed before local byte-level checking.

Historical verification JSON files are retained unchanged. A historical input
hash refers to the bytes used in that earlier check, and can legitimately
differ from the hash of an otherwise identical LF public CSV. For example,
the 144-cell comparison summary has these identities:

| Input bytes | SHA256 |
|---|---|
| Historical CRLF summary | `9ee56ad0a7a451fc0c05ed249581c24c49099bc28217149d2d1bb791726e8761` |
| Public LF summary | `35b39d5fca317a6209e3a70e86c4fb910160f9fce0d0231c7aec9bd9feb0a809` |

The correction was prepared locally after the public-baseline audit. That
stage54 audit did not write to GitHub. A subsequent stage55 read-only check
confirmed publication of the complete correction in commit
`46f59b074bcaa2994070ad88c9a77253930da2fe`, committed on 7 November 2026 at
03:19:04 UTC. All 146 public Git blob identities match the reviewed v54
candidate, with no missing, additional or changed files.

- [Verified metadata-correction baseline](https://github.com/hkaneko1985/dcekit/tree/46f59b074bcaa2994070ad88c9a77253930da2fe/interaction_descriptors)

This is the baseline for the subsequent Pillow dependency update. The initial
baseline link above remains an immutable reference for the first release.

## Stage54 verification scope

A working tree was reconstructed from the reviewed archive using exactly the
24 line-ending changes, and all 143 resulting blob identities were matched
to the public Git tree. This was not a full raw download or a git clone.
From that tree, the `tables`, `regressor-tables`, and `plot-comparison`
commands were executed to check the specific line-ending risk:

- All 17 table CSV exports are byte-identical to the verified v53 outputs.
- The Figure 8 PNG and comparison CSV are byte-identical to those outputs.
- Figure 8 analysis and metadata JSON files retain all numerical and other
  fields; only `summary_sha256` changes to the public LF input hash.
- No source model or target regressor was fitted in this check.

The prior 16 core tests, 6 diagnostic tests and 20 command checks remain
documented in the stage53 records. They were not repeated for this metadata
correction because all 40 Python files retain their verified v53 bytes.
The full source bank and all historical experiments have not been retrained
in this check. External data and trained weights remain subject to the
requirements documented in `data/README.md`, `models/README.md` and `PIPELINE.md`.

## Stage55 dependency security update

The local revision documented in [SECURITY_UPDATE.md](SECURITY_UPDATE.md)
updates the optional replay and plot environments to Pillow 13.0.0. The replay
preflight is synchronized with the new requirement; original scientific
routines, input identities, public CSVs and historical verification files are
preserved. New, separately scoped compatibility results are recorded in
[verification/security_update55.json](verification/security_update55.json).

The local stage55 review did not publish this dependency update or inspect
authenticated repository alerts. It therefore does not assign a public commit
to the update or attest that the Dependabot alerts have closed. Record the
actual update commit and alert resolution after publication.
