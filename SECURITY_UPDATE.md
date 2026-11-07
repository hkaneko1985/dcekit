# Pillow dependency security update — 7 November 2026

## Reason for the update

The published replay requirements pinned Pillow 12.3.0. Official Pillow and
GitHub advisories identify that version as affected by two Critical
vulnerabilities, with Pillow 13.0.0 as the first patched release:

| Advisory | Affected versions in the GitHub Advisory Database | First patched version |
|---|---|---|
| [GHSA-v9pv-5gc8-w2m6 / CVE-2026-96639](https://github.com/advisories/GHSA-v9pv-5gc8-w2m6) | >=11.0.0, <13.0.0 | 13.0.0 |
| [GHSA-5x8w-8jj5-q52h / CVE-2026-112398](https://github.com/advisories/GHSA-5x8w-8jj5-q52h) | <=12.3.0 | 13.0.0 |

The notification supplied for this review listed `pillow` and `Pillow` against
`interaction_descriptors/requirements-replay.txt`, but did not include advisory
IDs. The two public advisories above are strong matching candidates; their
identity with the repository's actual alerts has not been authenticated.
[Python package-name normalization](https://packaging.python.org/en/latest/specifications/name-normalization/)
treats the two capitalizations as the same distribution. Changing capitalization
does not remediate a vulnerable version.

The notification concerns declared vulnerable dependencies. It is not, by
itself, evidence that an intrusion or data disclosure has occurred.

## Changes in this revision

- Pin Pillow 13.0.0 in `requirements-replay.txt` and `requirements-plot.txt`.
- Synchronize `scripts/replay_v49.py`'s strict expected-version check with the
  replay requirement. Updating the requirement alone would fail preflight.
- Update the current installation guidance and runner provenance. Bootstrap
  reporting now distinguishes a maintained runner from the original v49 file,
  and separately checks that the supplied runner is unchanged during execution.
- Retain the original numerical routines, all seven other replay pins, input
  identities, public CSVs and historical verification records.
- Record this new validation separately and regenerate the current file manifest.

RDKit 2025.9.4 requires Pillow, and Matplotlib 3.10.8 also depends on Pillow.
Deleting the direct Pillow line would therefore leave transitive dependencies.
The inspected replay uses RDKit for periodic-table values; its public numerical
code does not directly decode external images. Plotting uses Matplotlib's image
output. This limited code-path assessment does not prove every dependency path
unreachable and is not a reason to retain a known affected version.

The core package remains Python >=3.10 plus NumPy, without Pillow. The maintained
steel replay retains Python 3.13 and exact numerical-package versions. Pillow
13.0.0 itself requires Python >=3.11; the plotting requirements reflect that
minimum. [PyPI release metadata](https://pypi.org/project/pillow/13.0.0/) and the
[Pillow release notes](https://pillow.readthedocs.io/en/stable/releasenotes/13.0.0.html)
provide the upstream release and compatibility information.

## Completed compatibility checks

The checks ran with Python 3.13.5 in separate copies of recorded isolated virtual
environments. Only Pillow was upgraded in the replay copy, from 12.3.0 to 13.0.0.
The original replay environment was retained and its versions were checked.
Both checked environments passed `pip check`. The replay preflight confirmed
the exact versions, environment isolation and package resolution inside the
selected environment.

| Check | Result |
|---|---|
| Replay source comparison | All executable AST nodes except the Pillow version value are unchanged; input hashes and numerical block are unchanged |
| Frozen feature regeneration | 12 latent arrays and the fixed 18 descriptors match their recorded references exactly |
| Historical metric reaggregation | 126 metrics match exactly; this reads stored predictions and does not refit all historical jobs |
| Representative target refitting | 6 configuration-repeat jobs, 18 outer fits, 1,872 predictions; maximum absolute prediction difference 0 |
| Figure 7 redraw | PNG and 36-row plotting CSV match the verified stage52 redraw byte for byte |
| Figure 8 redraw | PNG and 18-row comparison CSV match the stage54 redraw byte for byte |
| Scientific payload integrity | Core source, public CSVs and all pre-existing verification files are unchanged |

The copied plotting environment already had Pillow 13.0.0. Its redraw checks
validate the newly explicit version pin; they do not measure rendering changes
between Pillow 12.3.0 and 13.0.0. Figure 7 is compared with the prior verified
redraw, not with an unrecovered original plotting script.

This was **not a fresh installation of all dependencies**. The current bootstrap
workflow was reviewed but not executed from an empty environment in this check.
The full 540-fit steel replay, complete source-bank retraining, 500-job regressor
extension and historical Figures 1–6 were not repeated. No new material rows or
performance conditions were added. Successful compatibility checks are not a
complete vulnerability audit.

Machine-readable details, versions and comparison hashes are in
[verification/security_update55.json](verification/security_update55.json).
The original stage51 evidence still records its actual Pillow 12.3.0 environment;
it must not be relabelled as a run performed with Pillow 13.0.0.

## Applying the update

The verified public baseline is commit
`46f59b074bcaa2994070ad88c9a77253930da2fe`; all 146 files matched the reviewed
metadata-corrected v54 directory. The stage55 patch applies to that directory.
The full archive contains the equivalent updated directory. Extract the chosen
archive and preserve the `interaction_descriptors/` paths in the DCEKit repository.
The ZIP container itself is not the source-file update.

For an existing replay or plotting installation, updating GitHub files alone
does not update installed packages. Use the current requirements in the selected
virtual environment. For the Python 3.13 replay environment, from this directory:

```bash
python -m pip install -r requirements-replay.txt
python -m pip check
python -c "from importlib.metadata import version; print(version('Pillow'))"
```

Install `requirements-plot.txt` in the chosen plotting environment if plotting
is needed. Preserve historical environments as historical evidence and use a
separate maintained environment for new work. Detailed replay inputs and commands
remain in [REPLAY.md](REPLAY.md).

After committing the source-file update to the default branch, inspect the
repository's [Dependabot alerts](https://github.com/hkaneko1985/dcekit/security/dependabot).
Match the actual advisory IDs and detected manifest/version, then verify that
the relevant alerts are resolved as fixed. A manual dismissal is a different
state from remediation. The [GitHub alert-management guide](https://docs.github.com/en/code-security/how-tos/manage-security-alerts/manage-dependabot-alerts/view-dependabot-alerts)
describes the review and update workflow.

The local review did not write to GitHub, send a notification reply, inspect
authenticated alert details, or verify alert closure. Record the actual security
update commit and the observed resolution separately after publication.
