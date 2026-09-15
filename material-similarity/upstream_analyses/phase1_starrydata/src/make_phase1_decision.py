#!/usr/bin/env python3
"""Create a compact, machine-readable Phase 1 decision record."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    primary = json.loads(args.primary.read_text(encoding="utf-8"))
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    gates = primary["phase2_gate_results"]
    failed = [name for name, passed in gates.items() if not passed]

    decision = {
        "decision_version": "1.0.0",
        "strict_preregistered_decision": primary["decision_for_phase2"],
        "operational_decision": "PROCEED_TO_PHASE2_WITH_CAVEATS",
        "passed_gates": int(sum(bool(value) for value in gates.values())),
        "total_gates": len(gates),
        "failed_gates": failed,
        "primary_evidence": {
            "samples": primary["pilot_selection"]["selected_samples"],
            "confirmation_same_sid_pairs": primary["known_positive_evaluation"]["confirmation_same_sid_pairs"],
            "F0_lift": primary["evaluation"]["F0"]["selected_resolution"]["lift"],
            "F2_lift": primary["evaluation"]["F2"]["selected_resolution"]["lift"],
            "F2_minus_F0_delta_log_lift": primary["incremental_comparisons"]["F2_minus_F0"],
            "F2_median_subsample_ari": primary["evaluation"]["F2"]["stability"]["median_ari"],
            "empirical_permutation_pvalue": primary["evaluation"]["F2"]["composition_stratified_permutation_pvalue"],
        },
        "caveats": {
            "single_paper_increment_share": primary["falsification_controls"]["paper_dominance_F2_minus_F0"],
            "preregistered_threshold": 0.10,
            "observed_across_three_seeds": summary["paper_dominance"],
            "missingness_only_lift": summary["missingness_only_lift"],
            "cluster_resolution": summary["resolution_sensitivity"]["interpretation"],
            "same_sid_role": "Weak positive support only; different-SID pairs are unlabeled, not negatives.",
        },
        "phase2_requirements": [
            "Use HTEM sample-library/study identity as an external weak-positive context.",
            "Freeze F0/F1/F2 definitions before inspecting Phase 2 support labels.",
            "Keep target variables, measured properties, source IDs, and free text out of similarity features.",
            "Repeat complete-case, missingness-only, process-shuffle, order/tie, and multiresolution controls.",
            "Treat failure to replicate as rejection or revision of this representation, not as evidence that meaningful material similarity does not exist.",
        ],
        "claim_boundary": primary["claim_boundary"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "strict": decision["strict_preregistered_decision"],
        "operational": decision["operational_decision"],
        "gates": f"{decision['passed_gates']}/{decision['total_gates']}",
        "failed": failed,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
