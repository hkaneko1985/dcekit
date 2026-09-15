#!/usr/bin/env python3
"""Minimal example of the unified Phase 5 API."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from material_similarity import (
    DEFAULT_VIEWS,
    DocumentedValue as V,
    MaterialInstance,
    MaterialSimilarityEngine,
    ProcessStep,
    ReportingStatus,
)


records = [
    MaterialInstance(
        record_id="sample-a",
        composition=V.reported({"Al": 2, "O": 3}, kind="composition"),
        process_method=V.reported(["solution processing"], kind="set"),
        process_sequence_status=ReportingStatus.REPORTED,
        process_steps=(
            ProcessStep("mixing", {"time": V.reported(20, kind="numeric", unit_key="time_min")}),
            ProcessStep("heating", {"temperature": V.reported(120, kind="numeric", unit_key="temperature_c")}),
        ),
    ),
    MaterialInstance(
        record_id="sample-b",
        composition=V.reported({"Al": 2, "O": 3}, kind="composition"),
        process_method=V.reported(["solution processing"], kind="set"),
        process_sequence_status=ReportingStatus.REPORTED,
        process_steps=(
            ProcessStep("mixing", {"time": V.unknown(kind="numeric", unit_key="time_min")}),
            ProcessStep("heating", {"temperature": V.reported(130, kind="numeric", unit_key="temperature_c")}),
        ),
    ),
]

engine = MaterialSimilarityEngine(records)
comparison = engine.compare("sample-a", "sample-b", DEFAULT_VIEWS["balanced_instance"])
print(comparison.as_dict())
