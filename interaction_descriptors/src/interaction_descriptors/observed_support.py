"""Distinguish source vocabulary from positively observed source components.

This audits supplied source training compositions. A positive observation does
not establish learned physical effects, pair support, ratio coverage or useful
predictions. Neither source nor target values are changed by this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import fsum, isfinite
from typing import Iterable, Sequence

from .composition import (
    AlignmentResult, CompositionInputError, FRACTION_BASES, _names, audit_and_align,
)


@dataclass(frozen=True)
class SourceObservations:
    components: tuple[str, ...]
    fraction_basis: str
    n_source_rows: int
    positive_row_counts: tuple[int, ...]
    maximum_sum_error: float
    sum_tolerance: float

    @property
    def observed_components(self) -> tuple[str, ...]:
        return tuple(n for n, k in zip(self.components, self.positive_row_counts) if k > 0)

    @property
    def unobserved_components(self) -> tuple[str, ...]:
        return tuple(n for n, k in zip(self.components, self.positive_row_counts) if k == 0)


def inspect_source_observations(
    rows: Iterable[Sequence[float]], *, source_components: Sequence[str],
    source_basis: str, sum_tolerance: float = 1e-8,
) -> SourceObservations:
    """Validate actual source rows and count strictly positive observations."""
    names = _names(source_components, "source_components")
    if source_basis not in FRACTION_BASES:
        raise CompositionInputError("Source basis must be explicit")
    if not isfinite(sum_tolerance) or not 0 <= sum_tolerance < 1:
        raise CompositionInputError("sum_tolerance must be finite and in [0, 1)")
    counts = [0] * len(names)
    n_rows, max_error = 0, 0.0
    for i, row in enumerate(rows):
        try:
            vals = tuple(float(v) for v in row)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CompositionInputError(f"Source row {i}: nonnumeric fraction") from exc
        if len(vals) != len(names):
            raise CompositionInputError(f"Source row {i}: fraction/column count mismatch")
        if any(not isfinite(v) or v < 0 for v in vals):
            raise CompositionInputError(f"Source row {i}: fractions must be finite and nonnegative")
        try:
            error = abs(fsum(vals) - 1.0)
        except OverflowError as exc:
            raise CompositionInputError(f"Source row {i}: fraction sum overflows") from exc
        if error > sum_tolerance:
            raise CompositionInputError(f"Source row {i}: fractions do not sum to one")
        for j, v in enumerate(vals):
            counts[j] += int(v > 0)
        max_error = max(max_error, error)
        n_rows += 1
    if not n_rows:
        raise CompositionInputError("No source composition rows supplied")
    return SourceObservations(names, source_basis, n_rows, tuple(counts), max_error, sum_tolerance)


@dataclass(frozen=True)
class ObservedAlignmentResult:
    vocabulary_alignment: AlignmentResult
    source_observations: SourceObservations
    observed_supported_mask: tuple[bool, ...]
    unobserved_present: tuple[tuple[str, ...], ...]
    observed_source_inputs: tuple[tuple[float, ...] | None, ...]
    support_definition: str = "vocabulary_and_positive_observation_in_supplied_source_rows_only"

    def to_dict(self) -> dict:
        return asdict(self)


def audit_with_source_observations(
    rows: Iterable[Sequence[float]], *, input_components: Sequence[str],
    common_components: Sequence[str], input_basis: str,
    source_observations: SourceObservations, sum_tolerance: float = 1e-8,
) -> ObservedAlignmentResult:
    """Return distinct vocabulary and observed-component masks without projection.

    A positive target component in an all-zero source column invalidates the
    observation mask even though it passes the declared-vocabulary mask. The
    preserved full target vector is available in vocabulary_alignment.
    """
    a = audit_and_align(
        rows, input_components=input_components, common_components=common_components,
        source_components=source_observations.components, input_basis=input_basis,
        source_basis=source_observations.fraction_basis, sum_tolerance=sum_tolerance,
    )
    unseen = set(source_observations.unobserved_components)
    bad = tuple(tuple(n for n, v in zip(a.common_components, row) if v > 0 and n in unseen)
                for row in a.common_fractions)
    mask = tuple(ok and not missed for ok, missed in zip(a.supported_mask, bad))
    mapped = tuple(row if ok else None for row, ok in zip(a.source_inputs, mask))
    return ObservedAlignmentResult(a, source_observations, mask, bad, mapped)
