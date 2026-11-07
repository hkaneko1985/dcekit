"""C11 composition alignment without deleting present components.

This module checks declared input-vocabulary compatibility only. It does not
establish observed pair support, ratio coverage, physical applicability, or
prediction quality. Source preprocessing must be applied *after* this mapping,
using the original frozen parameters. No model weights are read or modified.

Fractions must be finite, nonnegative and sum to one within the stated numerical
tolerance. Values are neither clipped nor normalized. Any unsupported fraction
strictly greater than zero blocks the corresponding source input, even if tiny.
Atomic/mass conversion is deliberately not implicit in a column mapping.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import fsum, isfinite
from typing import Iterable, Sequence


FRACTION_BASES = frozenset({"atomic_fraction", "mass_fraction"})


class CompositionInputError(ValueError):
    """The composition or its declared schema cannot be used as supplied."""


def _names(values: Sequence[str], label: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise CompositionInputError(f"{label} must be a sequence of component names")
    names = tuple(values)
    if not names:
        raise CompositionInputError(f"{label} must not be empty")
    if any(not isinstance(s, str) or not s or s.strip() != s for s in names):
        raise CompositionInputError(f"{label} needs exact, nonempty component names")
    if len(set(names)) != len(names):
        raise CompositionInputError(f"{label} contains duplicate components")
    return names


def common_component_schema(*schemas: Sequence[str]) -> tuple[str, ...]:
    """Build an ordered union once, then reuse that fixed schema for all datasets.

    This helper does not rename species, infer chemical identity, or sort source
    model columns. Callers must retain the returned order as explicit metadata.
    """
    if not schemas:
        raise CompositionInputError("At least one component schema is required")
    union: dict[str, None] = {}
    for i, schema in enumerate(schemas):
        union.update(dict.fromkeys(_names(schema, f"schema[{i}]")))
    return tuple(union)


@dataclass(frozen=True)
class AlignmentResult:
    common_components: tuple[str, ...]
    source_components: tuple[str, ...]
    fraction_basis: str
    common_fractions: tuple[tuple[float, ...], ...]
    supported_mask: tuple[bool, ...]
    source_inputs: tuple[tuple[float, ...] | None, ...]
    absent_components: tuple[tuple[str, ...], ...]
    unsupported_present: tuple[tuple[str, ...], ...]
    unsupported_fractions: tuple[tuple[float, ...], ...]
    input_sums: tuple[float, ...]
    sum_tolerance: float
    support_definition: str = "declared_input_vocabulary_compatibility_only"
    empirical_support_audited: bool = False
    automatically_normalized: bool = False

    def to_dict(self) -> dict:
        """Return JSON-serializable values; unsupported source rows stay None."""
        return asdict(self)


def audit_and_align(
    rows: Iterable[Sequence[float]],
    *,
    input_components: Sequence[str],
    common_components: Sequence[str],
    source_components: Sequence[str],
    input_basis: str,
    source_basis: str,
    sum_tolerance: float = 1e-8,
) -> AlignmentResult:
    """Preserve all fractions and map compatible rows into saved source order.

    ``common_components`` must contain every named input and source column.
    Missing input columns in this union are genuine absent-component zeros.
    A source row is returned only when every positive component belongs to the
    source vocabulary. Otherwise it is None, and the unsupported names/fractions
    are returned alongside the unchanged complete composition.

    The mask is independent of sum_tolerance: that tolerance concerns floating
    point summation only, never a component-presence cutoff. An unsupported
    present component is not discarded or renormalized away.
    """
    input_names = _names(input_components, "input_components")
    common_names = _names(common_components, "common_components")
    source_names = _names(source_components, "source_components")
    if input_basis not in FRACTION_BASES or source_basis not in FRACTION_BASES:
        raise CompositionInputError("Use atomic_fraction or mass_fraction explicitly")
    if input_basis != source_basis:
        raise CompositionInputError(
            "Fraction bases differ; perform and document the physical conversion first"
        )
    if not isfinite(sum_tolerance) or not 0 <= sum_tolerance < 1:
        raise CompositionInputError("sum_tolerance must be finite and in [0, 1)")
    common_index = {name: i for i, name in enumerate(common_names)}
    missing = [n for n in (*input_names, *source_names) if n not in common_index]
    if missing:
        raise CompositionInputError(f"Common schema omits named components: {sorted(set(missing))}")
    source_set = set(source_names)
    complete_rows, mapped_rows, supported = [], [], []
    absent, unsupported, unsupported_values, row_sums = [], [], [], []
    for row_index, row in enumerate(rows):
        try:
            values = tuple(float(v) for v in row)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CompositionInputError(f"Row {row_index}: nonnumeric fraction") from exc
        if len(values) != len(input_names):
            raise CompositionInputError(f"Row {row_index}: fraction/column count mismatch")
        if any(not isfinite(v) or v < 0 for v in values):
            raise CompositionInputError(f"Row {row_index}: fractions must be finite and nonnegative")
        try:
            total = fsum(values)
        except OverflowError as exc:
            raise CompositionInputError(f"Row {row_index}: fraction sum overflows") from exc
        if abs(total - 1.0) > sum_tolerance:
            raise CompositionInputError(f"Row {row_index}: fractions sum to {total!r}, expected 1")
        full = [0.0] * len(common_names)
        for name, value in zip(input_names, values):
            full[common_index[name]] = value
        bad = tuple(n for n, v in zip(common_names, full) if v > 0 and n not in source_set)
        ok = not bad
        complete_rows.append(tuple(full))
        supported.append(ok)
        mapped_rows.append(tuple(full[common_index[n]] for n in source_names) if ok else None)
        absent.append(tuple(n for n, v in zip(common_names, full) if v == 0))
        unsupported.append(bad)
        unsupported_values.append(tuple(full[common_index[n]] for n in bad))
        row_sums.append(total)
    if not complete_rows:
        raise CompositionInputError("No composition rows supplied")
    return AlignmentResult(
        common_components=common_names,
        source_components=source_names,
        fraction_basis=input_basis,
        common_fractions=tuple(complete_rows),
        supported_mask=tuple(supported),
        source_inputs=tuple(mapped_rows),
        absent_components=tuple(absent),
        unsupported_present=tuple(unsupported),
        unsupported_fractions=tuple(unsupported_values),
        input_sums=tuple(row_sums),
        sum_tolerance=sum_tolerance,
    )
