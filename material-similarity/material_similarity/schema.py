"""Typed, property-free records for literature-derived material instances.

The schema deliberately uses reporting language rather than observation
language.  Literature metadata cannot generally distinguish an unset process
condition from an unreported one, while a process step that is known not to
exist is a different state again.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Mapping, Sequence


class ReportingStatus(str, Enum):
    """Status of a value in the source record."""

    REPORTED = "reported"
    UNREPORTED_OR_UNSET = "unreported_or_unset"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True)
class DocumentedValue:
    """A value together with its literature-reporting status.

    ``unit_key`` identifies already-harmonized units.  The similarity engine
    never converts or imputes a missing value; unit conversion belongs in the
    source adapter.
    """

    status: ReportingStatus
    value: Any = None
    kind: str = "categorical"
    unit_key: str | None = None

    def __post_init__(self) -> None:
        if self.status is ReportingStatus.REPORTED and self.value is None:
            raise ValueError("A reported value must contain a value")
        if self.status is not ReportingStatus.REPORTED and self.value is not None:
            raise ValueError("Unreported/unset and not-applicable values must be None")
        if self.kind not in {"categorical", "numeric", "set", "composition", "numeric_multiset", "categorical_multiset"}:
            raise ValueError(f"Unsupported value kind: {self.kind}")
        if self.kind == "numeric" and self.status is ReportingStatus.REPORTED:
            number = float(self.value)
            if not isfinite(number):
                raise ValueError("A reported numeric value must be finite")
        if self.kind.endswith('_multiset') and self.status is ReportingStatus.REPORTED:
            if not isinstance(self.value, (tuple, list)) or not self.value:
                raise ValueError('A reported multiset requires a nonempty list or tuple')
            if self.kind == 'numeric_multiset' and not all(isfinite(float(x)) for x in self.value):
                raise ValueError('All reported multiset numbers must be finite')

    @classmethod
    def reported(
        cls,
        value: Any,
        *,
        kind: str = "categorical",
        unit_key: str | None = None,
    ) -> "DocumentedValue":
        return cls(ReportingStatus.REPORTED, value, kind, unit_key)

    @classmethod
    def unknown(cls, *, kind: str = "categorical", unit_key: str | None = None) -> "DocumentedValue":
        return cls(ReportingStatus.UNREPORTED_OR_UNSET, None, kind, unit_key)

    @classmethod
    def not_applicable(
        cls, *, kind: str = "categorical", unit_key: str | None = None
    ) -> "DocumentedValue":
        return cls(ReportingStatus.NOT_APPLICABLE, None, kind, unit_key)


@dataclass(frozen=True)
class ProcessStep:
    """One ordered process step; repeated steps remain repeated objects."""

    step_type: str
    settings: Mapping[str, DocumentedValue] = field(default_factory=dict)
    source_id: str | None = field(default=None, compare=False)
    setting_provenance: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not str(self.step_type).strip():
            raise ValueError("ProcessStep.step_type cannot be empty")


@dataclass(frozen=True)
class MaterialInstance:
    """A target-property-free material instance.

    ``context`` may hold paper/study/author/institution identifiers for
    post-hoc support checks.  It is never read by the distance engine.
    """

    record_id: str
    composition: DocumentedValue = field(
        default_factory=lambda: DocumentedValue.unknown(kind="composition")
    )
    material_identity: Mapping[str, DocumentedValue] = field(default_factory=dict)
    process_method: DocumentedValue = field(
        default_factory=lambda: DocumentedValue.unknown(kind="set")
    )
    process_settings: Mapping[str, DocumentedValue] = field(default_factory=dict)
    process_steps: tuple[ProcessStep, ...] = field(default_factory=tuple)
    process_sequence_status: ReportingStatus = ReportingStatus.UNREPORTED_OR_UNSET
    context: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    def __post_init__(self) -> None:
        if not str(self.record_id).strip():
            raise ValueError("MaterialInstance.record_id cannot be empty")
        if self.composition.kind != "composition":
            raise ValueError("composition must have kind='composition'")
        if self.process_method.kind not in {"categorical", "set"}:
            raise ValueError("process_method must be categorical or set-valued")
        if self.process_sequence_status is not ReportingStatus.REPORTED and self.process_steps:
            raise ValueError("process_steps require process_sequence_status='reported'")

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "MaterialInstance":
        """Build a record from the package's JSON-compatible interchange form."""

        def parse_value(raw: Mapping[str, Any] | None, *, default_kind: str) -> DocumentedValue:
            if raw is None:
                return DocumentedValue.unknown(kind=default_kind)
            status = ReportingStatus(raw.get("status", ReportingStatus.REPORTED.value))
            return DocumentedValue(
                status=status,
                value=raw.get("value"),
                kind=raw.get("kind", default_kind),
                unit_key=raw.get("unit_key"),
            )

        raw_steps = row.get("process_steps", [])
        steps = []
        for step in raw_steps:
            settings = {
                str(key): parse_value(value, default_kind=value.get("kind", "categorical"))
                for key, value in step.get("settings", {}).items()
            }
            steps.append(ProcessStep(str(step["step_type"]), settings, step.get('source_id'), step.get('setting_provenance', {})))

        identity = {
            str(key): parse_value(value, default_kind=value.get("kind", "categorical"))
            for key, value in row.get("material_identity", {}).items()
        }
        return cls(
            record_id=str(row["record_id"]),
            composition=parse_value(row.get("composition"), default_kind="composition"),
            material_identity=identity,
            process_method=parse_value(row.get("process_method"), default_kind="set"),
            process_settings={
                str(key): parse_value(value, default_kind=value.get("kind", "categorical"))
                for key, value in row.get("process_settings", {}).items()
            },
            process_steps=tuple(steps),
            process_sequence_status=ReportingStatus(
                row.get("process_sequence_status", ReportingStatus.UNREPORTED_OR_UNSET.value)
            ),
            context=dict(row.get("context", {})),
        )

    def to_mapping(self) -> dict[str, Any]:
        """Return the JSON-compatible interchange form."""

        def dump_value(value: DocumentedValue) -> dict[str, Any]:
            return {
                "status": value.status.value,
                "value": value.value,
                "kind": value.kind,
                "unit_key": value.unit_key,
            }

        return {
            "record_id": self.record_id,
            "composition": dump_value(self.composition),
            "material_identity": {
                key: dump_value(value) for key, value in self.material_identity.items()
            },
            "process_method": dump_value(self.process_method),
            "process_settings": {
                key: dump_value(value) for key, value in self.process_settings.items()
            },
            "process_sequence_status": self.process_sequence_status.value,
            "process_steps": [
                {
                    "step_type": step.step_type,
                    "source_id": step.source_id,
                    "setting_provenance": dict(step.setting_provenance),
                    "settings": {
                        key: dump_value(value) for key, value in step.settings.items()
                    },
                }
                for step in self.process_steps
            ],
            "context": dict(self.context),
        }


def ensure_unique_ids(records: Sequence[MaterialInstance]) -> None:
    ids = [record.record_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("MaterialInstance.record_id values must be unique")
