"""Missingness-aware multi-view distances for material instances."""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from functools import lru_cache
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment

from .schema import (
    DocumentedValue,
    MaterialInstance,
    ProcessStep,
    ReportingStatus,
    ensure_unique_ids,
)


FACET_NAMES = (
    "composition",
    "material_identity",
    "process_method",
    "step_type",
    "sequence",
    "settings",
)


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("μ", "u").replace("µ", "u")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def process_step_keys(record: MaterialInstance):
    """Yield (absolute index, canonical occurrence key) in recorded order.

    The distance engine and masking experiments share this correspondence.
    Occurrences are counted within each normalized operation type, from zero.
    """
    counts: Counter[str] = Counter()
    for index, step in enumerate(record.process_steps):
        token = normalize_text(step.step_type)
        yield index, f"{token}#{counts[token]}"
        counts[token] += 1


def step_setting_locations(record: MaterialInstance):
    """Map canonical setting keys to their original step/field locations."""
    locations = defaultdict(list)
    for index, step_key in process_step_keys(record):
        for field in record.process_steps[index].settings:
            locations[f"{step_key}:{normalize_text(field)}"].append((index, field))
    return dict(locations)


def _as_set(value: Any) -> set[str]:
    if isinstance(value, str):
        return {normalize_text(value)} if normalize_text(value) else set()
    if isinstance(value, Iterable) and not isinstance(value, Mapping):
        return {normalized for item in value if (normalized := normalize_text(item))}
    normalized = normalize_text(value)
    return {normalized} if normalized else set()


def jaccard_distance(left: Iterable[str], right: Iterable[str]) -> float:
    a, b = set(left), set(right)
    if not a and not b:
        return 0.0
    return 1.0 - len(a & b) / len(a | b)


@lru_cache(maxsize=32768)
def _char_ngrams(value: Any, n: int = 3) -> set[str]:
    text = f"  {normalize_text(value)}  "
    return {text[index : index + n] for index in range(max(1, len(text) - n + 1))}


def lexical_distance(left: Any, right: Any) -> float:
    return jaccard_distance(_char_ngrams(left), _char_ngrams(right))


def soft_set_distance(left: Iterable[Any], right: Iterable[Any]) -> float:
    """Symmetric soft set distance using optimal assignment."""

    a, b = sorted(_as_set(left)), sorted(_as_set(right))
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    short, long = (a, b) if len(a) <= len(b) else (b, a)
    costs = np.asarray([[lexical_distance(x, y) for y in long] for x in short], dtype=float)
    rows, columns = linear_sum_assignment(costs)
    matched = float(costs[rows, columns].sum())
    return float((matched + len(long) - len(short)) / len(long))


def normalized_sequence_distance(left: Sequence[str], right: Sequence[str]) -> float:
    """Normalized edit distance; order and repetition are both retained."""

    if not left and not right:
        return 0.0
    if not left or not right:
        return 1.0
    previous = np.arange(len(right) + 1, dtype=float)
    for i, token_left in enumerate(left, start=1):
        current = np.empty(len(right) + 1, dtype=float)
        current[0] = i
        for j, token_right in enumerate(right, start=1):
            substitution = 0.0 if normalize_text(token_left) == normalize_text(token_right) else 1.0
            current[j] = min(
                previous[j] + 1.0,
                current[j - 1] + 1.0,
                previous[j - 1] + substitution,
            )
        previous = current
    return float(previous[-1] / max(len(left), len(right)))


@dataclass(frozen=True)
class NumericScale:
    scale: float
    transform: str = "linear"

    def __post_init__(self) -> None:
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("NumericScale.scale must be positive and finite")
        if self.transform not in {"linear", "log1p_signed"}:
            raise ValueError("Unsupported numeric transform")

    def apply(self, value: float) -> float:
        number = float(value)
        if self.transform == "log1p_signed":
            return math.copysign(math.log1p(abs(number)), number)
        return number


@dataclass(frozen=True)
class DistanceInterval:
    lower: float
    reported: float | None
    upper: float
    common_reported_fraction: float
    one_sided_fraction: float
    structural_difference_fraction: float
    active_weight: float
    excluded_weight: float

    def __post_init__(self) -> None:
        tolerance = 1e-9
        if not (-tolerance <= self.lower <= self.upper + tolerance <= 1.0 + tolerance):
            raise ValueError("Distance bounds must satisfy 0 <= lower <= upper <= 1")
        if self.reported is not None and not (-tolerance <= self.reported <= 1.0 + tolerance):
            raise ValueError("reported distance must be in [0, 1] or None")

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def similarity(self, bound: str = "reported") -> float | None:
        value = self.value(bound)
        return None if value is None else 1.0 - value

    def value(self, bound: str) -> float | None:
        if bound == "optimistic":
            return self.lower
        if bound == "reported":
            return self.reported
        if bound == "pessimistic":
            return self.upper
        raise ValueError("bound must be optimistic, reported, or pessimistic")

    def as_dict(self) -> dict[str, float | None]:
        return {
            "optimistic": self.lower,
            "reported": self.reported,
            "pessimistic": self.upper,
            "uncertainty_width": self.width,
            "common_reported_fraction": self.common_reported_fraction,
            "one_sided_fraction": self.one_sided_fraction,
            "structural_difference_fraction": self.structural_difference_fraction,
            "active_weight": self.active_weight,
            "excluded_weight": self.excluded_weight,
        }


@dataclass(frozen=True)
class _Contribution:
    lower: float
    reported: float | None
    upper: float
    weight: float
    category: str


def _aggregate(contributions: Sequence[_Contribution]) -> DistanceInterval:
    active = [item for item in contributions if item.category != "excluded"]
    excluded_weight = sum(item.weight for item in contributions if item.category == "excluded")
    if not active:
        return DistanceInterval(0.0, None, 1.0, 0.0, 0.0, 0.0, 0.0, excluded_weight)
    active_weight = sum(item.weight for item in active)
    lower = sum(item.lower * item.weight for item in active) / active_weight
    upper = sum(item.upper * item.weight for item in active) / active_weight
    decisive = [item for item in active if item.reported is not None]
    decisive_weight = sum(item.weight for item in decisive)
    reported = (
        sum(float(item.reported) * item.weight for item in decisive) / decisive_weight
        if decisive_weight
        else None
    )
    common = sum(item.weight for item in active if item.category == "common") / active_weight
    one_sided = sum(item.weight for item in active if item.category == "uncertain") / active_weight
    structural = sum(item.weight for item in active if item.category == "structural") / active_weight
    return DistanceInterval(
        float(np.clip(lower, 0.0, 1.0)),
        None if reported is None else float(np.clip(reported, 0.0, 1.0)),
        float(np.clip(upper, 0.0, 1.0)),
        float(common),
        float(one_sided),
        float(structural),
        float(active_weight),
        float(excluded_weight),
    )


def _status_contribution(
    left: DocumentedValue,
    right: DocumentedValue,
    comparator: Callable[[DocumentedValue, DocumentedValue], float],
    *,
    weight: float = 1.0,
    retain_joint_unknown: bool = False,
) -> _Contribution:
    a, b = left.status, right.status
    if a is ReportingStatus.REPORTED and b is ReportingStatus.REPORTED:
        raw = comparator(left, right)
        if isinstance(raw, tuple):
            lower, upper = raw
            return _Contribution(float(lower), float(lower), float(upper), weight, "common")
        distance = float(np.clip(raw, 0.0, 1.0))
        return _Contribution(distance, distance, distance, weight, "common")
    if a is ReportingStatus.NOT_APPLICABLE and b is ReportingStatus.NOT_APPLICABLE:
        # Joint non-applicability is not evidence of similarity.
        return _Contribution(0.0, None, 0.0, weight, "excluded")
    if {a, b} == {ReportingStatus.REPORTED, ReportingStatus.NOT_APPLICABLE}:
        return _Contribution(1.0, 1.0, 1.0, weight, "structural")
    if a is ReportingStatus.UNREPORTED_OR_UNSET and b is ReportingStatus.UNREPORTED_OR_UNSET:
        if retain_joint_unknown:
            return _Contribution(0.0, None, 1.0, weight, "uncertain")
        return _Contribution(0.0, None, 0.0, weight, "excluded")
    # A value missing on at least one side is not guessed.  It contributes the
    # full feasible interval instead.
    return _Contribution(0.0, None, 1.0, weight, "uncertain")


def _normalize_composition(value: Mapping[str, Any]) -> dict[str, float]:
    output = {}
    for key, raw in value.items():
        number = float(raw)
        if not math.isfinite(number) or number < 0:
            raise ValueError("Composition fractions must be finite and non-negative")
        if number > 0:
            output[str(key)] = number
    total = sum(output.values())
    if total <= 0:
        raise ValueError("A reported composition must contain positive mass")
    return {key: number / total for key, number in output.items()}


def _composition_value_distance(left: DocumentedValue, right: DocumentedValue) -> float:
    a = _normalize_composition(left.value)
    b = _normalize_composition(right.value)
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(key, 0.0) - b.get(key, 0.0)) for key in keys)


def _categorical_value_distance(
    left: DocumentedValue,
    right: DocumentedValue,
    *,
    soft: bool,
) -> float:
    if left.kind != right.kind:
        return 1.0
    if left.kind == "set":
        a, b = _as_set(left.value), _as_set(right.value)
        return soft_set_distance(a, b) if soft else jaccard_distance(a, b)
    if left.kind == "categorical":
        return lexical_distance(left.value, right.value) if soft else float(
            normalize_text(left.value) != normalize_text(right.value)
        )
    raise ValueError("categorical comparator received a non-categorical value")


def _numeric_value_distance(
    left: DocumentedValue,
    right: DocumentedValue,
    scales: Mapping[str, NumericScale],
) -> float:
    if left.kind not in {"numeric", "numeric_multiset"} or right.kind not in {"numeric", "numeric_multiset"}:
        return 1.0
    if left.unit_key != right.unit_key:
        return 1.0
    key = left.unit_key or "unitless"
    scale = scales.get(key, NumericScale(1.0))
    a = list(left.value) if left.kind == 'numeric_multiset' else [left.value]
    b = list(right.value) if right.kind == 'numeric_multiset' else [right.value]
    if len(a)==len(b)==1:
        return float(min(1.,abs(scale.apply(float(a[0]))-scale.apply(float(b[0])))/scale.scale))
    costs = np.asarray([[min(1.,abs(scale.apply(float(x))-scale.apply(float(y)))/scale.scale) for y in b] for x in a])
    rows,cols=linear_sum_assignment(costs)
    return float((costs[rows,cols].sum()+abs(len(a)-len(b)))/max(len(a),len(b)))


def _value_distance(
    left: DocumentedValue,
    right: DocumentedValue,
    *,
    scales: Mapping[str, NumericScale],
    soft: bool,
) -> float:
    if left.kind in {"numeric", "numeric_multiset"} and right.kind in {"numeric", "numeric_multiset"}:
        best = _numeric_value_distance(left, right, scales)
        if ('numeric_multiset' in {left.kind,right.kind} and left.unit_key==right.unit_key):
            scale=scales.get(left.unit_key or 'unitless',NumericScale(1.))
            a=left.value if left.kind=='numeric_multiset' else [left.value]
            b=right.value if right.kind=='numeric_multiset' else [right.value]
            cost=np.asarray([[min(1.,abs(scale.apply(float(x))-scale.apply(float(y)))/scale.scale) for y in b] for x in a])
            ii,jj=linear_sum_assignment(-cost)
            worst=float((cost[ii,jj].sum()+abs(len(a)-len(b)))/max(len(a),len(b)))
            # Without source zone identities, assignment uncertainty remains.
            # 'reported' is the unordered minimum-assignment dissimilarity.
            return best,max(best,worst)
        return best
    if left.kind == 'categorical_multiset' or right.kind == 'categorical_multiset':
        if left.kind not in {'categorical','categorical_multiset'} or right.kind not in {'categorical','categorical_multiset'}:
            return 1.0
        a = list(left.value) if left.kind == 'categorical_multiset' else [left.value]
        b = list(right.value) if right.kind == 'categorical_multiset' else [right.value]
        costs=np.asarray([[lexical_distance(x,y) if soft else float(normalize_text(x)!=normalize_text(y)) for y in b] for x in a])
        rows,cols=linear_sum_assignment(costs)
        return float((costs[rows,cols].sum()+abs(len(a)-len(b)))/max(len(a),len(b)))
    if left.kind != right.kind:
        return 1.0
    return _categorical_value_distance(left, right, soft=soft)


@dataclass(frozen=True)
class ViewSpec:
    name: str
    facet_weights: Mapping[str, float]
    soft_identity: bool = False

    def __post_init__(self) -> None:
        unknown = set(self.facet_weights) - set(FACET_NAMES)
        if unknown:
            raise ValueError(f"Unknown facets: {sorted(unknown)}")
        if not self.facet_weights or sum(float(value) for value in self.facet_weights.values()) <= 0:
            raise ValueError("A view must contain at least one positive facet weight")
        if any(not math.isfinite(float(value)) or float(value) < 0 for value in self.facet_weights.values()):
            raise ValueError("Facet weights must be finite and non-negative")


@dataclass(frozen=True)
class PairDistance:
    left_id: str
    right_id: str
    facets: Mapping[str, DistanceInterval]
    composite: DistanceInterval
    view_name: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "left_id": self.left_id,
            "right_id": self.right_id,
            "view": self.view_name,
            "facets": {key: value.as_dict() for key, value in self.facets.items()},
            "composite": self.composite.as_dict(),
        }


@dataclass(frozen=True)
class PairwiseDistanceResult:
    ids: tuple[str, ...]
    view: ViewSpec
    optimistic: np.ndarray
    reported: np.ndarray
    pessimistic: np.ndarray
    common_reported_fraction: np.ndarray
    uncertainty_width: np.ndarray

    def matrix(self, bound: str = "reported", *, undefined: str = "raise") -> np.ndarray:
        if bound == "optimistic":
            matrix = self.optimistic.copy()
        elif bound == "reported":
            matrix = self.reported.copy()
        elif bound == "pessimistic":
            matrix = self.pessimistic.copy()
        else:
            raise ValueError("bound must be optimistic, reported, or pessimistic")
        missing = ~np.isfinite(matrix)
        if missing.any():
            if undefined == "raise":
                raise ValueError(
                    "Reported distance is undefined for at least one pair; choose an explicit "
                    "optimistic or pessimistic bound, or set undefined accordingly"
                )
            if undefined == "optimistic":
                matrix[missing] = self.optimistic[missing]
            elif undefined == "pessimistic":
                matrix[missing] = self.pessimistic[missing]
            else:
                raise ValueError("undefined must be raise, optimistic, or pessimistic")
        np.fill_diagonal(matrix, 0.0)
        return matrix


class MaterialSimilarityEngine:
    """Compute facet intervals, weighted views, matrices, and explanations."""

    def __init__(
        self,
        records: Sequence[MaterialInstance],
        *,
        numeric_scales: Mapping[str, NumericScale] | None = None,
        feature_policy: str = 'active',
        settings_alignment: str = 'occurrence',
        feature_schema: Mapping[str, Sequence[str]] | None = None,
    ) -> None:
        self.records = tuple(records)
        ensure_unique_ids(self.records)
        self.by_id = {record.record_id: record for record in self.records}
        self.numeric_scales = dict(fit_numeric_scales(self.records) if numeric_scales is None else numeric_scales)
        if feature_policy not in {'active','fixed_schema'}:
            raise ValueError('feature_policy must be active or fixed_schema')
        if settings_alignment not in {'occurrence','sequence'}:
            raise ValueError('settings_alignment must be occurrence or sequence')
        self.feature_policy=feature_policy
        self.settings_alignment=settings_alignment
        self.feature_schema = dict(feature_schema) if feature_schema is not None else {
            'identity': sorted({key for r in self.records for key in r.material_identity}),
            'settings': sorted({key for r in self.records for key in self._flatten_settings(r)[0]}),
        }

    def _record(self, record_or_id: MaterialInstance | str) -> MaterialInstance:
        if isinstance(record_or_id, MaterialInstance):
            return record_or_id
        return self.by_id[str(record_or_id)]

    def facet_distances(
        self,
        left: MaterialInstance | str,
        right: MaterialInstance | str,
        *,
        soft_identity: bool = False,
    ) -> dict[str, DistanceInterval]:
        a, b = self._record(left), self._record(right)
        return {
            "composition": _aggregate([
                _status_contribution(a.composition, b.composition, _composition_value_distance)
            ]),
            "material_identity": self._mapping_distance(
                a.material_identity, b.material_identity, soft=soft_identity
            ),
            "process_method": _aggregate([
                _status_contribution(
                    a.process_method,
                    b.process_method,
                    lambda x, y: _categorical_value_distance(x, y, soft=soft_identity),
                )
            ]),
            "step_type": self._sequence_summary_distance(a, b, mode="types"),
            "sequence": self._sequence_summary_distance(a, b, mode="sequence"),
            "settings": self._settings_distance(a, b, soft=soft_identity),
        }

    def _mapping_distance(
        self,
        left: Mapping[str, DocumentedValue],
        right: Mapping[str, DocumentedValue],
        *,
        soft: bool,
    ) -> DistanceInterval:
        contributions = []
        keys=set(left)|set(right)
        if self.feature_policy=='fixed_schema': keys.update(self.feature_schema['identity'])
        for key in sorted(keys):
            a = left.get(key, DocumentedValue.unknown())
            b = right.get(key, DocumentedValue.unknown())
            contributions.append(
                _status_contribution(
                    a,
                    b,
                    lambda x, y: _value_distance(
                        x, y, scales=self.numeric_scales, soft=soft
                    ),
                    retain_joint_unknown=self.feature_policy=='fixed_schema',
                )
            )
        return _aggregate(contributions)

    def _sequence_value(self, record: MaterialInstance, mode: str) -> DocumentedValue:
        status = record.process_sequence_status
        if status is not ReportingStatus.REPORTED:
            return DocumentedValue(status=status, kind="set" if mode == "types" else "categorical")
        sequence = [step.step_type for step in record.process_steps]
        if mode == "types":
            return DocumentedValue.reported(sorted(set(sequence)), kind="set")
        return DocumentedValue.reported(sequence, kind="categorical")

    def _sequence_summary_distance(
        self, left: MaterialInstance, right: MaterialInstance, *, mode: str
    ) -> DistanceInterval:
        a, b = self._sequence_value(left, mode), self._sequence_value(right, mode)
        if mode == "types":
            comparator = lambda x, y: jaccard_distance(_as_set(x.value), _as_set(y.value))
        else:
            comparator = lambda x, y: normalized_sequence_distance(x.value, y.value)
        return _aggregate([_status_contribution(a, b, comparator)])

    @staticmethod
    def _step_occurrences(record: MaterialInstance) -> tuple[dict[str, ProcessStep], set[str]]:
        output = {key: record.process_steps[index]
                  for index, key in process_step_keys(record)}
        return output, set(output)

    def _flatten_settings(
        self, record: MaterialInstance
    ) -> tuple[dict[str, DocumentedValue], set[str]]:
        steps, present_steps = self._step_occurrences(record)
        output = {
            f"global:{normalize_text(key)}": value
            for key, value in record.process_settings.items()
        }
        for step_key, step in steps.items():
            for setting_key, value in step.settings.items():
                output[f"{step_key}:{normalize_text(setting_key)}"] = value
        return output, present_steps

    def _settings_distance(
        self, left: MaterialInstance, right: MaterialInstance, *, soft: bool
    ) -> DistanceInterval:
        if self.settings_alignment=='sequence':
            return self._aligned_settings_distance(left,right,soft=soft)
        a_map, a_steps = self._flatten_settings(left)
        b_map, b_steps = self._flatten_settings(right)
        contributions = []
        keys=set(a_map)|set(b_map)
        if self.feature_policy=='fixed_schema': keys.update(self.feature_schema['settings'])
        for key in sorted(keys):
            step_key = key.rsplit(":", 1)[0]

            def missing_value(record: MaterialInstance, present_steps: set[str]) -> DocumentedValue:
                if key.startswith("global:"):
                    return DocumentedValue.unknown()
                if record.process_sequence_status is not ReportingStatus.REPORTED:
                    return DocumentedValue.unknown()
                if step_key not in present_steps:
                    return DocumentedValue.not_applicable()
                return DocumentedValue.unknown()

            a = a_map.get(key, missing_value(left, a_steps))
            b = b_map.get(key, missing_value(right, b_steps))
            contributions.append(
                _status_contribution(
                    a,
                    b,
                    lambda x, y: _value_distance(
                        x, y, scales=self.numeric_scales, soft=soft
                    ),
                    retain_joint_unknown=self.feature_policy=='fixed_schema',
                )
            )
        return _aggregate(contributions)

    def _aligned_settings_distance(self, left, right, *, soft):
        """Sequence alignment sensitivity; canonical orientation makes ties symmetric.

        Operation tokens alone determine edit costs. A substitution between
        different operations supplies separate unmatched setting groups.
        This deterministic alignment is not a recovered physical-stage mapping.
        """
        def signature(r):
            return repr([(normalize_text(s.step_type),sorted((k,v.status.value,repr(v.value)) for k,v in s.settings.items())) for s in r.process_steps])
        if signature(left)>signature(right): left,right=right,left
        a,b=left.process_steps,right.process_steps
        n,m=len(a),len(b);dp=np.zeros((n+1,m+1),dtype=int)
        dp[:,0]=np.arange(n+1);dp[0,:]=np.arange(m+1)
        for i in range(1,n+1):
            for j in range(1,m+1):
                cost=int(normalize_text(a[i-1].step_type)!=normalize_text(b[j-1].step_type))
                dp[i,j]=min(dp[i-1,j]+1,dp[i,j-1]+1,dp[i-1,j-1]+cost)
        pairs=[];i,j=n,m
        while i or j:
            cost=int(normalize_text(a[i-1].step_type)!=normalize_text(b[j-1].step_type)) if i and j else 2
            if i and j and dp[i,j]==dp[i-1,j-1]+cost:
                if cost: pairs.extend([(a[i-1],None),(None,b[j-1])])
                else:pairs.append((a[i-1],b[j-1]))
                i-=1;j-=1
            elif i and dp[i,j]==dp[i-1,j]+1:pairs.append((a[i-1],None));i-=1
            else:pairs.append((None,b[j-1]));j-=1
        contributions=[]
        groups=[({normalize_text(k):v for k,v in left.process_settings.items()},{normalize_text(k):v for k,v in right.process_settings.items()},'global')]
        for x,y in reversed(pairs):
            token=normalize_text((x or y).step_type)
            groups.append((None if x is None else {normalize_text(k):v for k,v in x.settings.items()},
                           None if y is None else {normalize_text(k):v for k,v in y.settings.items()},token))
        for x,y,token in groups:
            keys=set(x or {})|set(y or {})
            if self.feature_policy=='fixed_schema':
                for key in self.feature_schema['settings']:
                    prefix,name=key.rsplit(':',1)
                    if (token=='global' and prefix=='global') or prefix.split('#')[0]==token:keys.add(name)
            for key in sorted(keys):
                def value(mapping,record):
                    if mapping is None:
                        return DocumentedValue.not_applicable() if record.process_sequence_status is ReportingStatus.REPORTED else DocumentedValue.unknown()
                    return mapping.get(key,DocumentedValue.unknown())
                contributions.append(_status_contribution(value(x,left),value(y,right),
                    lambda v,w:_value_distance(v,w,scales=self.numeric_scales,soft=soft),
                    retain_joint_unknown=self.feature_policy=='fixed_schema'))
        return _aggregate(contributions)

    @staticmethod
    def _composite(
        facets: Mapping[str, DistanceInterval], weights: Mapping[str, float]
    ) -> DistanceInterval:
        positive = {key: float(value) for key, value in weights.items() if float(value) > 0}
        total = sum(positive.values())
        lower = sum(positive[key] * facets[key].lower for key in positive) / total
        upper = sum(positive[key] * facets[key].upper for key in positive) / total
        finite = {key: value for key, value in positive.items() if facets[key].reported is not None}
        finite_total = sum(finite.values())
        reported = (
            sum(finite[key] * float(facets[key].reported) for key in finite) / finite_total
            if finite_total
            else None
        )
        common = sum(positive[key] * facets[key].common_reported_fraction for key in positive) / total
        one_sided = sum(positive[key] * facets[key].one_sided_fraction for key in positive) / total
        structural = sum(
            positive[key] * facets[key].structural_difference_fraction for key in positive
        ) / total
        return DistanceInterval(
            float(lower),
            None if reported is None else float(reported),
            float(upper),
            float(common),
            float(one_sided),
            float(structural),
            float(total),
            float(sum(positive[key] * facets[key].excluded_weight for key in positive)),
        )

    def compare(
        self,
        left: MaterialInstance | str,
        right: MaterialInstance | str,
        view: ViewSpec,
    ) -> PairDistance:
        a, b = self._record(left), self._record(right)
        facets = self.facet_distances(a, b, soft_identity=view.soft_identity)
        composite = self._composite(facets, view.facet_weights)
        return PairDistance(a.record_id, b.record_id, facets, composite, view.name)

    def pairwise(self, view: ViewSpec) -> PairwiseDistanceResult:
        n = len(self.records)
        optimistic = np.zeros((n, n), dtype=np.float32)
        reported = np.zeros((n, n), dtype=np.float32)
        pessimistic = np.zeros((n, n), dtype=np.float32)
        common = np.ones((n, n), dtype=np.float32)
        width = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i + 1, n):
                interval = self.compare(self.records[i], self.records[j], view).composite
                optimistic[i, j] = optimistic[j, i] = interval.lower
                reported_value = np.nan if interval.reported is None else interval.reported
                reported[i, j] = reported[j, i] = reported_value
                pessimistic[i, j] = pessimistic[j, i] = interval.upper
                common[i, j] = common[j, i] = interval.common_reported_fraction
                width[i, j] = width[j, i] = interval.width
        return PairwiseDistanceResult(
            tuple(record.record_id for record in self.records),
            view,
            optimistic,
            reported,
            pessimistic,
            common,
            width,
        )


def _collect_numeric_values(records: Sequence[MaterialInstance]) -> dict[str, list[float]]:
    values: dict[str, list[float]] = defaultdict(list)
    for record in records:
        candidates = list(record.material_identity.values()) + list(record.process_settings.values())
        for step in record.process_steps:
            candidates.extend(step.settings.values())
        for value in candidates:
            if value.status is ReportingStatus.REPORTED and value.kind in {"numeric","numeric_multiset"}:
                raw = value.value if value.kind=='numeric_multiset' else [value.value]
                values[value.unit_key or "unitless"].extend(float(x) for x in raw)
    return values


def fit_numeric_scales(
    records: Sequence[MaterialInstance],
    *,
    log_unit_keys: Iterable[str] = (),
) -> dict[str, NumericScale]:
    """Fit robust scales from reported values only; no missing value is filled."""

    log_keys = set(log_unit_keys)
    output = {}
    for key, raw_values in _collect_numeric_values(records).items():
        transform = "log1p_signed" if key in log_keys else "linear"
        transformed = np.asarray(
            [math.copysign(math.log1p(abs(x)), x) if transform != "linear" else x for x in raw_values],
            dtype=float,
        )
        if len(transformed) >= 4:
            low, high = np.quantile(transformed, [0.1, 0.9])
            scale = float(high - low)
        elif len(transformed) >= 2:
            scale = float(np.ptp(transformed))
        else:
            scale = 1.0
        output[key] = NumericScale(max(scale, 1e-12), transform)
    return output
