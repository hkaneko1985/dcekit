#!/usr/bin/env python3
"""Run the revised Phase 1 HTEM process-data audit without imputation."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


FIELD_META = {
    "deposition_compounds": ("source_material", "categorical_array"),
    "deposition_power": ("energy_delivery", "numeric_array"),
    "deposition_target_pulses": ("energy_delivery", "numeric_array"),
    "deposition_rep_rate": ("energy_delivery", "numeric_array"),
    "deposition_energy": ("energy_delivery", "numeric"),
    "deposition_base_pressure_mtorr": ("atmosphere", "numeric"),
    "deposition_growth_pressure_mtorr": ("atmosphere", "numeric"),
    "deposition_gases": ("atmosphere", "categorical_array"),
    "deposition_gas_flow_sccm": ("atmosphere", "numeric_array"),
    "deposition_initial_temp_c": ("thermal_setting", "numeric"),
    "deposition_sample_time_min": ("time_and_repetition", "numeric"),
    "deposition_cycles": ("time_and_repetition", "numeric"),
    "deposition_substrate_material": ("substrate_and_geometry", "categorical"),
    "deposition_ts_distance": ("substrate_and_geometry", "numeric"),
}

PULSE_EVIDENCE = {"deposition_target_pulses", "deposition_rep_rate", "deposition_energy"}
POWER_EVIDENCE = "deposition_power"
FORBIDDEN_TOKENS = ("resistivity", "conductivity", "bandgap", "trans", "xrd_peak", "thickness", "absolute_temp")


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return None
    if text[0] in "[{":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return value
    if text.casefold() in {"null", "none", "nan", "n/a", "unknown"}:
        return None
    return value


def slot_summary(value: Any) -> tuple[int, int, list[Any]]:
    """Return declared slots, reported slots, and reported values."""
    value = parse_jsonish(value)
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, (list, tuple)):
        total = len(value)
        reported_values = []
        for item in value:
            _, _, nested = slot_summary(item)
            reported_values.extend(nested)
        return total, len(reported_values), reported_values
    if value is None:
        return 0, 0, []
    if isinstance(value, str) and not value.strip():
        return 0, 0, []
    return 1, 1, [value]


def reported(value: Any) -> bool:
    return slot_summary(value)[1] > 0


def active_power(value: Any) -> bool:
    values = slot_summary(value)[2]
    for item in values:
        try:
            if math.isfinite(float(item)) and float(item) != 0:
                return True
        except (TypeError, ValueError):
            return True
    return False


def empirical_family(process: dict[str, Any]) -> str:
    power = active_power(process.get(POWER_EVIDENCE))
    pulse = any(reported(process.get(field)) for field in PULSE_EVIDENCE)
    if power and pulse:
        return "hybrid_parameterized"
    if power:
        return "power_parameterized"
    if pulse:
        return "pulse_parameterized"
    return "unresolved"


def quantile_from_hist(histogram: Counter[int], probability: float) -> int:
    total = sum(histogram.values())
    if total == 0:
        return 0
    threshold = probability * (total - 1)
    cumulative = 0
    for value in sorted(histogram):
        cumulative += histogram[value]
        if cumulative - 1 >= threshold:
            return value
    return max(histogram)


def cramers_v(table: dict[str, dict[str, int]]) -> float:
    rows = sorted(table)
    columns = sorted({column for row in table.values() for column in row})
    matrix = [[table[row].get(column, 0) for column in columns] for row in rows]
    n = sum(sum(row) for row in matrix)
    if n == 0 or min(len(rows) - 1, len(columns) - 1) <= 0:
        return 0.0
    row_totals = [sum(row) for row in matrix]
    column_totals = [sum(matrix[i][j] for i in range(len(rows))) for j in range(len(columns))]
    chi2 = 0.0
    for i in range(len(rows)):
        for j in range(len(columns)):
            expected = row_totals[i] * column_totals[j] / n
            if expected > 0:
                chi2 += (matrix[i][j] - expected) ** 2 / expected
    return math.sqrt(chi2 / (n * min(len(rows) - 1, len(columns) - 1)))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(result: dict[str, Any], output: Path) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    families = result["families"]
    fields = list(FIELD_META)
    rates = np.asarray([[result["field_reporting_by_family"][family][field]["rate"] for field in fields] for family in families])
    short = [field.replace("deposition_", "").replace("_mtorr", "").replace("_sccm", "") for field in fields]
    fig = plt.figure(figsize=(15, 9), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.05, 2.4])
    ax0 = fig.add_subplot(grid[0, 0])
    counts = [result["family_counts"][family] for family in families]
    ax0.barh(families[::-1], counts[::-1], color="#2f6f8f")
    ax0.set_title("Empirical control-profile families")
    ax0.set_xlabel("sample libraries")
    for index, value in enumerate(counts[::-1]):
        ax0.text(value + max(counts) * 0.01, index, str(value), va="center", fontsize=9)

    ax1 = fig.add_subplot(grid[:, 1])
    image = ax1.imshow(rates, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax1.set_yticks(range(len(families)), families)
    ax1.set_xticks(range(len(fields)), short, rotation=55, ha="right")
    ax1.set_title("Field-level reporting rate (not process existence)")
    for i in range(len(families)):
        for j in range(len(fields)):
            ax1.text(j, i, f"{rates[i, j]:.0%}", ha="center", va="center", fontsize=7,
                     color="white" if rates[i, j] < 0.35 else "black")
    fig.colorbar(image, ax=ax1, shrink=0.75, label="reporting rate")

    ax2 = fig.add_subplot(grid[1, 0])
    hist = result["pairwise_common_field_histogram"]
    x = [int(key) for key in sorted(hist, key=int)]
    y = [hist[str(key)] / result["pairwise_setting_coverage"]["pair_count"] for key in x]
    ax2.bar(x, y, color="#b8792d")
    ax2.set_title("Pairwise jointly reported settings")
    ax2.set_xlabel("number of common fields (of 14)")
    ax2.set_ylabel("fraction of all pairs")
    ax2.set_ylim(bottom=0)

    fig.suptitle("HTEM revised Phase 1: process metadata audit", fontsize=16)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix(".svg"))
    plt.close(fig)


def make_report(result: dict[str, Any], output: Path) -> None:
    n = result["sample_libraries"]
    family_lines = "\n".join(
        f"| `{family}` | {result['family_counts'][family]:,} | {result['family_counts'][family] / n:.1%} |"
        for family in result["families"]
    )
    field_lines = "\n".join(
        f"| `{field}` | {result['field_reporting'][field]['reported']:,} | {result['field_reporting'][field]['rate']:.1%} | "
        f"{result['field_reporting'][field]['partial_array_records']:,} |"
        for field in FIELD_META
    )
    proxy = result["context_proxy_audit"]
    coverage = result["pairwise_setting_coverage"]
    text = f"""# 改訂 Phase 1：HTEM 工程記載監査

## 結論

HTEM全 **{n:,} sample library** を再監査し、目的変数・専門家判断・欠損補完を一切用いないPhase 1を完了した。実質的な工程設定が1項目以上記載されたlibraryは **{result['libraries_with_any_reported_process']:,} ({result['libraries_with_any_reported_process']/n:.1%})**、全14項目が未記載だったlibraryは **{result['libraries_without_reported_process']:,} ({result['libraries_without_reported_process']/n:.1%})** だった。

最も重要な結果は、HTEMには工程設定値は豊富にある一方、**明示的な合成方法名、複数工程のイベント列、工程順序は収録されていない**ことである。`deposition_cycles` は {result['sequence_audit']['cycles_reported']:,} 件にあるが、何を1 cycleとするかと各cycle内の工程列がないため、焼成→冷却→再焼成のような系列には復元できない。配列はターゲット／ガス等の並列スロットであり、時系列として扱わない。

よってPhase 2でHTEMから直接評価できるのは、主に**共通して記載された設定条件の類似度と、その未記載による区間幅**である。工程種類、順序・反復の重みを含む一般式は維持するが、HTEMに情報がないブロックを推定値で埋めない。工程系列の本格検証はNanoMine等の明示的工程記録を持つデータに移す。

## 1. 使用情報と隔離

- 使用：元素集合、14のdeposition設定項目、sample library ID。
- 監査専用：`deposition_instrument`。これは装置コードであり合成方法名ではないため、類似度へ入れない。
- 不使用：電気・光学物性、膜厚、XRDピーク数、測定温度、所有者、日付、研究ID。
- 欠損補完：なし。空欄、`null`、`[null,...]` は `NOT_REPORTED` とした。
- `0`：明示的な設定値として保持した。欠損とはしない。
- `NOT_PERFORMED`、`UNCONTROLLED`、`NOT_APPLICABLE`：HTEMの現行スキーマから明示判定できる記録は0件であり、空欄から推定していない。

## 2. 方法の代わりに識別できた記載プロファイル

方法名がないため、設定項目だけから「empirical control-profile family」を機械的に作った。これは方法分類ではなく、データ監査用の記載プロファイルである。

| 記載プロファイル | library数 | 割合 |
|---|---:|---:|
{family_lines}

`pulse_parameterized` {result['family_counts'].get('pulse_parameterized',0):,}件中、`deposition_cycles` も記載されたものは **{result['cycles_by_family'].get('pulse_parameterized',0):,}件**だった。一方、`power_parameterized` ではcycle記載は **{result['cycles_by_family'].get('power_parameterized',0):,}件**だった。これは興味深いデータ構造だが、方法名の代替ラベルではなく、次段階で検討する仮説である。

装置コードと記載プロファイルのCramér's Vは **{result['instrument_family_association']['cramers_v']:.3f}** だった。したがって記載パターンは装置・入力様式に強く依存し得る。装置IDや欠損マスクだけで得られるクラスターは材料類似度の発見とは扱わない。

## 3. 項目別の記載状況

| 設定項目 | 記載library | 記載率 | 一部nullを含む配列 |
|---|---:|---:|---:|
{field_lines}

同一配列内の値とnullの混在は、値が記載されたスロットだけを利用し、nullスロットを非実施とも一致ともみなさない。工程テンプレートはプロファイル別の記載頻度を示すだけで、個別試料の実工程を補完しない。

## 4. ペアごとの共通記載量

全 **{coverage['pair_count']:,}** ペアについて、14項目中で両方に値が記載された項目数を集計した。

- 共通項目0：**{coverage['zero_common_pairs']:,} ({coverage['zero_common_fraction']:.1%})**
- 中央値：**{coverage['quantiles']['0.50']}項目**
- 10–90%点：**{coverage['quantiles']['0.10']}–{coverage['quantiles']['0.90']}項目**
- 最大：**{coverage['maximum']}項目**

Phase 2では、共通記載項目だけから条件付き類似度を計算する一方、未記載部分の重みは再配分せず、支持下限・可能上限の幅へ入れる。共通項目0のペアは「工程が同じ」ではなく「工程類似度未解決」とする。

## 5. 工程テンプレートの固定結果

閾値は事前に `common >= 80%`、`optional = 20–80%`、`rare < 20%`、テンプレート作成の最小群サイズを **20件** と固定した。1件しかない `hybrid_parameterized` はテンプレートを確定せず `insufficient_sample_size` とした。テンプレートの `structural_not_applicable` は全て空である。実方法が不明なため、未記載を非該当へ変換する根拠がないからである。

生成物：

- `results/process_templates.json`：記載プロファイル別テンプレート
- `results/field_reporting_by_family.csv`：方法候補×設定項目の記載率
- `results/component_occurrence_by_family.csv`：工程facetの記載率
- `results/library_process_inventory.csv.gz`：library別状態・プロファイル

## 6. 工程系列の評価可能性

| 類似度成分 | HTEMでの状態 | Phase 2での扱い |
|---|---|---|
| 工程種類 | 単一のdeposition記録しかなく識別力がほぼない | 対照・被覆報告のみ |
| 工程順序 | 明示イベント列0件 | 欠損補完せず未解決 |
| 反復 | cycle数は一部あるが反復単位・内部順序不明 | 設定値としてのみ探索、系列とはしない |
| 工程内設定 | 複数項目に実データあり | 主な検討対象。共通記載項目だけで比較し区間化 |

## 7. 弱い支持情報の可用性

HTEM study表には **{proxy['studies']} studies**、複数libraryを含むstudyは **{proxy['multilibrary_studies']}**、既知libraryとのリンクは **{proxy['linked_libraries']} libraries** ある。同一studyはPhase 2の弱い正例としてのみ使用し、異なるstudyを負例とはしない。機関IDは現在のスナップショットにないため、同一機関検証はHTEMでは実施不能である。

## 8. Phase 1判定

判定は **`{result['phase1_decision']['status']}`** とした。

これは「一般的な工程系列類似度をHTEMで検証できる」というGOではない。次を意味する。

1. 設定条件ブロックはHTEMでPhase 2解析を行える。
2. 未記載を補完せず、類似度区間とpairwise reporting coverageを必須とする。
3. 方法名、工程順序、反復系列についてHTEMから結論を出さない。
4. 装置・欠損パターン対照を必須とする。
5. NanoMineで方法・工程列を取得できた場合に、種類・順序・反復を包括検証する。

## 9. 改訂した最終判断基準

本研究は教師なし探索なので、全重み・全解像度で一つの結論に収束することを要求しない。Phase 2以降では、事前固定した解析格子の一部であっても、次を満たすクラスターまたは材料対が得られれば「興味深い探索結果あり」とする。

- 組成・工程のどの差がまとまりを作ったか機械的に記述できる。
- 行順変更または再標本化で再現する。
- reported/lower/upper境界のどこで維持・消失するか示せる。
- 同一study等の弱い支持情報がある、または未知の組合せとして明確な仮説を生む。
- 欠損マスクのみ・装置のみ・工程シャッフル対照だけでは同じ結果を説明できない。

「一部でよい」は事後的に都合のよい一例を選ぶ意味ではない。全プロファイルを開示し、興味深い結果が存在した領域と存在しなかった領域を併記する。

![Phase 1 process audit](../figures/phase1r_process_audit.png)
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    common_threshold = float(spec["template_policy"]["common_reporting_rate"])
    optional_threshold = float(spec["template_policy"]["optional_reporting_rate"])
    minimum_group_size = int(spec["template_policy"]["minimum_group_size"])

    records = []
    with gzip.open(args.input, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            serialized_keys = json.dumps(row, sort_keys=True).casefold()
            if any(token in serialized_keys for token in FORBIDDEN_TOKENS):
                raise ValueError("Forbidden target/state variable found in audit input")
            records.append(row)
    if not records:
        raise ValueError("No records")

    family_counts = Counter()
    cycles_by_family = Counter()
    field_counts = {field: Counter() for field in FIELD_META}
    family_field_counts: dict[str, dict[str, Counter]] = defaultdict(lambda: {field: Counter() for field in FIELD_META})
    component_counts: dict[str, Counter] = defaultdict(Counter)
    family_instrument: dict[str, Counter] = defaultdict(Counter)
    masks = []
    inventory = []
    no_process = 0

    for row in records:
        process = row["process"]
        family = empirical_family(process)
        family_counts[family] += 1
        instrument = (row.get("validation_only") or {}).get("deposition_instrument") or "NOT_REPORTED"
        family_instrument[family][str(instrument)] += 1
        mask = 0
        statuses = {}
        components_present = Counter()
        for bit, (field, (component, kind)) in enumerate(FIELD_META.items()):
            total_slots, reported_slots, _ = slot_summary(process.get(field))
            is_reported = reported_slots > 0
            statuses[field] = "REPORTED" if is_reported else "NOT_REPORTED"
            if is_reported:
                mask |= 1 << bit
                components_present[component] = 1
                field_counts[field]["reported"] += 1
                family_field_counts[family][field]["reported"] += 1
            if total_slots > reported_slots > 0:
                field_counts[field]["partial_array_records"] += 1
                family_field_counts[family][field]["partial_array_records"] += 1
        masks.append(mask)
        if mask == 0:
            no_process += 1
        if reported(process.get("deposition_cycles")):
            cycles_by_family[family] += 1
        for component in sorted({value[0] for value in FIELD_META.values()}):
            component_counts[family][component] += int(bool(components_present[component]))
        inventory.append({
            "sample_library_id": row["sample_library_id"],
            "empirical_family": family,
            "reported_field_count": mask.bit_count(),
            "process_completeness_tier": "C_single_event_settings" if mask else "E_no_usable_process",
            "explicit_method_name": "NOT_REPORTED",
            "ordered_event_sequence": "NOT_REPORTED",
            "cycle_count_status": "REPORTED_UNIT_UNRESOLVED" if reported(process.get("deposition_cycles")) else "NOT_REPORTED",
            "deposition_instrument_validation_only": instrument,
            **{f"state:{field}": state for field, state in statuses.items()},
        })

    families = sorted(family_counts, key=lambda name: (-family_counts[name], name))
    n = len(records)
    reporting = {
        field: {
            "reported": field_counts[field]["reported"],
            "rate": field_counts[field]["reported"] / n,
            "partial_array_records": field_counts[field]["partial_array_records"],
        }
        for field in FIELD_META
    }
    by_family = {}
    templates = {}
    field_rows = []
    component_rows = []
    for family in families:
        denominator = family_counts[family]
        by_family[family] = {}
        template = {
            "method_defining_evidence": [],
            "common": [],
            "optional": [],
            "rare": [],
            "unresolved": [],
            "structural_not_applicable": [],
            "unresolved_due_to_small_group": [],
        }
        for field in FIELD_META:
            count = family_field_counts[family][field]["reported"]
            rate = count / denominator
            data = {"reported": count, "rate": rate, "partial_array_records": family_field_counts[family][field]["partial_array_records"]}
            by_family[family][field] = data
            if denominator < minimum_group_size:
                if count:
                    template["unresolved_due_to_small_group"].append(field)
                else:
                    template["unresolved"].append(field)
            elif family in {"power_parameterized", "hybrid_parameterized"} and field == POWER_EVIDENCE:
                template["method_defining_evidence"].append(field)
            elif family in {"pulse_parameterized", "hybrid_parameterized"} and field in PULSE_EVIDENCE and count:
                template["method_defining_evidence"].append(field)
            elif rate >= common_threshold:
                template["common"].append(field)
            elif rate >= optional_threshold:
                template["optional"].append(field)
            elif rate > 0:
                template["rare"].append(field)
            else:
                template["unresolved"].append(field)
            field_rows.append({"empirical_family": family, "field": field, "facet": FIELD_META[field][0], **data})
        templates[family] = {
            "label_status": "reporting/control-profile only; not a synthesis-method label",
            "template_status": "frozen" if denominator >= minimum_group_size else "insufficient_sample_size",
            "minimum_group_size": minimum_group_size,
            "sample_libraries": denominator,
            **template,
            "warning": "The template defines comparability and reporting expectations only; it never fills an individual missing value or infers an unreported step.",
        }
        for component in sorted({value[0] for value in FIELD_META.values()}):
            count = component_counts[family][component]
            component_rows.append({"empirical_family": family, "component": component, "reported": count, "rate": count / denominator})

    histogram = Counter()
    for left in range(n):
        for right in range(left + 1, n):
            histogram[(masks[left] & masks[right]).bit_count()] += 1
    pair_count = n * (n - 1) // 2
    quantiles = {f"{q:.2f}": quantile_from_hist(histogram, q) for q in [0.10, 0.25, 0.50, 0.75, 0.90]}

    studies = json.loads(args.studies.read_text(encoding="utf-8"))
    known_ids = {int(row["sample_library_id"]) for row in records}
    linked = {int(value) for study in studies for value in (study.get("sample_library") or []) if int(value) in known_ids}
    result = {
        "version": "2.0.0",
        "sample_libraries": n,
        "libraries_with_any_reported_process": n - no_process,
        "libraries_without_reported_process": no_process,
        "families": families,
        "family_counts": dict(family_counts),
        "field_reporting": reporting,
        "field_reporting_by_family": by_family,
        "cycles_by_family": dict(cycles_by_family),
        "missing_state_counts": {
            "REPORTED_field_cells": sum(value["reported"] for value in reporting.values()),
            "NOT_REPORTED_field_cells": n * len(FIELD_META) - sum(value["reported"] for value in reporting.values()),
            "NOT_PERFORMED": 0,
            "UNCONTROLLED": 0,
            "NOT_APPLICABLE": 0,
            "AMBIGUOUS_explicit": 0,
            "EXTRACTION_FAILED": 0,
            "partial_array_records": sum(value["partial_array_records"] for value in reporting.values()),
        },
        "process_completeness_tiers": {
            "A_full_method_sequence_repetition_settings": 0,
            "B_method_sequence_partial_settings": 0,
            "C_single_event_settings": n - no_process,
            "D_method_name_only": 0,
            "E_no_usable_process": no_process,
        },
        "sequence_audit": {
            "explicit_method_names": 0,
            "explicit_ordered_event_sequences": 0,
            "cycles_reported": reporting["deposition_cycles"]["reported"],
            "cycles_with_defined_repeat_unit_and_internal_order": 0,
            "array_interpretation": "parallel target/gas slots, not temporal events",
        },
        "pairwise_setting_coverage": {
            "pair_count": pair_count,
            "zero_common_pairs": histogram[0],
            "zero_common_fraction": histogram[0] / pair_count,
            "quantiles": quantiles,
            "maximum": max(histogram),
            "definition": "number of the 14 setting fields reported in both records; no imputation and no weight redistribution",
        },
        "pairwise_common_field_histogram": {str(key): histogram[key] for key in sorted(histogram)},
        "instrument_family_association": {
            "role": "source/reporting-pattern diagnostic only",
            "cross_tabulation": {family: dict(counts) for family, counts in family_instrument.items()},
            "cramers_v": cramers_v({family: dict(counts) for family, counts in family_instrument.items()}),
        },
        "context_proxy_audit": {
            "studies": len(studies),
            "multilibrary_studies": sum(len(study.get("sample_library") or []) >= 2 for study in studies),
            "linked_libraries": len(linked),
            "institution_identifier_available": False,
            "policy": "same-study is a weak positive; different-study is unlabeled",
        },
        "phase1_decision": {
            "status": "PROCEED_WITH_SCOPE_RESTRICTION",
            "setting_similarity": "testable with pairwise reporting coverage and interval bounds",
            "operation_type_similarity": "not discriminative in the current HTEM schema",
            "order_and_repetition_similarity": "not testable as a sequence; do not impute",
            "synthesis_method_similarity": "not testable because no explicit method label is present",
            "next": "Phase 2 HTEM settings-focused clustering plus negative controls; defer full sequence validation to NanoMine or another sequence-bearing dataset",
        },
        "final_judgment": spec["phase2_fixed_design"]["final_judgment"],
    }

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    (output / "results").mkdir(exist_ok=True)
    (output / "reports").mkdir(exist_ok=True)
    (output / "figures").mkdir(exist_ok=True)
    (output / "results" / "phase1r_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "results" / "process_templates.json").write_text(json.dumps(templates, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output / "results" / "field_reporting_by_family.csv", ["empirical_family", "field", "facet", "reported", "rate", "partial_array_records"], field_rows)
    write_csv(output / "results" / "component_occurrence_by_family.csv", ["empirical_family", "component", "reported", "rate"], component_rows)
    inventory_path = output / "results" / "library_process_inventory.csv.gz"
    with gzip.open(inventory_path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(inventory[0]))
        writer.writeheader()
        writer.writerows(inventory)
    make_figure(result, output / "figures" / "phase1r_process_audit.png")
    make_report(result, output / "reports" / "phase1r_report_ja.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
