#!/usr/bin/env python3
"""Generate the Japanese Phase 2 analysis report from machine-readable results."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "report" / "PHASE2_REPORT_JA.md"


def load_json(name: str):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def f(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def main() -> None:
    result = load_json("phase2_results.json")
    posthoc = load_json("phase2_posthoc_sensitivity.json")
    pairs = load_json("interesting_pairs.json")
    profiles = load_json("selected_cluster_profiles.json")
    spec = json.loads((ROOT / "config" / "phase2_spec.json").read_text(encoding="utf-8"))
    with (RESULTS / "phase2_grid.csv").open(encoding="utf-8", newline="") as handle:
        grid = list(csv.DictReader(handle))

    process_rows = [row for row in grid if row["profile"] != "COMPOSITION_ONLY"]
    positive_source = sum(float(row["source_matched_delta_vs_composition_same_k"]) > 0 for row in process_rows)
    positive_mask = sum(float(row["source_matched_delta_vs_mask_control"]) > 0 for row in process_rows)
    positive_both = sum(
        float(row["source_matched_delta_vs_composition_same_k"]) > 0
        and float(row["source_matched_delta_vs_mask_control"]) > 0
        for row in process_rows
    )
    positive_mrr = sum(float(row["mrr_delta_vs_composition"]) > 0 for row in process_rows)

    variant_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in process_rows:
        variant_rows[row["variant"]].append(row)

    sensitivity = {row["descriptive_rank"]: row for row in posthoc["selected_configuration_sensitivity"]}
    diagnostics = {row["descriptive_rank"]: row for row in result["diagnostics"]}
    supported = set(result["context_supported_diagnostic_ranks"])
    selected_lines = []
    for rank, row in enumerate(result["descriptively_ranked_configurations"], 1):
        diag = diagnostics[rank]
        sens = sensitivity[rank]["all_libraries_source_matched_component_bootstrap"]
        q_sens = sensitivity[rank]["quantitative_composition_only"]["delta_bootstrap"]
        shuffle = diag["process_shuffle_within_element_system_and_instrument"]["source_matched_delta"]
        bound = diag["ari_against_uncertainty_variants"]
        selected_lines.append(
            "| {rank} | `{config}` | {clusters} | {delta} | {mask} | {ci} | {shuffle95} | {pvalue} | {rowari} | {bounds} | {mrr} | {decision} |".format(
                rank=rank,
                config=row["config_id"],
                clusters=row["clusters"],
                delta=f(row["source_matched_delta_vs_composition_same_k"]),
                mask=f(row["source_matched_delta_vs_mask_control"]),
                ci=f"[{f(sens['ci95'][0])}, {f(sens['ci95'][1])}]",
                shuffle95=f(shuffle["percentile_95"]),
                pvalue=f(shuffle["plus_one_p_one_sided"], 4),
                rowari=f(diag["row_order_stability"]["median_ari"]),
                bounds=f"{f(bound['optimistic'])} / {f(bound['pessimistic'])}",
                mrr=f(row["mrr_delta_vs_composition"], 4),
                decision="支持診断を通過" if rank in supported else "シャッフル対照を通過せず",
            )
        )

    variant_lines = []
    import numpy as np
    for variant in ["reported", "optimistic", "pessimistic"]:
        rows = variant_rows[variant]
        delta = np.asarray([float(row["source_matched_delta_vs_composition_same_k"]) for row in rows])
        mask = np.asarray([float(row["source_matched_delta_vs_mask_control"]) for row in rows])
        mrr = np.asarray([float(row["mrr_delta_vs_composition"]) for row in rows])
        variant_lines.append(
            f"| {variant} | {f(float(np.median(delta)))} | {f(float(np.max(delta)))} | "
            f"{sum((delta > 0) & (mask > 0))}/{len(rows)} | {f(float(np.min(mrr)), 4)} ～ {f(float(np.max(mrr)), 4)} |"
        )

    reported_rows = variant_rows["reported"]
    profile_lines = []
    for profile in sorted({row["profile"] for row in reported_rows}):
        rows = [row for row in reported_rows if row["profile"] == profile]
        delta = np.asarray([float(row["source_matched_delta_vs_composition_same_k"]) for row in rows])
        mask = np.asarray([float(row["source_matched_delta_vs_mask_control"]) for row in rows])
        best = rows[int(np.argmax(delta))]
        profile_lines.append(
            f"| `{profile}` | {f(float(np.median(delta)))} | {f(float(np.max(delta)))} | "
            f"{float(best['process_lambda']):g} / {int(float(best['clusters']))} | "
            f"{int(np.sum((delta > 0) & (mask > 0)))}/{len(rows)} | "
            f"{f(max(float(row['mrr_delta_vs_composition']) for row in rows), 4)} |"
        )

    lambda_lines = []
    for process_lambda in sorted({float(row["process_lambda"]) for row in reported_rows}):
        rows = [row for row in reported_rows if float(row["process_lambda"]) == process_lambda]
        delta = np.asarray([float(row["source_matched_delta_vs_composition_same_k"]) for row in rows])
        mask = np.asarray([float(row["source_matched_delta_vs_mask_control"]) for row in rows])
        lambda_lines.append(
            f"| {process_lambda:g} | {f(float(np.median(delta)))} | {f(float(np.max(delta)))} | "
            f"{int(np.sum((delta > 0) & (mask > 0)))}/{len(rows)} |"
        )

    top_pair = pairs["cross_study_cross_instrument_hypotheses"][0]
    cluster_candidates = [
        row for row in profiles["clusters"]
        if any(system == top_pair["element_system_1"] for system, _ in row["top_element_systems"])
    ]
    top_cluster = max(
        cluster_candidates,
        key=lambda row: next(
            (count for system, count in row["top_element_systems"] if system == top_pair["element_system_1"]),
            0,
        ),
    )
    process_details = []
    for row in top_pair["shared_process_fields"]:
        process_details.append(
            f"| `{row['field'].replace('deposition_', '')}` | {f(row['conditional_similarity'])} | {f(row['slot_coverage'])} |"
        )

    comp_base = result["composition_baseline"]["retrieval"]
    mask_retrieval = posthoc["reporting_mask_retrieval_control"]
    counts = result["counts"]
    coverage = result["coverage_by_profile"]["EQ"]
    instrument = result["instrument_only_control"]

    text = rf"""# HTEM材料類似度 Phase 2 実施報告

## 結論

Phase 2 は完了した。目的変数を使わず、HTEM薄膜ライブラリ **{counts['sample_libraries']:,}件**について、組成類似度と成膜設定類似度を組み合わせ、事前に定めた **{len(spec['process_weight_profiles'])}重みプロファイル × 3欠損シナリオ × 5非ゼロ工程重み × 10クラスタ解像度 = {counts['grid_clusterings'] - 10:,}件**と、組成のみ10件を解析した。

最終判断は、**「興味深いが限定的な文脈依存パターンが得られた」**である。原料・投入エネルギー、雰囲気、温度・時間を重視した複数の距離は、同じ研究に属するライブラリの大域的クラスタ共所属を組成のみより改善した。この改善は成膜装置と組成距離をそろえた比較、欠損マスク対照、工程値シャッフル対照を入れても5つの記述的候補で残った。一方、局所的な同一研究ライブラリ検索は改善しなかった。したがって、現時点で「単一の最良類似度」を選ぶ根拠はなく、**用途別・多視点の類似度**として扱うべきである。

![Phase 2 summary](figures/phase2_summary.png)

## 1. 解析対象と固定した判断

- HTEMは共通の薄膜成膜スキーマとして扱い、合成方法名、工程順序、繰り返し系列は今回の距離に含めなかった。
- 目的変数、物性、構造・形態の測定結果、膜厚、XRD、電気・光学特性は入力から除外した。
- 組成は元素集合に加え、XRFの定量組成がある **{counts['quantitative_composition_libraries']:,}件**では組成分率、周期表上の族・周期、ライブラリ内組成幅を用いた。
- 工程は14設定を4側面に分けた：原料・投入エネルギー、雰囲気、温度・時間、基板・幾何。
- 欠損値は補完しなかった。「0」は設定値として保持し、未設定・未記載と区別した。
- 同一研究の別ライブラリを弱い正例とした。異なる研究の組は負例ではなく、未ラベル比較対象とした。専門家判断は用いていない。
- 重みとクラスタ数は最適化せず、全グリッドを開示した。上位6条件は探索後の記述的要約であり、事前選択された最適条件ではない。

## 2. 類似度と欠損の扱い

組成距離を \(C_{{ij}}\)、工程距離を \(P_{{ij}}\)、工程重みを \(\lambda\) とし、材料距離を

\[
D_{{ij}} = C_{{ij}} + (1-C_{{ij}})\lambda P_{{ij}}
\]

とした。これにより組成が遠い材料を、工程が似ているだけで過度に近づけない。

各工程変数 \(v\) について、2試料で共通に記載された設定だけから条件付き類似度 \(s_{{v,ij}}\) を計算し、比較可能な設定スロット割合を \(q_{{v,ij}}\) とした。重み \(w_v\) に対して、記載から支持される類似度は

\[
S^{{\mathrm{{supported}}}}_{{ij}}
=\sum_v w_v q_{{v,ij}}s_{{v,ij}},\qquad
Q_{{ij}}=\sum_v w_vq_{{v,ij}}
\]

である。補完の代わりに次の3通りを全て計算した。

| シナリオ | 工程類似度 | 意味 |
|---|---:|---|
| optimistic | \(S^{{\mathrm{{supported}}}}+(1-Q)\) | 未記載部分が全て一致する上端 |
| reported | \(S^{{\mathrm{{supported}}}}/Q\)（\(Q=0\)なら工程寄与なし） | 共通記載部分に条件付けた主解析 |
| pessimistic | \(S^{{\mathrm{{supported}}}}\) | 未記載部分が全て不一致の下端 |

同時欠損は一致として数えていない。数値配列は双方に存在するスロット数まで比較し、残りは \(Q\) の低下として不確実性幅に入れた。カテゴリ配列は正規化後の集合Jaccard類似度を使った。

等重み（EQ）で全ペアの比較可能工程割合は平均 **{f(coverage['all_pair_mean'])}**、中央値 **{f(coverage['all_pair_median'])}**であり、弱い正例では平均 **{f(coverage['positive_pair_mean'])}**、中央値 **{f(coverage['positive_pair_median'])}**だった。弱い正例 {counts['known_positive_pair_instances']:,}件のうち **{coverage['positive_pairs_zero']}件**は、EQで共通記載工程がゼロだった。欠損境界への感度は周辺事項ではなく主要結果である。

## 3. 評価設計と交絡補正

弱い正例は {counts['studies_with_multiple_known_libraries']}研究、{counts['study_components']}独立研究成分、{counts['known_positive_pair_instances']:,}ペアインスタンス（{counts['known_positive_unique_pairs']:,}一意ペア）。未ラベル比較は {counts['unlabeled_comparison_pairs']:,}ペアである。同一研究ペアのクラスタ共所属率から、組成距離が近い未ラベルペアの期待共所属率を差し引き、研究成分を等重みで平均した。

初期監査で、弱い正例の **{counts['known_positive_pair_instances_same_instrument']:,}/{counts['known_positive_pair_instances']:,}件**が同じ成膜装置だった。このため、最終評価では未ラベル側も「組成距離ビン＋同じ装置／装置対」で完全一致させた。装置IDは距離の特徴には使わず、評価時の交絡調整だけに使った。装置だけでクラスタリングした未補正の成分等重み超過は **{f(instrument['all']['component_equal_excess'])}**だったが、装置照合後は **{f(instrument['all_source_matched']['component_equal_excess'])}**となり、この対照が意図通り働いた。

さらに、工程ベクトルを（a）元素系内、（b）元素系＋成膜装置内で50回シャッフルし、行順を3通り変えて安定性を検証した。装置補正は初回ドライラン後に必要性が判明して追加したため、以下は確認的検定ではなく探索的解析である。

## 4. 全グリッドの傾向

工程を含む {len(process_rows):,}クラスタリングのうち、装置補正後の成分等重み指標が組成のみを上回ったのは **{positive_source:,}件（{pct(positive_source/len(process_rows))}）**、欠損マスク対照を上回ったのは **{positive_mask:,}件（{pct(positive_mask/len(process_rows))}）**、両方を上回ったのは **{positive_both:,}件（{pct(positive_both/len(process_rows))}）**だった。局所検索MRRが組成のみを上回ったのは **{positive_mrr:,}件（{pct(positive_mrr/len(process_rows))}）**にとどまった。

| 欠損シナリオ | 装置補正delta中央値 | 最大値 | 組成・マスク対照の両方を上回る | MRR delta範囲 |
|---|---:|---:|---:|---:|
{chr(10).join(variant_lines)}

`pessimistic`でMRRだけが大きくなるのは、物理的な近さよりも欠損・記載パターンによる近傍を作るためと解釈するのが妥当である。したがって主解析は`reported`、optimistic/pessimisticは不確実性境界とした。

`reported`主解析を重みプロファイル別に集約すると次のようになった。各行の「最大条件」は選択結果の説明用であり、最適化結果ではない。

|工程内の固定重みプロファイル|装置補正delta中央値|最大値|最大時のlambda / クラスタ数|組成・マスク対照の両方を上回る|最大MRR delta|
|---|---:|---:|---:|---:|---:|
{chr(10).join(profile_lines)}

基板・幾何だけを重くした条件は相対的に弱く、原料・投入エネルギー、雰囲気、温度・時間を含む条件でより一貫した正の傾向が見られた。ただし、フィールドの記載率と研究デザインも異なるため、これを物理的重要度の順位とは解釈しない。

工程全体の重み \(\lambda\) 別では、主解析の中央値は重みとともに増えたが、最大値は0.75で現れた。

|lambda|装置補正delta中央値|最大値|組成・マスク対照の両方を上回る|
|---:|---:|---:|---:|
{chr(10).join(lambda_lines)}

したがって、今回の範囲では工程情報を極端に小さくするより中～高程度に効かせる方が大域クラスタに寄与したが、唯一の推奨値は定めない。

## 5. 記述的上位条件と頑健性

|順位|条件|クラスタ数|装置補正delta<br>vs組成|装置補正delta<br>vs欠損マスク|成分bootstrap 95%区間|装置維持shuffle 95%点|選択後p値|行順ARI|optimistic / pessimistic ARI|MRR delta|判断|
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(selected_lines)}

順位1～5は、定めた文脈支持診断を通過した。順位6は観測値がシャッフル95%点を超えず、支持候補から除外した。ここでのp値は50回シャッフルに対する各選択条件の参考値であり、**全1,960結果からの事後選択や多重性を補正していない**。有意差の主張には使えない。

研究成分bootstrapでは上位5件の平均deltaは **0.138～0.196**、95%区間の下端は **0.037～0.071**で正だったが、最大成分の絶対寄与割合は概ね0.28～0.34だった。効果は単一研究だけではない一方、研究系横断で一様でもない。

定量XRF組成を持つ1,252件だけで再解析すると、上位6件の装置補正delta平均は **0.163～0.257**で、全てのbootstrap区間下端が正だった。元素集合だけの粗い組成表現が結果を単独で作ったとは考えにくい。

## 6. 大域クラスタリングと局所検索は別の挙動

組成のみの同一研究検索は、138アンカーで MRR **{f(comp_base['mrr'],4)}**、Hit@10 **{f(comp_base['hit_at_10'],4)}**、Recall@10 **{f(comp_base['recall_at_10'],4)}**だった。記述的上位6条件のMRR差は全て負（約−0.0013～−0.0033）だった。一方、欠損マスクだけでは MRR **{f(mask_retrieval['mask_only']['mrr'],4)}**だった。

これは矛盾ではない。同一研究では同じ材料系・装置の中で工程条件を意図的に走査するため、研究全体は大域的にまとまっても、最も近い個別試料は工程値が異なり得る。したがって、

- 文献群・実験キャンペーンをまとめる類似度
- 最近傍材料を検索する類似度
- 改変効果を転移する類似度

は同じ重みである必要がない。今回の結果は「クラスタリング用途では工程を加える価値がある」が、「局所検索にも同じ距離を転用できる」とは示していない。

## 7. 興味深いクラスタ仮説

全195の工程込み距離をクラスタ数150で比較し、異なる研究・異なる成膜装置でも80%以上同じクラスタとなるペアを抽出した。20ペアが基準を満たした。最上位はライブラリ **{top_pair['sample_library_id_1']}**（装置{top_pair['instrument_1_validation_only']}）と **{top_pair['sample_library_id_2']}**（装置{top_pair['instrument_2_validation_only']}）で、どちらも **{top_pair['element_system_1']}**、組成距離 **{f(top_pair['composition_distance'])}**、EQの条件付き工程距離 **{f(top_pair['process_reported_distance_EQ'])}**、比較可能割合 **{f(top_pair['process_reporting_coverage_EQ'])}**、全条件共クラスタ率 **{pct(top_pair['coassignment_fraction_across_grid_at_k150'])}**だった。

選択条件の該当クラスタは87件で、N-Sn-Znが82件、装置1が59件、装置5が28件を含む。これは装置IDを特徴に使わずに現れた、装置横断の再現候補である。

|共通工程設定|条件付き類似度|ペア比較可能割合|
|---|---:|---:|
{chr(10).join(process_details)}

![Cross-instrument pair](figures/phase2_cross_instrument_pair.png)

このペア群は「同じ物理材料」と検証された正解ではなく、次段階で構造・物性・原論文記載を確認する候補である。

## 8. 到達した判断と限界

今回の教師なしPhase 2の判断基準――一部でも、対照を越えて解釈可能かつ再現的なクラスタ／材料ペアパターンが得られること――は満たした。ただし結論は次に限定する。

1. HTEMでは組成だけでなく工程設定を含む類似度が、大域クラスタリングの弱い文脈整合性を改善し得る。
2. とくに原料・投入エネルギー、雰囲気、温度・時間に信号がある。ただしこれは因果的重要度や普遍的な最適重みを意味しない。
3. 欠損境界でクラスタが大きく変わる条件がある。1個の点推定類似度ではなく、`reported`とoptimistic/pessimisticを併記すべきである。
4. 同一研究は独立した真値ラベルではなく、研究テーマ、装置、記載様式を共有する弱い代理情報である。装置交絡は補正したが、研究機関や未収録のバッチ情報は残り得る。
5. 事後選択、多重比較、弱い正例17成分という制約があるため、確認的な統計的有意性は主張しない。
6. 「最良距離」は選ばない。全グリッドの傾向と、用途別に残った候補群をPhase 3へ渡す。
7. 複数ターゲット、電力、パルス等の配列は、装置スロット順に依存しない別々の集合／ソート済み数値列として比較した。そのため「どのターゲットにどの電力を与えたか」というスロット間対応は表現していない。これは合成工程順序とは別の課題であり、対応関係が信頼できるHTEMレコードが得られる場合の追加感度解析対象である。

## 9. Phase 3へ渡す具体案

- 主候補として `SOURCE_ONLY__reported__L075`、`ATMOS_ONLY__reported__L075`、`ATMOS55__reported__L100` を保持する。
- 比較基準として組成のみ、EQ、欠損マスクのみを必ず保持する。
- NanoMineの包括検証では、材料インスタンスを組成、ポリマー／フィラー、加工条件、測定条件に分け、HTEMと共通する「補完なし・ペアごとの共通記載・区間境界」を維持する。
- NanoMineに工程系列がある場合は、HTEMでは省略した工程種類・順序・繰り返し距離を独立成分として追加する。
- クラスタ整合性と近傍検索を別々に評価し、同一重みを前提にしない。
- HTEM側では、ターゲット化学種とターゲット別電力・パルスの対応を保持した順序不変マッチングを追加し、`SOURCE_ONLY`結果の感度を確認する。
- N-Sn-Zn装置横断クラスタなどの候補は、目的変数を導入しない範囲では原レコード整合性と別データセット再現性で検証する。

## 10. 再現手順と成果物

リポジトリ直下で以下を実行する。

```bash
python src/prepare_phase2_input.py --input ../htem_download --output-root .
python src/run_phase2.py --config config/phase2_run_config.json
python src/run_posthoc_sensitivity.py --config config/phase2_run_config.json
python src/make_phase2_figures.py
python src/make_phase2_report.py
python -m unittest discover -s tests -v
```

主要成果物は次の通り。

- `results/phase2_results.json`: 主要集計、診断、判断
- `results/phase2_grid.csv`: 全1,960クラスタリング結果
- `results/phase2_posthoc_sensitivity.json`: 成分bootstrap、定量組成限定、検索対照
- `results/interesting_pairs.json`: 同一研究および研究・装置横断の候補ペア
- `results/selected_assignments.jsonl.gz`: 記述的上位条件の全割当
- `results/selected_cluster_profiles.json`: 上位条件のクラスタ要約
- `config/phase2_spec.json`: 解析仕様、欠損方針、監査履歴

---

解析ステータス: `{result['status']}`  
乱数seed: `{json.loads((ROOT / 'config' / 'phase2_run_config.json').read_text())['seed']}`  
主要解析実行時間: {result['runtime_seconds']:.1f}秒
"""

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
