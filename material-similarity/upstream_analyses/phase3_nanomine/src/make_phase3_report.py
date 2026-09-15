#!/usr/bin/env python3
"""Build the Japanese Phase 3 report from computed outputs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "report" / "PHASE3_REPORT_JA.md"


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{float(value):.{digits}f}"


def table(headers, rows):
    def clean(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    output = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    output.extend("| " + " | ".join(clean(value) for value in row) + " |" for row in rows)
    return "\n".join(output)


def finite_median(series):
    values = pd.Series(series).dropna()
    return float(values.median()) if len(values) else np.nan


def sentinel_frame(grid):
    specs = [
        ("材料のみ", "identity_strict", "equal", 0.0),
        ("均衡", "full_strict", "equal", 0.5),
        ("順序重視", "identity_soft", "sequence", 0.5),
        ("設定値重視", "identity_soft", "settings_all", 0.5),
        ("数値設定重視", "identity_soft", "settings_numeric", 0.5),
        ("工程のみ", "identity_soft", "equal", 1.0),
    ]
    rows = []
    for label, material, process, weight in specs:
        subset = grid[
            (grid.material_profile == material)
            & (grid.process_profile == process)
            & np.isclose(grid.process_lambda, weight)
            & (grid.bound == "reported")
            & (grid.requested_clusters == 50)
        ].iloc[0].copy()
        subset["view_label"] = label
        rows.append(subset)
    return pd.DataFrame(rows)


def main():
    manifest = json.loads((ROOT / "data" / "input_manifest.json").read_text(encoding="utf-8"))
    summary = json.loads((RESULTS / "phase3_summary.json").read_text(encoding="utf-8"))
    breadth_summary = json.loads((RESULTS / "pncextract_breadth_summary.json").read_text(encoding="utf-8"))
    grid = pd.read_csv(RESULTS / "phase3_grid.csv")
    sentinel = sentinel_frame(grid)
    retrieval = pd.read_csv(RESULTS / "sentinel_retrieval.csv")
    facet = pd.read_csv(RESULTS / "facet_audit.csv")
    missing = pd.read_csv(RESULTS / "missingness_control.csv")
    shuffle = pd.read_csv(RESULTS / "process_shuffle_control_summary.csv")
    stability = pd.read_csv(RESULTS / "bound_stability.csv")
    family = pd.read_csv(RESULTS / "phase3_family_grid.csv")
    duplicate = pd.read_csv(RESULTS / "duplicate_signature_audit.csv")
    dedup = pd.read_csv(RESULTS / "deduplicated_sentinel_sensitivity.csv")
    breadth = pd.read_csv(RESULTS / "pncextract_breadth_grid.csv")
    breadth_retrieval = pd.read_csv(RESULTS / "pncextract_breadth_retrieval.csv")
    candidates = pd.read_csv(RESULTS / "consensus_cross_paper_pairs.csv")
    inventory = pd.read_csv(ROOT / "data" / "process_family_inventory.csv")

    missing50 = missing[missing.k == 50].iloc[0]
    sent_retrieval = retrieval[retrieval.bound == "reported"].set_index("view")
    sentinel_names = {
        "material_only": "材料のみ", "balanced": "均衡", "sequence_focus": "順序重視",
        "settings_focus": "設定値重視", "numeric_settings_focus": "数値設定重視", "process_only": "工程のみ",
    }
    view_keys = ["material_only", "balanced", "sequence_focus", "settings_focus", "numeric_settings_focus", "process_only"]
    view_rows = []
    for (_, row), key in zip(sentinel.iterrows(), view_keys):
        view_rows.append([
            row.view_label,
            fmt(row.same_paper_cocluster),
            fmt(row.cross_paper_cocluster),
            fmt(row.same_paper_recall_lift, 2),
            fmt(row.paper_ari),
            fmt(row.shared_author_lift, 2),
            fmt(sent_retrieval.loc[key, "mrr_midrank"]),
            fmt(row.mean_interval_width),
        ])

    facet_label = {
        "matrix_strict": "マトリックス", "filler_strict": "フィラー", "surface_strict": "表面処理",
        "loading": "充填量", "descriptor": "構成材記述子", "process_family": "合成方法",
        "step_type": "工程種類", "sequence": "順序・反復", "settings_all": "全設定値",
        "settings_numeric": "数値設定値",
    }
    facet_rows = []
    for name in facet_label:
        row = facet[facet.facet == name].iloc[0]
        facet_rows.append([
            facet_label[name], fmt(row.reported_pair_fraction), fmt(row.mean_common_coverage),
            fmt(row.mean_interval_width), fmt(row.median_reported_distance),
        ])

    family_rows = []
    for family_name, group in family[(family.bound == "reported") & (family.requested_clusters == 20)].groupby("process_family"):
        family_rows.append([
            family_name, int(group.samples.iloc[0]), fmt(finite_median(group.same_paper_recall_lift), 2),
            fmt(finite_median(group.paper_ari)), fmt(finite_median(group.shared_author_lift), 2),
        ])

    operation_rows = []
    op = inventory[inventory.record_type == "operation"]
    for family_name, group in op.groupby("process_family"):
        expected = group[group.prevalence >= 0.10].sort_values("prevalence", ascending=False)
        operation_rows.append([
            family_name,
            int(group.family_sample_count.iloc[0]),
            ", ".join(f"{row['name']} ({row['prevalence']:.0%})" for _, row in expected.iterrows()),
        ])

    control_rows = []
    for control_name, display in (("sequence_order_shuffle", "順序シャッフル"), ("settings_value_shuffle", "設定値シャッフル")):
        subset = shuffle[(shuffle.control == control_name) & (shuffle.k == 50)]
        for metric, metric_label in (("same_paper_cocluster", "同一論文併合率"), ("paper_ari", "論文ARI"), ("shared_author_lift", "共著者lift")):
            row = subset[subset.metric == metric].iloc[0]
            control_rows.append([display, metric_label, fmt(row.original), fmt(row.shuffled_mean), fmt(row.original_minus_shuffled)])

    breadth50 = breadth[(breadth.bound == "reported") & (breadth.requested_clusters == 50)].sort_values(
        "same_paper_recall_lift", ascending=False
    )
    breadth_rows = [
        [row.profile, fmt(row.same_paper_cocluster), fmt(row.same_paper_recall_lift, 2), fmt(row.paper_ari), fmt(row.shared_author_lift, 2), fmt(row.mean_interval_width)]
        for _, row in breadth50.iterrows()
    ]

    candidate = candidates[
        (candidates.left_sample_id == "l157-s2-zhao-2008")
        & (candidates.right_sample_id == "l238-s2-zhao-2008")
    ].iloc[0]
    material_dedup = dedup[(dedup.view == "material_only") & (dedup.k == 50)].iloc[0]
    process_dedup = dedup[(dedup.view == "process_only") & (dedup.k == 50)].iloc[0]
    balanced_dedup = dedup[(dedup.view == "balanced") & (dedup.k == 50)].iloc[0]
    process_dup = duplicate[duplicate.signature == "process"].iloc[0]
    full_dup = duplicate[duplicate.signature == "full"].iloc[0]

    content = f"""# NanoMine材料類似度 Phase 3 実施報告

作成日: 2026-09-10  
解析種別: 目的変数を使わない教師なしクラスタリング  
結論方針: 単一の「最良類似度」は選ばず、用途別の複数ビューを維持する

## 1. 結論

Phase 3を実行した。主解析はNanoMine/MaterialsMineの **{summary['samples']:,}試料・{summary['papers']:,}論文**、補助的な組成幅広さ解析はNanoMine由来PNCExtractの **{breadth_summary['samples']:,}試料・{breadth_summary['papers']:,}論文**である。物性値・目的変数・顕微鏡像は一切使用していない。

得られた主要結論は次のとおりである。

1. **材料構成と工程を合わせると、同一論文内の近傍検索が大きく改善した。** 報告値距離のtie-aware MRRは、材料のみ {sent_retrieval.loc['material_only','mrr_midrank']:.3f}、工程のみ {sent_retrieval.loc['process_only','mrr_midrank']:.3f}、均衡ビュー {sent_retrieval.loc['balanced','mrr_midrank']:.3f} であった。したがって、二者は代替ではなく相補的である。
2. **工程中心のクラスタは同一論文を強く再現するが、欠損・記載様式も強く再現している。** k=50の論文ARIは工程のみ {sentinel[sentinel.view_label=='工程のみ'].paper_ari.iloc[0]:.3f} に対し、欠損マスクのみでも {missing50.paper_ari:.3f} であった。工程類似度の成功を、そのまま物理的類似性の証明と解釈してはいけない。
3. **工程設定値には欠損マスクを超える情報が少量ながら見られた。** k=50で設定値を層内シャッフルすると、論文ARIは {shuffle[(shuffle.control=='settings_value_shuffle')&(shuffle.k==50)&(shuffle.metric=='paper_ari')].original.iloc[0]:.3f} から平均 {shuffle[(shuffle.control=='settings_value_shuffle')&(shuffle.k==50)&(shuffle.metric=='paper_ari')].shuffled_mean.iloc[0]:.3f} へ低下した。一方、工程順序シャッフルではARIが低下せず、順序重視を支持する独立証拠は今回の弱いラベルからは得られなかった。
4. **欠損の影響はファセットごとに大きく異なる。** 表面処理の平均区間幅は {facet[facet.facet=='surface_strict'].mean_interval_width.iloc[0]:.3f}、全工程設定値は {facet[facet.facet=='settings_all'].mean_interval_width.iloc[0]:.3f} であり、単一値での順位付けは危険である。合成方法、工程種類、順序・反復は今回の抽出集合では系列が存在するため区間幅0であった。
5. **「興味深い結果」は得られた。** 異なる2論文 L157 と L238 のアルミナ/DGEBAエポキシ試料は、18個のセンチネル距離のうち {int(candidate.view_support)} 個で相互近傍候補となった。著者と所属Locationも一致し、工程はほぼ同一で、L238側にsolvent工程が1段追加されていた。専門家ラベルなしでも、関連研究系列を回収できた具体例である。
6. **組成のみの幅広さ解析でも傾向は再現したが、評価軸によって順位が変わった。** PNCExtractでk=50の同一論文liftはstrict_equal {breadth50[breadth50.profile=='strict_equal'].same_paper_recall_lift.iloc[0]:.2f}、loading_focus {breadth50[breadth50.profile=='loading_focus'].same_paper_recall_lift.iloc[0]:.2f}。一方、近傍MRRはstrict_equal {breadth_retrieval[(breadth_retrieval.profile=='strict_equal')&(breadth_retrieval.bound=='reported')].mrr_midrank.iloc[0]:.3f}、soft_loading {breadth_retrieval[(breadth_retrieval.profile=='soft_loading')&(breadth_retrieval.bound=='reported')].mrr_midrank.iloc[0]:.3f} であり、クラスタ分割と局所検索で望ましい類似度が異なった。

![Phase 3 summary](figures/phase3_summary.png)

## 2. データ取得と解析母集団

### 2.1 主解析

[MaterialsMine](https://materialsmine.org/) の公開サンプル索引には取得時点で1,610件があった。ライブSPARQLは知識グラフ移行中にHTTP 503を返したため、ウェブサイト自身が利用する公開検索キャッシュから、次の既存レスポンスを取得した。

- 材料構成ビュー: 907ユニーク試料
- 工程ビュー: 832ユニーク試料
- DOI・試料ラベル・合成方法ビュー: 836ユニーク試料
- 3ビューの積集合: **832試料**

同一試料に複数のキャッシュ文書があったが、内容衝突は0件であった。主解析は「公開キャッシュに工程ビューが存在した試料」の全件であり、NanoMine全1,610件の無作為標本ではない。この選択バイアスを明記する。

論文・著者・Locationの補助メタデータには、[PNCExtract](https://github.com/ghazalkhalighinejad/PNCExtract) の公開データを用いた。解析対象832件のうち794件で補助メタデータが得られた。ソース固定コミットはMaterialsMine `{manifest['source']['materialsmine_commit']}`、PNCExtract `{manifest['source']['pncextract_commit']}` である。

NanoMineがポリマーナノコンポジットのprocessing–structure–property情報をXMLスキーマと知識グラフで統合する設計であることは、[NanoMine schema論文](https://doi.org/10.1063/1.5046839) および [knowledge graph論文](https://doi.org/10.1007/978-3-030-62466-8_10) と整合する。本解析はそのうち材料構成とprocessingだけを使った。

### 2.2 合成方法別の工程棚卸し

まず合成方法で層別し、各層で10%以上の試料に記載された工程を棚卸しした。これは欠損補完には使わず、工程語彙とデータ完全性の把握にだけ使用した。

{table(['合成方法', '試料数', '10%以上に記載された工程（記載率）'], operation_rows)}

工程系列は、単なる集合ではなく、工程インデックス順の列として保持した。同一工程の繰返しも列中に複数回残した。

### 2.3 PNCExtract幅広さ集合

PNCExtractの手動整理済みsample_dataから1,103試料・217論文を得た。変数はmatrix名、filler名、各略称、filler mass fraction、volume fractionである。工程は含まれないため、組成類似度の補助検証専用とした。充填量は、mass/volumeの片方のみ記載1,016件、両方9件、両方なし78件であり、補完には適さない構成だった。

## 3. 類似度の定義

### 3.1 材料構成ファセット

- matrix identity: 正規化完全一致Jaccard、および文字3-gramによるsoft matching
- filler identity: 同上
- surface treatment identity: 同上
- loading: filler mass fraction、volume fraction
- component descriptor: density、width、aspect ratio、specific surface area

化学名のsoft matchingは表記差を緩和するだけで、化学構造・官能基・機構の同値性を推定するものではない。

### 3.2 工程ファセット

- 合成方法: Solution Processing、Melt Mixing、In-Situ Polymerization、Other Processing
- 工程種類: 記載工程タイプ集合のJaccard距離
- 順序・反復: 工程タイプ列の正規化編集距離。置換コストには工程ラベル集合のJaccard距離を使った
- 工程内設定: `工程タイプ#同種工程の出現番号:設定名` で対応付けた設定値距離

温度、時間、圧力、回転速度、長さは単位変換し、その他の数値は同一設定キー内のrobust rangeで尺度化した。カテゴリ設定は正規化文字3-gram距離とした。自由記述Descriptionは論文文体の漏洩を避けるため主解析から除いた。

材料距離と工程距離は、

$$
d_{{\\mathrm{{total}}}}=(1-\\lambda)d_{{\\mathrm{{material}}}}+\\lambda d_{{\\mathrm{{process}}}},
\\qquad 0\\leq\\lambda\\leq1
$$

で統合した。実際の $\\lambda$ は0、0.25、0.5、0.75、1とした。材料4プロファイル、工程6プロファイルを固定し、重み最適化は行っていない。

## 4. 欠損値の扱い

欠損値は一切補完していない。2試料で比較可能な変数集合を $C$、片側だけ記載された集合を $U$ とすると、

$$
d_L=\\frac{{\\sum_{{v\\in C}}d_v}}{{|C\\cup U|}},\\quad
d_R=\\frac{{\\sum_{{v\\in C}}d_v}}{{|C|}},\\quad
d_U=\\frac{{\\sum_{{v\\in C}}d_v+|U|}}{{|C\\cup U|}}
$$

とした。$d_L$ はoptimistic、$d_R$ はreported、$d_U$ はpessimisticである。両試料とも未記載の変数は比較から外す。共通変数がないファセットのreported値は未定義とし、他ファセットの重みを再正規化した。

- 工程ブロック自体がない試料は主解析から除外済み
- 記載された工程列に工程がない場合は「記載系列にない」という観測差に使用
- ある工程内の設定値がない場合は、未設定か未記載か識別できないため距離区間を広げる
- fillerやsurface roleが取得ビューに現れない場合、存在しないと断定せず未記載扱い

ファセット別の比較可能性は次のとおりである。

{table(['ファセット', 'reported可能ペア率', '平均共通被覆', '平均区間幅', 'reported距離中央値'], facet_rows)}

表面処理と工程設定は情報不足の影響が非常に大きい。これらを高重みにする場合、reportedだけでなく上下界を必ず併記すべきである。

## 5. クラスタリングと検証

- アルゴリズム: precomputed distanceに対するaverage-linkage階層クラスタリング
- グローバルcluster数: 8、12、20、30、50、80
- 合成方法内cluster数: 4、8、12、20
- 主グリッド: 360距離ビュー × 6 cluster数 = **2,160クラスタリング**
- 合成方法別: **288クラスタリング**
- PNCExtract幅広さ解析: **126クラスタリング**

検証情報は距離作成には用いず、クラスタ結果の後から評価した。

- 主弱ラベル: 同一NanoMine文献ID（L/E接頭辞）の別試料、4,985ペア
- 補助情報: 異なる論文で著者が1名以上一致、8,437ペア
- 補助情報: 異なる論文でLocation文字列が完全一致、92ペア
- 異なる論文ペアは負例ではなく、比較背景集合

liftはcluster数が多いほど背景併合率が下がって機械的に大きくなるため、異なるk間で「最良」を決める目的には使っていない。

## 6. 主解析結果

### 6.1 センチネル6ビュー

k=50、reported距離の結果を示す。

{table(['ビュー', '同一論文併合率', '異論文併合率', '同一論文lift', '論文ARI', '共著者lift', '近傍MRR', '平均区間幅'], view_rows)}

全センチネルで論文ラベル200回置換に対する片側p値は最小解像度の $1/(200+1)=0.00498$ であった。ただし、これは「同じ論文の試料が偶然以上にまとまる」ことだけを示し、材料物理の真のクラスを示さない。

均衡ビューの近傍MRR {sent_retrieval.loc['balanced','mrr_midrank']:.3f} は、材料のみ・工程のみのいずれよりも高い。材料構成と工程の一致を同時に要求すると、同一研究系列が近傍の先頭へ移動する。一方、工程のみはk=50で論文ARI {sentinel[sentinel.view_label=='工程のみ'].paper_ari.iloc[0]:.3f} と高く、論文固有のプロトコルを強く表現する。

### 6.2 欠損マスク対照

欠損マスクだけでも、k=50で同一論文併合率 {missing50.same_paper_cocluster:.3f}、lift {missing50.same_paper_recall_lift:.2f}、論文ARI {missing50.paper_ari:.3f}、近傍MRR {missing50.mrr_midrank:.3f} となった。したがって、同一論文ラベルは「材料が似る」だけでなく「同じ項目が同じ様式で記載される」ことも捉える。

この結果から、工程のみの高い論文再現率を主要な成功指標にするのは不適切である。むしろ、意味情報を用いた距離が欠損マスクをどれだけ超えるか、異論文・共著者系列をどれだけ回収するかを併記すべきである。

### 6.3 順序・設定値シャッフル

{table(['対照', '指標 (k=50)', '元データ', 'シャッフル平均', '差'], control_rows)}

順序シャッフルでは同一論文併合率は0.041低下したが、論文ARIは低下しなかった。このため、工程順序が無意味とは言えないものの、今回の弱ラベルで順序重視を選ぶ根拠は得られていない。設定値シャッフルでは併合率、ARI、共著者liftがすべて低下し、実設定値が欠損パターン以外の情報を持つことが示唆された。

### 6.4 欠損上下界の安定性

optimisticとpessimisticのクラスタARI中央値は全設定で {summary['bound_stability']['median_optimistic_pessimistic_ari']:.3f} だった。process weight $\\lambda$ が0の材料のみではk=50中央値 {stability[(stability.process_lambda==0)&(stability.requested_clusters==50)].ari_optimistic_pessimistic.median():.3f}、$\\lambda=1$ の工程のみでは {stability[(stability.process_lambda==1)&(stability.requested_clusters==50)].ari_optimistic_pessimistic.median():.3f} である。材料側、とくにsurface/loadingの未記載がクラスタを動かしている。

### 6.5 合成方法別クラスタリング

方法ラベルが単一の試料だけを用いた層別結果である。表はk=20、6センチネルビューの中央値。

{table(['合成方法', '試料数', '同一論文lift', '論文ARI', '共著者lift'], family_rows)}

In-Situ Polymerizationの極端に高い値は、論文ごとに工程テンプレートがほぼ固有であることを反映する可能性が高く、他方法との優劣比較には使えない。Other Processingは27件と小さく、一部指標が未定義である。

![Phase 3 sensitivity](figures/phase3_sensitivity.png)

### 6.6 完全重複の影響

工程署名は222種類しかなく、同一論文内の完全重複工程ペアが {int(process_dup.within_paper_duplicate_pairs):,} あった。一方、材料構成と工程を合わせた完全重複は {int(full_dup.within_paper_duplicate_pairs):,} ペアである。論文内の完全入力重複91試料を除き741試料にした感度解析でも、k=50の論文ARIは材料のみ {material_dedup.paper_ari:.3f}、均衡 {balanced_dedup.paper_ari:.3f}、工程のみ {process_dedup.paper_ari:.3f} で、主要傾向は消えなかった。

## 7. PNCExtract組成幅広さ解析

k=50、reported距離の結果を示す。

{table(['プロファイル', '同一論文併合率', '同一論文lift', '論文ARI', '共著者lift', '平均区間幅'], breadth_rows)}

identityを厳密に使うかsoftにするか、loadingを入れるかで結果が変わる。特にloading_focusは、k=50の同一論文liftがoptimistic {breadth[(breadth.profile=='loading_focus')&(breadth.bound=='optimistic')&(breadth.requested_clusters==50)].same_paper_recall_lift.iloc[0]:.2f} からpessimistic {breadth[(breadth.profile=='loading_focus')&(breadth.bound=='pessimistic')&(breadth.requested_clusters==50)].same_paper_recall_lift.iloc[0]:.2f} まで動いた。充填量の欠損を単一値へ潰すべきでない具体例である。

## 8. 興味深い横断論文例

代表例は次の2試料である。

- `L157_S2_Zhao_2008`: *Mechanisms leading to improved mechanical performance in nanoscale alumina filled epoxy*、DOI `10.1016/j.compscitech.2008.01.009`
- `L238_S2_Zhao_2008`: *Improvements and mechanisms of fracture and fatigue properties of well-dispersed alumina/epoxy nanocomposites*、DOI `10.1016/j.compscitech.2008.07.010`

共通点はDGEBA epoxy / aluminium oxide、Solution Processing、190 °C・1,440 minの乾燥、3,450 rpm・20 minのhigh-shear mixing、同じ添加剤、80 °C・360 minと135 °C・600 minの2段加熱である。相違はL238側に`solvent`工程が追加される点である。著者集合とLocationが一致し、異なる論文であることも確認できた。

これは「全く同じ試料」の回収ではなく、**材料と主要工程が共通し、一つの工程差を持つ関連実験系列**の回収である。将来、文献内の介入効果を転移するときに最も有用な候補形である。

別の監査例として、`E139_S6_Hassinger_2016` と `E410_S2_Prasad_2021` はpolypropylene/silicaと工程列 `mixing → drying evaporation → extrusion/output` が一致して全18ビューで近傍になった一方、上位方法ラベルはSolution ProcessingとMelt Mixingに分かれた。これは、上位の合成方法名と実工程列を別ファセットにした設計が必要であることを示す。

## 9. 解釈と推奨する類似度の保持方法

単一の最良類似度は定めない。少なくとも次の4ビューを残す。

| 用途 | 主に使うファセット | 必須の併記 |
|---|---|---|
| 材料系列の俯瞰 | matrix・fillerのstrict/soft identity | 表記揺れ感度 |
| 配合近傍検索 | identity + loading | optimistic/reported/pessimistic、共通変数数 |
| 合成経路検索 | 方法 + 工程種類 + 順序・反復 | 順序シャッフル感度 |
| 実験プロトコル検索 | 方法 + 工程列 + 設定値 | 欠損マスク対照、設定値シャッフル |
| 総合材料インスタンス検索 | 材料 + 工程の均衡 | 材料のみ・工程のみとの比較 |

各ペアには総合距離だけでなく、ファセット距離ベクトル、共通記載率、区間幅を返すべきである。たとえば「総合的に近い」だけでなく、「材料は同一、工程は1段違い、設定値は11/12項目一致、未記載幅0.08」と説明できる形が望ましい。

## 10. 限界

1. 同一論文・共著者・Locationは弱い補助情報であり、真の材料クラスではない。
2. 公開キャッシュに工程ビューが存在した832件は、全1,610件からの無作為抽出ではない。
3. 欠損マスクだけでも論文を強く再現し、論文固有の記載様式が交絡する。
4. 工程列にない工程は「実施なし」ではなく「記載系列になし」としか言えない。
5. soft identityは文字列類似度であり、化学的同義語辞書や分子構造距離ではない。
6. クラスタ数によってliftが機械的に変わるため、絶対値の横比較には注意が必要である。
7. 目的変数を使っていないため、得られたクラスタが性能転移に最適であるとはまだ言えない。

## 11. Phase 3の判断

Phase 3の目的は達成した。

- 複数の材料・工程類似度を補完なしで実装した
- 工程種類、順序・反復、工程設定を分離して重み付けした
- 欠損を区間として伝播した
- 同一論文、共著者、Locationを距離外の補助情報として検証した
- 欠損マスク、順序シャッフル、設定値シャッフル、重複除去で交絡を調べた
- NanoMine由来のより広い組成データでも補助再現した
- 異論文間の具体的な関連実験系列を回収した

結論は「この類似度が最良」ではない。**材料同一性、配合、合成経路、実験プロトコル、総合材料インスタンスという用途ごとに異なる類似度を保持し、欠損区間と交絡対照を一緒に提示する**ことが、現時点で最も妥当である。

## 12. 再現ファイル

- `config/phase3_spec.json`: 凍結した解析仕様
- `config/phase3_run_config.json`: 全重み・cluster数・乱数seed
- `data/input_manifest.json`: 取得件数、ハッシュ、ソースcommit
- `data/nanomine_phase3_records.jsonl.gz`: 主解析の正規化レコード
- `results/phase3_grid.csv`: 2,160クラスタリングの全結果
- `results/phase3_family_grid.csv`: 合成方法別結果
- `results/facet_audit.csv`: 欠損被覆・区間幅
- `results/process_shuffle_controls.csv`: 全シャッフル反復
- `results/consensus_cross_paper_pairs.csv`: 横断論文候補
- `results/pncextract_breadth_grid.csv`: 組成幅広さ解析
- `src/`: 取得、前処理、距離、解析、作図、報告生成コード
- `tests/`: 距離・欠損処理の単体テスト
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(content, encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
