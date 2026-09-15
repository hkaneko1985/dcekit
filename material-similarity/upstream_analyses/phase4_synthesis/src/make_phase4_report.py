#!/usr/bin/env python3
"""Generate the Japanese Phase 4 report from machine-readable results."""

from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
REPORT = ROOT / "report" / "PHASE4_REPORT_JA.md"


def pct(value: float, digits: int = 1) -> str:
    return f"{100 * value:.{digits}f}%"


def num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def main() -> None:
    diag = pd.read_csv(RESULTS / "cross_dataset_diagnostics.csv").set_index("dataset")
    views = pd.read_csv(RESULTS / "view_disagreement_summary.csv").set_index("dataset")
    objectives = pd.read_csv(RESULTS / "objective_disagreement.csv")
    weights = pd.read_csv(RESULTS / "weight_response.csv")
    nano = pd.read_csv(RESULTS / "nanomine_sentinel_objectives.csv").set_index("view")
    pnc = pd.read_csv(RESULTS / "pncextract_k50_objectives.csv").set_index("profile")
    decision = json.loads((RESULTS / "phase4_decision.json").read_text(encoding="utf-8"))

    htem = diag.loc["HTEM"]
    nanod = diag.loc["NanoMine"]
    starry = diag.loc["Starrydata"]
    pncd = diag.loc["PNCExtract"]

    criteria_labels = {
        "A_cross_domain_context_change": "A. 3領域で組成以外の情報による文脈変化が正方向",
        "B_value_beyond_mask": "B. 実値が欠損・シャッフル対照を超える領域が2以上",
        "C_task_dependence": "C. 大域クラスタと局所検索の用途依存が2領域以上",
        "D_multi_view_nonidentity": "D. 主要ビューが同一分割へ収束しない領域が2以上",
        "E_cross_source_candidates": "E. 2領域で異sourceの合意候補を回収",
        "F_reproducibility": "F. ハッシュ・コード・機械可読結果・試験を保存",
    }
    criteria_table = "\n".join(
        f"| {criteria_labels[key]} | {'PASS' if value else 'FAIL'} |"
        for key, value in decision["criteria"].items()
    )
    weight_lines = []
    for dataset in ["HTEM", "NanoMine"]:
        subset = weights.loc[weights["dataset"] == dataset]
        values = ", ".join(
            f"λ={row.process_weight:.2f}: {row.global_gain_median:.3f}"
            for row in subset.itertuples()
        )
        weight_lines.append(f"- **{dataset}:** {values}")

    report = f"""# 材料類似度研究 Phase 4：データセット横断統合解析と論文化判断

作成日：2026-09-10  
解析種別：目的変数・専門家ラベル・欠損補完を用いない教師なし解析  
最終判断：**方法論論文としてGO、性能・改善効果転移の主張は保留**

## 1. 結論

Phase 1のStarrydata、Phase 2のHTEM、Phase 3のNanoMine、およびPNCExtract補助解析を統合した。独立トラックの延べ試料数は **{int(diag["samples"].sum()):,}** である。ただし、これは異なるデータベースを単純合計した数であり、一つの重複除去済みコーパスではない。

結論は「最良類似度を選べなかった」ではなく、次のように積極的に定式化できる。

> **材料類似度は単一の数値ではなく、材料構成、工程、欠損不確実性および利用目的からなる多視点量である。安定な材料関係は、単一ビューの順位ではなく、複数ビューの合意として抽出する。**

主要結果は次のとおりである。

1. Starrydata、HTEM、NanoMineの3領域すべてで、組成以外の形態・工程情報を加えたときの代表的な文脈指標が正方向だった。
2. 欠損マスク自体が論文・studyを強く再現した。工程類似度の成功を物理的類似性と同一視できない。
3. 実際の工程値は欠損マスクを超える信号を一部持つが、その証拠は「部分的」である。
4. 大域クラスタリングと局所近傍検索は異なる類似度を支持した。単一重みを両用途に共用すべきでない。
5. HTEMとNanoMineでは、複数ビューで安定した異source候補を回収できた。
6. 方法論論文は執筆可能である。一方、性能予測や文献内改善効果の転移可能性は未検証であり、別研究として扱う。

![Phase 4 cross-dataset overview](figures/phase4_cross_dataset_overview.png)

## 2. Phase 4の位置づけ

今回のPhase 4は、上流Phaseの結果を変更せずに統合する段階である。目的変数、物性値、特性測定結果、専門家による正解クラスは使用していない。同一論文、同一study、共著者、所属は類似度入力に入れず、クラスタリング後の弱い支持情報に限った。

Phase 4の判断基準は統合計算前に固定したが、Phase 1–3の結果を確認した後の基準である。したがって、統計的な事前登録ではなく、**論文化可否を決める研究管理上のルール**として扱う。

異なるデータセットでは弱い支持情報と評価量が異なるため、効果量を数値的にプールしなかった。共通の命題について、各データセット内の方向、対照、感度、分割不一致を統合した。

## 3. 統合対象

| データセット | 領域 | 試料 | 弱い支持群 | 主な非目的変数情報 |
|---|---|---:|---:|---|
| Starrydata | 無機材料文献横断 | {int(starry.samples):,} | {int(starry.weak_groups):,} SID | 組成、形態、合成・製造工程 |
| HTEM | 無機薄膜ライブラリ | {int(htem.samples):,} | {int(htem.weak_groups)} study連結成分 | 組成、原料・投入、雰囲気、温度・時間、基板・幾何 |
| NanoMine | ポリマーナノコンポジット | {int(nanod.samples):,} | {int(nanod.weak_groups)} 論文 | matrix、filler、界面処理、loading、工程種類・系列・設定 |
| PNCExtract | 組成幅広さ補助集合 | {int(pncd.samples):,} | {int(pncd.weak_groups)} 論文 | matrix、filler、mass/volume loading |

## 4. 横断命題の検証

![Phase 4 evidence matrix](figures/phase4_evidence_matrix.png)

### 4.1 組成以外の情報はクラスタ構造を変える

- Starrydata：F2−F0の対数共所属lift差は3 seed中央値 **{num(starry.context_gain_median)}**、正方向率 **{pct(starry.context_gain_positive_fraction)}**。
- HTEM：工程を含むreported 650ビューで、source補正後excessの組成基準との差は中央値 **{num(htem.context_gain_median)}**、正方向率 **{pct(htem.context_gain_positive_fraction)}**。
- NanoMine：工程を加えたreported 576ビューで、材料のみからの論文ARI差は中央値 **{num(nanod.context_gain_median)}**、正方向率 **{pct(nanod.context_gain_positive_fraction)}**。
- PNCExtract：strict identityへloadingを加えると、k=50の論文ARIは **{num(pnc.loc["strict_equal", "paper_ari"])} → {num(pnc.loc["strict_loading", "paper_ari"])}** と低下したが、局所MRRは **{num(pnc.loc["strict_equal", "mrr_midrank"])} → {num(pnc.loc["strict_loading", "mrr_midrank"])}** と上昇した。

正方向率は重みグリッド内の感度集計であり、独立反復の成功確率ではない。

### 4.2 実値と欠損・記載パターンを分離する

- Starrydata：F2と欠損マスクのlift比の対数は中央値 **{num(starry.semantic_vs_mask_median)}**、工程値シャッフルとの差は **{num(starry.semantic_vs_shuffle_median)}** だった。
- HTEM：reportedビューの意味情報−mask対照は中央値 **{num(htem.semantic_vs_mask_median)}**。選択済み6ビューのうち **{int(round(htem.shuffle_supported_fraction * 6))}/6** が、装置・元素系を保持した工程シャッフル95%点を上回った。
- NanoMine：設定値重視ビューのk=50論文ARIは設定値シャッフル平均より **{num(nanod.semantic_vs_shuffle_median)}** 高かった。一方、センチネル全体の論文ARIはmaskのみより中央値 **{num(nanod.semantic_vs_mask_median)}** 低く、局所MRRは中央値 **{num(nanod.semantic_vs_mask_local_median)}** 高かった。

工程値に信号はあるが、論文文脈の再現に関して欠損・記載様式は非常に強い。総合距離だけでなく、意味距離と記載完全度を別々に返す必要がある。

### 4.3 欠損境界は結論を動かす

HTEMではoptimistic・reported・pessimisticの全境界で工程増分が正だった設定は **{pct(htem.all_bounds_positive_fraction)}** に限られ、**{pct(htem.bound_sign_change_fraction)}** では符号が変わった。NanoMineでは全境界正が **{pct(nanod.all_bounds_positive_fraction)}**、符号変化が **{pct(nanod.bound_sign_change_fraction)}** だった。

NanoMineのoptimistic対pessimistic分割ARI中央値は **{num(nanod.bound_partition_ari_median)}**、HTEMの選択ビューと欠損境界とのARI中央値は **{num(htem.bound_partition_ari_median)}** だった。欠損を補完して単一距離にすると、この不安定性が不可視になる。

### 4.4 大域クラスタリングと局所検索は別の問題である

HTEMでは大域文脈指標とMRRの両方が組成基準を上回ったのは650ビューの **{pct(htem.global_local_joint_positive_fraction)}** だけだった。大域増分と局所増分のSpearman相関は **ρ={num(htem.global_local_spearman)}** だが、局所増分の中央値は負である。

NanoMineの6センチネルでは、大域論文ARIと局所MRRの順位相関は **ρ={num(nanod.global_local_spearman)}** で、実質的に無相関だった。均衡ビューはMRR **{num(nano.loc["balanced", "mrr"])}**、工程のみは論文ARI **{num(nano.loc["process_only", "paper_ari"])}** と、それぞれ異なる側面で強かった。

PNCExtractではloading追加により、大域論文ARIが低下しながら局所MRRが上昇した。これは最も明瞭な方向反転である。

## 5. 単一のクラスタ分割へ収束しない

主要ビュー間のpairwise ARIは次のとおりだった。

| データセット | ビュー対数 | ARI中央値 | 最小 | 最大 |
|---|---:|---:|---:|---:|
| Starrydata | {int(views.loc["Starrydata", "pair_count"])} | {num(views.loc["Starrydata", "median"])} | {num(views.loc["Starrydata", "minimum"])} | {num(views.loc["Starrydata", "maximum"])} |
| HTEM | {int(views.loc["HTEM", "pair_count"])} | {num(views.loc["HTEM", "median"])} | {num(views.loc["HTEM", "minimum"])} | {num(views.loc["HTEM", "maximum"])} |
| NanoMine | {int(views.loc["NanoMine", "pair_count"])} | {num(views.loc["NanoMine", "median"])} | {num(views.loc["NanoMine", "minimum"])} | {num(views.loc["NanoMine", "maximum"])} |

いずれも中央値は0.90未満である。単一の自然な分割が不明なのではなく、**組成系列、工程系列、実験プロトコル系列が異なる関係を表すため、分割が異なること自体が期待される**。

工程重みごとの大域増分中央値は次のとおりである。

{chr(10).join(weight_lines)}

これは「工程重み1が最良」という意味ではない。弱い支持ラベル自身が工程テンプレートと記載様式を共有するためであり、配合検索や局所検索では異なる重みが必要になる。

## 6. 複数ビュー合意による異source候補

単一ビューでなく複数ビューにわたる共所属・近傍支持を用いると、次の候補が得られた。

- HTEM：異study・異成膜装置で、195工程込み設定の80%以上に共所属する候補 **{int(htem.cross_source_consensus_pairs)}対**。
- NanoMine：異論文で18センチネル条件中16以上に支持される候補 **{int(nanod.cross_source_consensus_pairs)}対**。このうち **{int(nanod.cross_source_candidates_with_author_or_location_support)}対** は共著者またはLocation一致という外部文脈支持を持った。

NanoMineの代表例 L157_S2_Zhao_2008 と L238_S2_Zhao_2008 は、DGEBA epoxy / aluminium oxide、主要な混合・乾燥・二段加熱条件が共通し、一方にsolvent工程が追加された関連研究系列だった。HTEMの代表例はlibrary 10336と10725のN–Sn–Zn系で、異なる装置でも組成と複数設定が近かった。

これらは真の正例ではないが、**追加確認に回すべき関係を専門家採点なしで機械的に絞る**という用途には成立している。

## 7. 推奨する類似度の最終表現

材料ペア i,j について、単一距離ではなく次を保持する。

$$
\\mathbf{{d}}_{{ij}}=
\\left(
d_{{\\mathrm{{composition}}}},
d_{{\\mathrm{{material\\ identity}}}},
d_{{\\mathrm{{process\\ method}}}},
d_{{\\mathrm{{step\\ type}}}},
d_{{\\mathrm{{sequence}}}},
d_{{\\mathrm{{settings}}}}
\\right)
$$

各ファセットは補完せず、共通記載集合 C_ij に基づくreported距離と、片側記載集合 U_ij による上下界を返す。

$$
d_{{ij}}^{{R}}
=
\\frac{{\\sum_{{v\\in C_{{ij}}}}w_v d_{{ijv}}}}
{{\\sum_{{v\\in C_{{ij}}}}w_v}},
\\qquad
d_{{ij}}\\in[d_{{ij}}^L,d_{{ij}}^U]
$$

用途別ビュー m=1,…,M で得た関係の合意度を、

$$
c_{{ij}}=
\\frac{{1}}{{M}}
\\sum_{{m=1}}^M
I\\!\\left(i,j\\text{{ が同一クラスタまたは相互近傍}}\\right)
$$

として保持する。実際の出力は、総合値ではなく、

- ファセット距離ベクトル
- optimistic／reported／pessimistic
- 共通記載率と片側記載率
- 用途別クラスタ・近傍結果
- cross-view合意度

の組とする。

## 8. Phase 4 GO／NO-GO判断

| 判断基準 | 結果 |
|---|---|
{criteria_table}

全6基準がPASSした。規則上の最終判断は、

> **GO_FOR_METHODS_PAPER_HOLD_FOR_PERFORMANCE_TRANSFER_CLAIMS**

である。

### GOとする論文主張

1. 文献由来の不完全な材料メタデータから、組成と工程を分離した多視点類似度を構築できる。
2. 欠損を補完せず、ペアごとの共通記載と距離区間として扱える。
3. 組成以外の工程情報は複数材料領域で文脈的クラスタ構造を変える。
4. 欠損・記載様式は強い交絡であり、maskおよびshuffle対照が不可欠である。
5. クラスタリング、局所検索、実験系列検索では異なるビューが必要である。
6. 複数ビュー合意により、異sourceの安定候補を抽出できる。

### NO-GO／保留とする主張

- 普遍的に最良な材料類似度を得た
- クラスタが材料物理上の真の分類である
- 性能を予測できる
- 文献内の改善効果を対象材料へ定量転移できる
- 同一論文・studyが材料類似性の正解ラベルである

## 9. 論文化方針

方法論論文としての仮題は次が適切である。

**Multi-view and Missingness-aware Material Similarity from Incomplete Literature Metadata without Target Properties**

中心メッセージは、単一の最良類似度を競うことではなく、**similarity is query-dependent and uncertainty-bearing**である。

推奨構成は次のとおりである。

1. Introduction：組成類似度だけでは材料インスタンスを表現できず、文献工程は不完全である。
2. Methods：ファセット分解、補完なし距離区間、多重みビュー、弱い支持情報、mask/shuffle対照、cross-view consensus。
3. Starrydata：無機文献データで形態・工程の増分とsource交絡を示す。
4. HTEM：薄膜設定への外部展開とglobal/local不一致を示す。
5. NanoMine：工程種類・順序・反復・設定を分離し、詳細工程へ拡張する。
6. Cross-domain synthesis：共通する結論と領域依存の相違を整理する。
7. Discussion：単一最適化をしない理由、欠損の意味、性能転移との境界。

主図候補は、方法概念図、3領域の結果、欠損対照、global/local不一致、異source合意例の5点である。

## 10. 限界

1. すべての支持情報は文脈proxyであり、物理的類似性の正解ではない。
2. Starrydataは5,001試料のパイロット、NanoMineは工程キャッシュが得られた832試料であり、母集団の無作為標本ではない。
3. HTEMの弱い支持群は17連結成分と少なく、study間の均質性を仮定できない。
4. 重みグリッド内の設定は相関しており、正方向率を統計的な成功率とは解釈できない。
5. PNCExtractではloading値シャッフルと分割ARIを再計算していないため、補助的証拠に限る。
6. Phase 4は上流結果を確認後に統合したため、最終的な論文では探索的統合と明記する必要がある。

## 11. 次段階

次のPhaseでは、この結果をそのまま用いて**日本語論文草稿**を作成し、同時に用途別距離・距離区間・合意度を返す**統一API**へ既存コードを整理する。

性能や改善効果の転移は、本研究の結果を距離基盤として使用する別研究とする。そこで初めて目的変数変換、効果修飾因子、文献内改善幅、被覆条件付き上界を導入する。

## 12. 再現ファイル

- config/phase4_spec.json：統合方針と研究管理上の判断基準
- data/upstream_manifest.json：入力成果物のパス・サイズ・SHA-256
- results/cross_dataset_diagnostics.csv：データセット別の共通診断
- results/view_disagreement_pairs.csv：主要ビュー間ARI
- results/objective_disagreement.csv：大域・局所評価の全比較
- results/uncertainty_sensitivity.csv：欠損境界感度
- results/weight_response.csv：工程重み感度
- results/hypothesis_evidence_matrix.csv：横断命題の証拠表
- results/phase4_decision.json：機械可読なGO／NO-GO判断
- src/：統合、作図、報告生成コード
- tests/：統合結果の不変条件試験
"""
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(report, encoding="utf-8")
    print(REPORT)


if __name__ == "__main__":
    main()
