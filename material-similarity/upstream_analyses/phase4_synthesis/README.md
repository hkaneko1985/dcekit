# 材料類似度 Phase 4

Phase 1（Starrydata）、Phase 2（HTEM）、Phase 3（NanoMine）およびPNCExtract補助解析を統合し、目的変数を用いない材料類似度研究の横断的な結論と論文化可否を判断する再現パッケージです。

## 方針

- 単一の最良類似度・最良重み・最良クラスタ数は選びません。
- 目的変数、物性値、専門家ラベルは使いません。
- 欠損値は補完しません。
- 同一論文・同一study・共著者・所属は類似度入力ではなく、事後的な弱い支持情報に限ります。
- 異なるデータセットの効果量は尺度が異なるため、数値的にプールしません。

## 実行

ワークスペース直下から実行します。

```bash
python phase4_material_similarity_synthesis/src/run_phase4.py
python phase4_material_similarity_synthesis/src/make_phase4_figures.py
python phase4_material_similarity_synthesis/src/make_phase4_report.py
python -m unittest discover -s phase4_material_similarity_synthesis/tests -v
```

## 主要成果物

- `report/PHASE4_REPORT_JA.md`: 統合報告書
- `results/cross_dataset_diagnostics.csv`: データセット別の共通診断
- `results/view_disagreement_pairs.csv`: 主要ビュー間のARI
- `results/objective_disagreement.csv`: 大域クラスタリングと局所検索の不一致
- `results/hypothesis_evidence_matrix.csv`: 仮説別の証拠表
- `results/phase4_decision.json`: 論文化GO／NO-GO判断
- `data/upstream_manifest.json`: 使用したPhase 1–3成果物のハッシュ

統合に必要な凍結済み上流結果は data/upstream に同梱しているため、
Phase 1–3のディレクトリがなくても再実行できます。元データから各Phaseを
再計算する場合は、それぞれのPhaseパッケージを使用してください。
