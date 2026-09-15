# HTEM材料類似度 Phase 2

目的変数を使わず、HTEM薄膜ライブラリの組成と成膜設定から、欠損非補完の区間類似度を構築・比較する再現パッケージです。

## 実行

Python 3.11以降を想定しています。

```bash
python -m pip install -r requirements-phase2.txt
python src/run_phase2.py --config config/phase2_run_config.json
python src/run_posthoc_sensitivity.py --config config/phase2_run_config.json
python src/make_phase2_figures.py
python src/make_phase2_report.py
python -m unittest discover -s tests -v
```

入力を再構成する場合は、先に次を実行します。

```bash
python src/prepare_phase2_input.py --input ../htem_download --output-root .
```

## 設計上の要点

- HTEMは薄膜試料なので、合成方法名と工程順序は距離に含めません。
- 組成と14の成膜設定だけを使い、物性・性能・構造結果は使いません。
- 欠損値は補完せず、2試料で共通して記載された設定だけを比較します。
- 欠損部分はoptimistic／reported／pessimisticの幅として評価します。
- 同一研究の別ライブラリは弱い正例であり、異なる研究のペアを負例とはみなしません。
- 重みやクラスタ数は最適化せず、事前定義した全グリッドを開示します。
- 成膜装置は距離の特徴には入れず、同一研究代理ラベルの交絡調整にだけ使います。

詳細は `report/PHASE2_REPORT_JA.md` を参照してください。
