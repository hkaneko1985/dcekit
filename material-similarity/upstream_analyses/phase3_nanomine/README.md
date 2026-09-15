# NanoMine材料類似度 Phase 3

目的変数を使わず、NanoMineの材料構成と合成工程から複数の距離を構築し、教師なしクラスタリングを比較する再現パッケージです。

主要結果は [`report/PHASE3_REPORT_JA.md`](report/PHASE3_REPORT_JA.md) を参照してください。

## 解析の原則

- 単一の「最良類似度」を選びません。
- matrix、filler、surface treatment、loading、構成材記述子を材料側の別ファセットとして保持します。
- 合成方法、工程種類、順序・反復、工程内設定値を工程側の別ファセットとして保持します。
- 欠損値は補完せず、2試料に共通して記載された変数だけでreported距離を計算します。
- 片側だけ記載された変数はoptimistic／pessimistic区間に伝播します。
- DOI、論文ID、著者、所属は距離特徴に含めず、クラスタ結果の弱い補助検証にだけ使います。
- 物性、目的変数、画像、characterization結果は使いません。

## 同梱データ

- 主解析: 832試料、119文献
- 主解析グリッド: 360距離ビュー、2,160クラスタリング
- 合成方法別: 288クラスタリング
- PNCExtract組成幅広さ解析: 1,103試料、217文献、126クラスタリング

`data/raw_cache` は2026-09-10にMaterialsMineの公開検索APIから取得した固定スナップショットです。取得時、ライブSPARQLはHTTP 503だったため、公開サンプル画面用の既存キャッシュを利用しました。キャッシュ重複の内容衝突は0件です。

## 再実行

Python 3.11以降を想定しています。

```bash
python -m pip install -r requirements-phase3.txt
python src/run_phase3.py
python src/run_controls.py
python src/run_duplicate_sensitivity.py
python src/make_phase3_figures.py
python src/make_phase3_report.py
python -m unittest discover -s tests -v
```

正規化レコードを固定raw cacheから作り直す場合:

```bash
python src/prepare_phase3_input.py \
  --raw-dir data/raw_cache \
  --output-root . \
  --citation-dir ../PNCExtract/articles/data_source \
  --pnc-repo ../PNCExtract \
  --materialsmine-repo ../materialsmine
```

公開キャッシュを再取得する場合:

```bash
python src/acquire_phase3_cache.py --output data/raw_cache
```

PNCExtract幅広さ解析を作り直す場合:

```bash
git clone https://github.com/ghazalkhalighinejad/PNCExtract.git ../PNCExtract
python src/run_pncextract_breadth.py \
  --sample-root ../PNCExtract/sample_data \
  --metadata-dir ../PNCExtract/articles/data_source
```

## 主要出力

- `results/phase3_grid.csv`: 全グローバルクラスタリング
- `results/phase3_family_grid.csv`: 合成方法別結果
- `results/facet_audit.csv`: 比較可能率・区間幅
- `results/missingness_control.csv`: 欠損マスクのみの対照
- `results/process_shuffle_controls.csv`: 順序・設定値シャッフル反復
- `results/deduplicated_sentinel_sensitivity.csv`: 完全入力重複除去感度
- `results/consensus_cross_paper_pairs.csv`: 複数ビューで安定した異論文近傍候補
- `results/pncextract_breadth_grid.csv`: 組成のみの幅広さ解析

## 注意

同一論文、共著者、同一Locationはいずれも真の材料クラスではありません。異なる論文のペアも負例ではありません。評価値は、類似度の正解率ではなく、弱い文脈情報との整合と交絡を調べる診断値として解釈してください。

データ出典と利用条件は [`data/SOURCES_AND_LICENSES.md`](data/SOURCES_AND_LICENSES.md) に記載しています。

