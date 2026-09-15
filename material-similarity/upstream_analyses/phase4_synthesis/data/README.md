# Phase 4 upstream inputs

data/upstream には、Phase 4統合結果を単独で再現するために必要な
Phase 1–3の凍結済み集計・クラスタ割当だけを収録しています。

- starry_*: Starrydata Phase 1の集計と確認集合を含む割当
- htem_*: HTEM Phase 2の集計、全感度グリッド、選択ビュー割当、候補ペア
- nano_*: NanoMine Phase 3の集計、全感度グリッド、センチネル割当、
  欠損・シャッフル対照、候補ペア
- pnc_*: PNCExtract補助解析の集計と感度グリッド

目的変数、物性値、特性測定結果は含みません。論文・study由来の情報は、
上流Phaseですでに距離特徴から隔離され、弱い事後支持情報としてだけ
集計されています。

data/upstream_manifest.json は各ファイルのSHA-256とサイズを記録します。
