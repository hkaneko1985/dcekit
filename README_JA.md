# 相互作用を考慮した組成記述子

本ディレクトリは、組成と物性の関係を学習したモデルを固定し、その潜在変数や予測値を別の物性予測の説明変数として利用する研究コードです。全組成を直接入力する経路と、すべての成分ペアを計算して境界整合的に拡張する経路を含みます。

著者は **Hiromasa Kaneko（金子弘昌）** です。対応原稿は *Reusable interaction-aware descriptors from frozen property models for materials prediction* です。

## DCEKitの公開先

研究コードは[DCEKitのinteraction_descriptorsディレクトリ](https://github.com/hkaneko1985/dcekit/tree/master/interaction_descriptors)で公開されています。2026年11月7日に確認した初回公開版は[コミット0d8d31e9d061208f85d3ef4d8de9dc587c43ca89](https://github.com/hkaneko1985/dcekit/tree/0d8d31e9d061208f85d3ef4d8de9dc587c43ca89/interaction_descriptors)です。

初回公開143ファイルのうち119件は確認済みZIPとバイト単位で同一です。残るCSV24件は改行形式がCRLFからLFになっただけで、数値・文字列は同一です。この改行変更に伴うチェックサム一覧の修正は、[コミット46f59b074bcaa2994070ad88c9a77253930da2fe](https://github.com/hkaneko1985/dcekit/tree/46f59b074bcaa2994070ad88c9a77253930da2fe/interaction_descriptors)で公開され、全146ファイルがv54修正候補と一致することを確認しました。確認範囲は[PUBLICATION_STATUS.md](PUBLICATION_STATUS.md)に記録しています。

今回のPillow依存関係の更新内容と検証範囲は[SECURITY_UPDATE.md](SECURITY_UPDATE.md)にまとめています。更新候補の作成・検証では、GitHubへの書き込みやDependabotアラートの終了確認は行っていません。

配置先はDCEKitリポジトリ直下の **`interaction_descriptors/`** です。既存のライブラリ本体である内側の`dcekit/`には入れません。配置とインストールの補足は[DCEKIT_INTEGRATION.md](DCEKIT_INTEGRATION.md)を参照してください。

## 最初の実行

Python 3.10以上とNumPyを使用します。DCEKitのリポジトリ直下で、使用するPython環境を選んでから次を実行します。

```bash
python -m pip install -e ./interaction_descriptors
cd interaction_descriptors
python -m interaction_descriptors synthetic --output example_output
python scripts/verify_synthetic.py --output example_output/synthetic_validation.json
```

通常の`pip install dcekit`とは別に、この研究用パッケージを明示してインストールします。以後のコマンドは`interaction_descriptors/`内で実行してください。仮想環境の利用を推奨します。

2026年11月の依存関係の点検に伴い、追加機能である過去の鋼材計算の再実行と作図の環境では **Pillow 13.0.0** を使用します。新しい環境には現在の`requirements-replay.txt`または`requirements-plot.txt`を使用してください。鋼材計算の再実行は従来どおりPython 3.13、作図はPython 3.11以上が必要です。コア機能は引き続きPython 3.10以上とNumPyのみで動作します。

過去のstage51の検証記録にあるPillow 12.3.0は、当時の実行環境を正確に残すための記録です。現在のインストール推奨ではありません。数値計算、入力データ、既存の研究結果は変更せず、再実行スクリプトの依存バージョン検査も新しい指定に合わせています。詳細は[REPLAY.md](REPLAY.md)を参照してください。

上の例は人工的な組成と小さな固定ネットワークで動作し、論文用の学習済み重みや材料データを必要としません。純成分・二成分・三成分・未対応成分を含む入力を計算し、記述子と適用可否を出力します。出力は材料物性の実予測としては使いません。

## 何を計算するコードか

| 内容 | 主な場所 |
|---|---|
| 全成分を保持する入力整列、固定モデル推論、全ペア出力、Kohler型境界拡張 | `src/interaction_descriptors/` |
| 原稿の図表出力と各再現経路の入口 | `scripts/reproduce.py` |
| 元の源モデル学習・図作成コードと由来 | `reference/`、`manifests/` |
| NN・YAE・YVAEの単独利用、三方式併用、Borutaの比較 | `experiments/` |
| 図表に用いる集計値 | `publication/`、`experiments/regressor_comparison/frozen_reference/` |
| 数値・再現・公開準備の確認記録 | `verification/` |

源モデルの潜在軸を平均する処理は行わず、独立に学習されたブロックを連結します。YVAEは固定した潜在平均を使用します。目的変数の情報を使った特徴量選択や前処理は、対象データの訓練部分の中で行います。

## 入力の扱い

成分名、列順、分率基準を明示してください。保存された元素モデルの入力は原子分率です。質量分率をそのまま原子分率として扱うことはできません。

実際に含まれない成分は0で補います。正の分率がある成分を源モデルが扱えない場合、その源モデルを使用不可とし、成分を削除して再正規化する処理は行いません。ペアの二つの分率はそれぞれ直接割り算で計算し、微量成分が`1-t`の丸めによって0になることを防ぎます。今回の修正と確認例は[RELEASE_NOTES.md](RELEASE_NOTES.md)に記録しています。

## ZIPだけで実行できる処理と外部資材が必要な処理

| 処理 | ZIP以外に必要なもの |
|---|---|
| 人工データの実行例・16件のコアテスト・6件の診断テスト | Python、NumPy |
| 本文・補足の表を集計値から出力 | Python |
| Figure 8の再生成 | Python 3.11以上と`requirements-plot.txt`の依存パッケージ |
| 実材料に対する固定モデル推論 | 対応する学習済みモデルとモデルカード |
| 原稿の予測比較の再学習 | 準備済みの元入力、分割・選択記録、対応する依存環境 |
| 源モデルの再学習・過去の図の厳密再生成 | 由来とハッシュを指定した元資材 |

外部資材の一覧は`data/README.md`、`models/README.md`、`manifests/`にあります。第三者の行単位データ、学習済み重み、行別予測のキャッシュは公開用ZIPに含めていません。公開用コードのMITライセンスが、それらの資材の配布条件を一括で変更するものではありません。

## 再現性と結果の位置づけ

これまでに、源モデルの代表1条件で27個の重み・尺度配列と143組成の記述子出力の完全一致を確認しました。Figure 1〜6は元コードから画像のバイト単位の一致を確認し、Figure 7は元の36集計値を使った別描画です。全源モデルと全実験を最初から一括で再学習したという意味ではありません。

今回の公開コード点検では新しい予測実験を追加せず、微量成分の数値処理、コマンドの入力・出力、配置後の実行を確認しています。論文v52の500件の追加ジョブ、144セルの集計値、良好・不良の両方の結果は維持します。

論文の比較では、金属で三方式の併用、酸化物で特徴量選択が有効な条件があり、高分子では従来条件が最良でした。既存の開発用データに対する比較であり、常に精度が改善することや、潜在座標が唯一の物理的相互作用係数であることを主張するものではありません。

詳しい仕様と式は[README.md](README.md)、実行手順は[PIPELINE.md](PIPELINE.md)、実行済み範囲は`pipeline_status.json`を参照してください。

## 引用・研究費

本研究コードを引用する場合は、ディレクトリ内の[CITATION.cff](CITATION.cff)を参照してください。これはDCEKit全体とは別に、本研究コードを示す引用情報です。確認済みの公開コミットと固定URLは[PUBLICATION_STATUS.md](PUBLICATION_STATUS.md)に記録しています。

研究費：JSPS KAKENHI **24K08152、24K01234**。
