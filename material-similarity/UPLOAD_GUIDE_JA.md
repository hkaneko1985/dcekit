# GitHubへのアップロード（改訂版 0.2.2）

推奨ディレクトリ名は **`material-similarity`** です。Pythonでのimport名は `material_similarity` です。

ZIPを展開すると `material-similarity/` ができます。既存リポジトリに追加する場合は、このフォルダをそのまま配置してください。専用の新規リポジトリを作る場合は、このフォルダの中身をリポジトリ直下に配置できます。どちらの場合もREADMEのコマンドは `material-similarity` の内容があるディレクトリ内で実行します。

## 二つのZIPの用途

| ZIP | 用途・収録範囲 |
| --- | --- |
| `material-similarity_github_v0.2.2.zip` | GitHub公開準備用。独自コード、現在の図表・結果、出典表示付きの収録データ、選択・ハッシュ情報、復元手順を含みます。NanoMineのAPIスナップショットと詳細な原値監査表は除外しています。 |
| `material-similarity_author_reproduction_v0.2.2.zip` | 著者の再解析用。添付されたNanoMine研究入力とその監査表を保持し、現行解析を再実行できます。そのまま公開する用途ではありません。 |

NanoMineのAPIデータは、MaterialsMineのソフトウェアライセンスの適用対象外と明記されています。データ自体の再配布許諾は確認できていません。公開用ZIPに、著者用ZIPの内容を無条件に追加しないでください。除外対象は `data/external_data_manifest.json` に列挙しています。

## 主な収録内容

- HTEM：1,891ライブラリの正規化済み組成・成膜設定。
- Starrydata：選択済み5,001件の特徴量・クラスター割当。
- PNCExtract：1,103件の構成材料・含有量・文献情報。
- NanoMine：著者用のみ832試料の研究入力。両ZIPに選択情報・ハッシュ・独自解析結果を収録。
- 共通API 0.2.2、数値・単位辞書、多重設定値の保持、欠損・工程対応の感度解析、同じ情報を用いた比較評価。
- 主要解析13ファイルと追加評価18ファイルの参照結果、Figure 1〜7・S1〜S3・TOC。
- 歴史的なPhase 1〜4のコード・設定・結果。現在の共通APIと同じ解析であるとは扱いません。

Starrydataの抽出前全件CSVと元のHTEM取得キャッシュは含まれません。全資源の取得からの完全再現とは区別しています。詳細は `upstream_analyses/README.md` と `docs/FIGURE_REPRODUCTION.csv` を参照してください。

## 実行

Python 3.12の環境を推奨します。Windowsでの例です。

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install --no-deps -e .
.venv\Scripts\python.exe run_all.py --tables-only
```

公開用ZIPだけの場合は `--tables-only` を指定して、保存結果の検査と作図を実行します。新たなデータ解析ではなく、NanoMine入力に依存するテストは明示的にスキップします。

著者用ZIPの場合は `python run_all.py` で39件のチェック、主要解析の13ファイル照合、追加評価の18ファイル照合と作図を実行します。再計算は `outputs/`、参照結果は `results/` に保存されます。実行確認環境はLinux/Python 3.12です。

適切な権限で保持している同一スナップショットを公開用パッケージにローカル復元する場合：

```bash
python src/restore_external_data.py --archive /path/to/authorized/archive.zip
python run_all.py
```

復元はハッシュが一致するファイルに限定され、欠損値補完は行いません。公開APIの確認では2026年9月15日にHTTP 503が返り、現在のライブ取得だけで凍結スナップショットを再現できるとは保証していません。

## 公開情報

独自コード・独自文書のMIT Licenseは第三者データへ一律には適用しません。`NOTICE.md`、`data/SOURCES_AND_LICENSES.md`、`LICENSES/` の各資源の通知を保持してください。公開先URL、リリース・コミット、DOIは実際の公開後に `CITATION.cff` と論文へ追記します。未発行の識別子は記載していません。

検証記録は `docs/VERIFICATION.md`、査読に伴う変更は `docs/REVISION_NOTES.md`、ZIPごとの収録範囲は `BUNDLE_SCOPE.md`、ファイルハッシュは `MANIFEST.sha256` にあります。

## 再査読の残件

著者用ZIPは今回の返却物に含みますが、査読者への限定共有や公開の許諾が得られたとは扱いません。ローカル復元は初回入手経路ではありません。新しい査読者が凍結入力へ到達するには、利用条件の確認と実際に機能する提供経路が必要です。詳しくは `docs/DATA_ACCESS_STATUS.md` を参照してください。
