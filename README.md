# 実験3：全モデル全県実行環境（実験２はサブフォルダ）

クマ出没の有無を 3 次メッシュ単位で予測する実験コード一式です。
東北 6 県のデータを用い、決定木・ロジスティック回帰・LightGBM、CNN系手法（CNN、CNN-DANN）を同一環境で比較します。

## 構成

| ファイル | 内容 |
| --- | --- |
| `実験3-宮城県-ResCNN.ipynb` | ResNet18 ベース CNN による予測（県別） |
| `実験3-宮城県-logreg.ipynb` | ロジスティック回帰による予測（県別） |
| `ベースライン実験３DT.ipynb` | 決定木（Decision Tree）ベースライン |
| `ベースライン実験３LGBM.ipynb` | LightGBM ベースライン |
| `ベースライン実験３LGBM_seed固定.ipynb` | シード固定版 LightGBM|
| `CNN-DANN.ipynb` | ドメイン敵対的学習（DANN）付き CNN |
| `CNNtoolbox.py` | メッシュコード→緯度経度変換、GeoJSON のラスタ化、メトリクス保存などの共通関数 |
| `run_one.sh` | ノートブックを nbconvert でヘッドレス実行するスクリプト |

### 生成物の出力先

| ディレクトリ | 内容 |
| --- | --- |
| `optuna_results/` | Optuna 探索の各試行スコア（CSV） |
| `results/` | モデル別の予測結果（CSV） |
| `optuna_models/` | 学習済みモデル |
| `optuna_pickeles/` | 学習曲線・各種メトリクスの pickle |
| `logs/` | `run_one.sh` の実行ログ |

> `（参考）実験２の環境` は実験2のコードを参考用に同梱したものです。

## 必要な入力データ

### `dataset/` に配置（ノートブックは `dataset/...` を参照）

- `くま発見3次メッシュデータセット_東北全県.csv` … 教師データ（メッシュ別の出没有無）
- `dataset_tensor_CNN.pt` … CNN / ロジスティック回帰用の入力テンソル
- `dataset_tensor_DANN.pt` … CNN-DANN 用の入力テンソル
- `人口_4528_6242_Japan.tif`, `平均標高_..._.tif`, `最大傾斜角度_..._.tif`,
  `最大傾斜方向_..._.tif`, `平均傾斜角度_人口_..._.tif` … 人口・標高・傾斜などのラスタ（DT / LGBM 用）
- `現存植生図2024_2020_東北ブロック.geojson`, `..._2022_..._.geojson` … 植生（GeoJSON、DT / LGBM 用）
- `衛星写真13/<meshcode>.png` … メッシュコード別の衛星写真（DT / LGBM 用）

## 環境構築

### conda を使う場合

```bash
conda env create -f environment.yaml
conda activate kuma-exp3
```

### pip を使う場合

```bash
pip install -r requirements.txt
```

GPU（CUDA）で PyTorch を使う場合は、環境に合わせて公式の手順で `torch` を導入してください。
動作確認は **Python 3.11.7 / PyTorch 2.5.1 + CUDA 12.4** で行っています。

### 動作確認環境（Jupyter カーネル `pytorch_env`）

`environment.yaml` / `requirements.txt` のバージョンは、`run_one.sh` が指定する
Jupyter カーネル `pytorch_env`（実体は `/home/hasegawakazuki/anaconda3/bin/python`）の
実環境から取得したものです。主なバージョンは以下のとおりです。

| パッケージ | バージョン |
| --- | --- |
| python | 3.11.7 |
| torch | 2.5.1+cu124 (CUDA 12.4) |
| numpy | 2.2.6 |
| pandas | 3.0.1 |
| scikit-learn | 1.7.2 |
| optuna | 4.5.0 |
| lightgbm | 4.6.0 |
| geopandas | 1.1.1 |
| rasterio | 1.4.3 |
| shapely | 2.1.2 |
| affine | 2.4.0 |
| tifffile | 2023.4.12 |
| pillow | 12.1.1 |
| joblib | 1.5.2 |
| matplotlib | 3.10.5 |

## 実行方法

### Jupyter 上で対話的に実行

各ノートブックを開いて上から順に実行します。

### ヘッドレス（バックグラウンド）実行

SSH 切断後も継続させたい場合は `run_one.sh` を使います。

```bash
bash run_one.sh "ベースライン実験３LGBM.ipynb"
```

実行ログは `logs/<ノートブック名>.log`、実行済みノートブックは `_executed/` に出力されます。

> 注意: `run_one.sh` 内の `DIR`（作業ディレクトリ）と
> `--ExecutePreprocessor.kernel_name`（Jupyter カーネル名）は実行環境に合わせて書き換えてください。
