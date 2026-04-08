# ParallelVC

本ディレクトリでは、パラレルデータとシンプルな3層MLPを用いたパラレル音声変換を作って学びます。

WORLD vocoderで抽出したメルケプストラム（MCEP）をDTWでアライメントし、フレーム単位でMLPにより変換元→変換先の特徴量マッピングを学習します。

## ディレクトリ構成

```
ParallelVC/
├── run.sh                 # 全ステージを順に実行するスクリプト
├── extract_features.py    # Stage 1: WORLD特徴量抽出
├── calculate_stats.py     # Stage 2: 統計量（F0・MCEP）の計算
├── dtw_alignment.py       # Stage 3: DTWアライメント
├── train.py               # Stage 4: モデル学習
├── eval.py                # Stage 5: 推論・評価
├── config/
│   └── config.yaml        # Hydra設定ファイル
├── src/
│   ├── audio.py           # 音声の読み込み・前処理
│   ├── world.py           # WORLD vocoder（特徴量抽出・合成）
│   ├── feature.py         # 特徴量の正規化・保存・読み込み
│   ├── dtw.py             # DTWアライメント
│   ├── data_split.py      # データ分割（train/val/test）
│   ├── dataset.py         # データセット
│   ├── model.py           # 3層MLPモデル
│   └── mcd.py             # Mel-Cepstral Distortion (MCD) 計算
├── pyproject.toml
└── uv.lock
```

## 環境設定

### 1. uvのインストール
uvのインストールを行います。すでに、インストールを完了している場合は、2.環境構築に進んでください。

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 環境構築

`uv sync` により、Python 3.10 のインストールと `pyproject.toml` に記載された全依存パッケージ（PyTorch, librosa, pyworld, hydra-core 等）のインストールが自動で行われます。

```bash
uv sync
```

### 3. JVSコーパスの配置

JVSコーパスが `corpus/jvs_ver1/` に配置されていることを確認してください。配置されていない場合は、[こちらのガイド](../README.md#3-jvsコーパスのダウンロードと配置)に従い、配置を完了してください。

## 実行方法

全ステージの実行は `run.sh` を通して行います。第1引数に開始ステージ、第2引数に終了ステージを指定します（省略時はデフォルトで Stage 1〜5 を一括実行）。

```bash
# 全ステージ一括実行（Stage 1〜5）
bash run.sh

# Stage 3 から Stage 5 まで実行
bash run.sh 3 5

# Stage 1 のみ実行
bash run.sh 1 1
```

以下では、各ステージの処理内容と出力ファイルを説明します。

### Stage 1: WORLD特徴量抽出

JVSコーパスの2話者（デフォルト: jvs001, jvs010）の音声から、WORLD vocoderを用いてF0（基本周波数）、スペクトル包絡、メルケプストラム（MCEP）、非周期性指標（AP）を抽出します。同時に、データの分割（train/val/test）も行います。

出力ファイル:

```
output/
├── split_info.json                        # データ分割情報（train/val/testのファイル名一覧）
└── feature/
    ├── jvs001/
    │   ├── train_features.npz             # 学習データの特徴量（F0, SP, MCEP, AP）
    │   ├── val_features.npz               # 検証データの特徴量
    │   └── test_features.npz              # 評価データの特徴量
    └── jvs010/
        ├── train_features.npz
        ├── val_features.npz
        └── test_features.npz
```

### Stage 2: 統計量の計算

Stage 1 で抽出した学習データの特徴量から、話者ごとの統計量（対数F0の平均・標準偏差、MCEPの平均・標準偏差）を計算します。これらは特徴量の正規化およびF0変換に使用されます。

出力ファイル:

```
output/feature/
├── jvs001/
│   └── stats.npz                          # 対数F0とMCEPの平均・標準偏差
└── jvs010/
    └── stats.npz
```

### Stage 3: DTWアライメント

Stage 1 で抽出したMCEP特徴量に対してDTW（Dynamic Time Warping）を適用し、変換元話者と変換先話者の特徴量の時間軸を揃えます。

出力ファイル:

```
output/feature/
├── jvs001/
│   ├── train_features_aligned.npz         # アライメント済み学習データ（F0, MCEP, AP）
│   ├── val_features_aligned.npz           # アライメント済み検証データ
│   └── test_features_aligned.npz          # アライメント済み評価データ
└── jvs010/
    ├── train_features_aligned.npz
    ├── val_features_aligned.npz
    └── test_features_aligned.npz
```

### Stage 4: 学習

DTWアライメント済みのMCEP特徴量を用いて、3層MLPモデルを学習します。学習の進捗はTensorBoardで確認できます。

出力ファイル:

```
output/
├── checkpoint/
│   ├── latest.pth                         # 最新エポックのモデル
│   └── epoch_0010.pth                     # 10エポックごとの定期保存（interval=10）
└── train_log/
    └── YYYYMMDD_HHMMSS/                   # TensorBoardログ（実行日時ごと）
        └── events.out.tfevents.*
```

TensorBoardで学習ログを確認する場合:

```bash
uv run tensorboard --logdir output/train_log
```

### Stage 5: 推論・評価

学習済みモデルを使って音声変換を実行し、MCD（Mel-Cepstral Distortion）による客観評価と変換音声の生成を行います。

出力ファイル:

```
output/eval/
└── VOICEID_jvs001_to_jvs010.wav           # 変換音声（評価データの各発話ごと）
```