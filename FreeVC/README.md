# FreeVC

本ディレクトリでは、ゼロショット音声変換モデル「FreeVC」を動かしながら学びます。

本コードは、[FreeVC公式実装リポジトリ](https://github.com/OlaWod/FreeVC)をベースにしています。

## ディレクトリ構成

```
FreeVC/
├── run.sh                     # 全ステージを順に実行するスクリプト
├── convert.py                 # Stage 0, 7: 音声変換（推論）
├── downsample_jvs.py          # Stage 1: JVSコーパスのダウンサンプリング
├── preprocess_flist_jvs.py    # Stage 2: ファイルリスト作成
├── preprocess_spk.py          # Stage 3: 話者埋め込み抽出
├── preprocess_ssl_jvs.py      # Stage 4: WavLM特徴量抽出
├── preprocess_sr.py           # Stage 5: SRデータ拡張
├── train.py                   # Stage 6: FreeVCモデル学習
├── config/
│   └── jvs_w_sr.yaml          # Hydra設定ファイル
├── src/
│   ├── commons.py
│   ├── data_utils.py          # データセット・データローダ
│   ├── losses.py              # 損失関数
│   ├── mel_processing.py      # メルスペクトログラム処理
│   ├── models.py              # FreeVCモデル
│   ├── modules.py
│   └── utils.py               # チェックポイント管理・WavLMロード等
├── speaker_encoder/           # 話者エンコーダを管理するフォルダ
│   ├── voice_encoder.py
│   ├── audio.py
│   └── ckpt/
│       └── pretrained_bak_5805000.pt
├── wavlm/                     # WavLMを管理するフォルダ
│   ├── WavLM.py
│   ├── modules.py
│   └── WavLM-Large.pt
├── hifigan/                   # HiFiGANを管理するフォルダ
│   ├── __init__.py
│   ├── models.py
│   ├── config.json
│   └── g_02500000
├── model/                     # 事前学習済みFreeVCモデル
│   └── jvs_w_sr/
│       └── G_50000.pth
├── convert_jvs.txt            # 推論対象の音声ファイルリスト
├── pyproject.toml
└── uv.lock
```

## 環境設定

### 1. uvのインストール

uvのインストールを行います。すでにインストールを完了している場合は、2.環境構築に進んでください。

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 2. 環境構築

`uv sync` により、Python 3.10 のインストールと `pyproject.toml` に記載された全依存パッケージ（PyTorch, librosa, hydra-core 等）のインストールが自動で行われます。

```bash
uv sync
```

### 3. JVSコーパスの配置

JVSコーパスが `../corpus/jvs_ver1/` に配置されていることを確認してください。配置されていない場合は、[こちらのガイド](../README.md#3-jvsコーパスのダウンロードと配置)に従い、配置を完了してください。

### 4. 事前学習済みモデルのダウンロードと配置

FreeVCの実行には、以下の事前学習済みモデルが必要です。それぞれダウンロードし、所定のディレクトリに配置してください。

#### WavLM-Large（コンテンツ特徴量の抽出に使用）

[WavLMの公式リポジトリ](https://github.com/microsoft/unilm/tree/master/wavlm)にアクセスし、**WavLM-Large** のチェックポイントをダウンロードしてください。ダウンロードしたファイルを `wavlm/` ディレクトリに配置します。

```
FreeVC/
└── wavlm/
    └── WavLM-Large.pt
```

#### HiFiGAN v1（SR拡張時の音声合成に使用）

[HiFiGANの公式リポジトリ](https://github.com/jik876/hifi-gan)にアクセスし、**UNIVERSAL V1** のGenerator（`g_02500000`）と設定ファイル（`config.json`）をダウンロードしてください。ダウンロードしたファイルを `hifigan/` ディレクトリに配置します。

```
FreeVC/
└── hifigan/
    ├── config.json
    └── g_02500000
```

> **注意**: HiFiGANのモデルは Stage 5（SRデータ拡張）で使用します。Stage 5 をスキップする場合（`use_sr: false` の設定で学習する場合）は不要です。

#### FreeVC事前学習済みモデル（推論のみ行う場合）

事前学習済みのFreeVCモデルを使用して推論のみ行いたい場合は、モデルファイルを[こちら](https://drive.google.com/file/d/1JtlpAm2MnhBntFIv2rX1R5E5hFJoEy6C/view?usp=sharing)からダウンロードし、`model/jvs_w_sr/` ディレクトリに配置してください。

```
FreeVC/
└── model/
    └── jvs_w_sr/
        └── G_50000.pth
```

> **注意**: 自分で一から学習する場合は、この事前学習済みモデルは不要です。

## 実行方法

全ステージの実行は `run.sh` を通して行います。第1引数に設定ファイル名、第2引数に開始ステージ、第3引数に終了ステージを指定します（省略時はデフォルトで `jvs_w_sr` 設定の Stage 0〜7 を一括実行）。

設定ファイル名には、`config/` ディレクトリ内のYAML設定ファイル名（拡張子なし）を指定します。

```bash
# デフォルト設定で全ステージ一括実行（Stage 0〜7）
bash run.sh

# Stage 0 のみ実行（事前学習済みモデルで推論）
bash run.sh jvs_w_sr 0 0

# Stage 1 から Stage 5 まで実行（前処理のみ）
bash run.sh jvs_w_sr 1 5

# Stage 6 のみ実行（学習）
bash run.sh jvs_w_sr 6 6
```

> **まずは変換音声を聴いてみたい方へ**: 事前学習済みFreeVCモデルと WavLM-Large、話者エンコーダをダウンロード・配置した後、**Stage 0** を実行するだけで変換音声を生成できます。
>
> ```bash
> bash run.sh jvs_w_sr 0 0
> ```

以下では、各ステージの処理内容と出力ファイルを説明します。

### Stage 0: 事前学習済みモデルで推論

事前学習済みのFreeVCモデルを使用して音声変換を実行します。`convert_jvs.txt` に記載された変換ペアについて、変換元話者のコンテンツを変換先話者の声質に変換します。

> **必要なモデル**: WavLM-Large、話者エンコーダ、事前学習済みFreeVCモデル

出力ファイル:

```
output/eval/
├── JVS001to010.wav            # 変換音声（話者ペアごとに1ファイル）
├── JVS004to006.wav
└── ...
```

### Stage 1: JVSコーパスのダウンサンプリング

JVSコーパスの音声を16kHzと22.05kHzにダウンサンプリングします。無音区間のトリミングと音量正規化も行います。

出力ファイル:

```
output/feature/
├── jvs-16k/                   # 16kHzにダウンサンプリングした音声
│   ├── jvs001/
│   │   ├── BASIC5000_0518.wav
│   │   └── ...
│   └── ...
└── jvs-22k/                   # 22.05kHzにダウンサンプリングした音声
    ├── jvs001/
    └── ...
```

### Stage 2: ファイルリスト作成

Stage 1 で生成した16kHz音声から、学習・検証・テスト用のファイルリストを作成します。

出力ファイル:

```
output/filelists/
├── jvs_train.txt              # 学習データ一覧
├── jvs_val.txt                # 検証データ一覧
└── jvs_test.txt               # テストデータ一覧
```

### Stage 3: 話者埋め込み抽出

事前学習済みの話者エンコーダ（d-vector）を用いて、各発話の話者埋め込みベクトル（256次元）を抽出します。

出力ファイル:

```
output/feature/spk/
├── jvs001/
│   ├── BASIC5000_0518.npy     # 話者埋め込み（256次元）
│   └── ...
└── ...
```

### Stage 4: WavLM特徴量抽出

事前学習済みのWavLM-Largeを用いて、各発話から自己教師あり音声特徴量（1024次元）を抽出します。

> **必要なモデル**: WavLM-Large（`wavlm/WavLM-Large.pt`）

出力ファイル:

```
output/feature/wavlm/
├── jvs001/
│   ├── BASIC5000_0518.pt      # WavLM特徴量
│   └── ...
└── ...
```

### Stage 5: SRデータ拡張

Spectrogram Resampling（SR）によるデータ拡張を行います。メルスペクトログラムの高さを変化させてHiFiGANで再合成し、その音声からWavLM特徴量を抽出します。

> **必要なモデル**: WavLM-Large、HiFiGAN v1（`hifigan/config.json`, `hifigan/g_02500000`）

出力ファイル:

```
output/feature/sr/
├── wav/                       # SR拡張された音声
│   └── jvs001/
│       ├── BASIC5000_0518_68.wav
│       └── ...
└── wavlm/                     # SR拡張音声のWavLM特徴量
    └── jvs001/
        ├── BASIC5000_0518_68.pt
        └── ...
```

### Stage 6: FreeVCモデル学習

前処理済みのデータを用いてFreeVCモデル（Generator + Discriminator）を学習します。マルチGPU分散学習（DDP）に対応しています。チェックポイントは `eval_interval`（デフォルト: 10000ステップ）ごとに保存されます。

> **注意**: GPU（CUDA）が必要です。

出力ファイル:

```
logs/jvs_w_sr/
├── G_10000.pth                # Generatorチェックポイント
├── D_10000.pth                # Discriminatorチェックポイント
├── config.json                # 学習時の設定
└── train.log                  # 学習ログ
```

### Stage 7: 学習済みモデルで推論

Stage 6 で学習したモデルを使用して音声変換を実行します。

出力ファイル:

```
output/eval/
├── JVS001to010.wav
├── JVS004to006.wav
└── ...
```

## 引用

[FreeVCの公式リポジトリ](https://github.com/OlaWod/FreeVC)からコードの大部分で引用を行いました。