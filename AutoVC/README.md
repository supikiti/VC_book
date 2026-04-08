# AutoVC

本ディレクトリでは、オートエンコーダに基づくノンパラレル音声変換モデル「AutoVC」を動かしながら学びます。
本コードは、[公式実装リポジトリ](https://github.com/auspicious3000/autovc)をベースにしています。

## ディレクトリ構成

```
AutoVC/
├── run.sh                     # 全ステージを順に実行するスクリプト
├── make_spect.py              # Stage 1: メルスペクトログラム抽出
├── make_metadata_for_train.py # Stage 2: 学習用メタデータ生成
├── main.py                    # Stage 3: AutoVCモデル学習
├── make_metadata_for_eval.py  # Stage 4: 推論用メタデータ生成（話者埋め込み + スペクトログラム）
├── conversion.py              # Stage 5: AutoVCによる声質変換（推論）
├── vocoder.py                 # Stage 6: WaveNetボコーダによる波形合成
├── model_vc.py                # AutoVCモデル定義（Encoder + Decoder + Postnet）
├── model_bl.py                # 話者エンコーダ（d-vector, LSTM）
├── solver_encoder.py          # 学習ループ
├── data_loader.py             # データセット・データローダ
├── synthesis.py               # WaveNetでの波形合成
├── hparams.py                 # WaveNetハイパーパラメータ
├── config/
│   ├── vctk.yaml              # VCTK用設定ファイル
│   ├── jvs-full.yaml          # JVS（全話者）用設定ファイル
│   └── jvs-mini.yaml          # JVS（少数話者）用設定ファイル
├── model/                     # 事前学習済みモデル
│   ├── 3000000-BL.ckpt
│   ├── autovc.ckpt
│   └── checkpoint_step001000000_ema.pth
├── output/
│   ├── feature/{tag}/         # Stage 1 出力: メルスペクトログラム
│   ├── checkpoint/            # Stage 3 出力: 学習途中のモデル
│   └── eval/{tag}/            # Stage 6 出力: 変換音声
├── vctk-wavs/                 # サンプル音声（VCTKコーパス, 4話者）
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

`uv sync` により、Python 3.10 のインストールと `pyproject.toml` に記載された全依存パッケージ（PyTorch, librosa, soundfile 等）のインストールが自動で行われます。

```bash
uv sync
```

### 3. 事前学習済みモデルのダウンロードと配置

AutoVCの実行には、以下の3つの事前学習済みモデルが必要です。それぞれ下記のリンクからダウンロードし、`AutoVC/model/` ディレクトリに配置してください。

| ファイル名 | 説明 | ダウンロードリンク |
|---|---|---|
| `3000000-BL.ckpt` | 話者エンコーダ（d-vector）の事前学習済みモデル。話者埋め込みの抽出に使用します（Stage 2, 4）。 | [ダウンロード](https://ibm.box.com/s/k4nyvzkz26r9ux1o8rcq9tov36h69o2e) |
| `autovc.ckpt` | AutoVC本体の学習済みモデル。自分で学習する場合は不要です（Stage 5で使用）。 | [ダウンロード](https://ibm.box.com/s/upmfbstk8n8ded0xd3aaic4ayclh90ee) |
| `checkpoint_step001000000_ema.pth` | WaveNetボコーダの事前学習済みモデル。変換後のメルスペクトログラムから波形を合成するために使用します（Stage 6）。 | [ダウンロード](https://ibm.box.com/s/pdr8b0z5oim9q0p4be496mt3ue5lb004) |

配置後のディレクトリ構成が以下のようになっていることを確認してください。

```
AutoVC/
└── model/
    ├── 3000000-BL.ckpt
    ├── autovc.ckpt
    └── checkpoint_step001000000_ema.pth
```

> **注意**: IBM Box のリンクからダウンロードする際は、ブラウザでリンクを開き、ダウンロードボタンからファイルを取得してください。

## 実行方法

全ステージの実行は `run.sh` を通して行います。第1引数にコーパス名、第2引数に開始ステージ、第3引数に終了ステージを指定します（省略時はデフォルトで VCTK コーパスの Stage 1〜6 を一括実行）。

コーパス名には、`config/` ディレクトリ内のYAML設定ファイル名（拡張子なし）を指定します。

| コーパス名 | 設定ファイル | 説明 |
|---|---|---|
| `vctk` | `config/vctk.yaml` | VCTKコーパス（サンプル4話者、`sample/vctk` に同梱） |
| `jvs-full` | `config/jvs-full.yaml` | JVSコーパス全話者（`../corpus/jvs_ver1/` を参照） |
| `jvs-mini` | `config/jvs-mini.yaml` | JVSコーパス少数話者（`sample/jvs` を参照） |

```bash
# VCTK で全ステージ一括実行（Stage 1〜6）
bash run.sh

# JVS（全話者）で全ステージ一括実行
bash run.sh jvs-full

# JVS（少数話者）で Stage 1 から Stage 3 まで実行
bash run.sh jvs-mini 1 3

# VCTK で Stage 5 のみ実行
bash run.sh vctk 5 5
```

> **推論のみ行う場合**: 学習済みモデル（`model/autovc.ckpt` 等）を使って推論だけ行いたい場合は、Stage 2・3（学習用メタデータ生成・モデル学習）をスキップできます。Stage 1 でメルスペクトログラムを抽出した後、Stage 4 以降を実行してください。
>
> ```bash
> # Stage 1: メルスペクトログラム抽出
> bash run.sh vctk 1 1
>
> # Stage 4〜6: 推論用メタデータ生成 → 声質変換 → 波形合成
> bash run.sh vctk 4 6
> ```

以下では、各ステージの処理内容と出力ファイルを説明します。

### Stage 1: メルスペクトログラム抽出

設定ファイル`config/*.yaml`の`root_dir`で指定したディレクトリ内の各話者の音声ファイルから、80次元のメルスペクトログラムを抽出します。16kHzにリサンプリングされた音声に対してSTFTを適用し、メルフィルタバンクで変換します。

出力ファイル:

```
output/feature/{tag}/
├── spk1/
│   ├── spk1_001.npy          # メルスペクトログラム（numpy配列）
│   ├── spk1_002.npy
│   └── ...
├── spk2/
├── spk3/
└── spk4/
```

### (モデル学習を行う場合のみ実行してください) Stage 2: 学習用メタデータ生成

Stage 1 で生成したメルスペクトログラムから、事前学習済みの話者エンコーダ（d-vector）を用いて話者埋め込みベクトルを抽出します。各話者について複数発話の埋め込みを平均化し、全発話のファイルパスとともに学習用のメタデータファイル（`train.pkl`）を生成します。

> **注意**: このスクリプトの実行には `model/3000000-BL.ckpt` が必要です。

出力ファイル:

```
output/feature/{tag}/
└── train.pkl                 # メタデータ（話者名, 話者埋め込み, 発話ファイルパス...）
```

`train.pkl` の各エントリは `[話者名, 話者埋め込み(256,), ファイルパス1, ファイルパス2, ...]` の可変長リストです。学習時に DataLoader がファイルパスからスペクトログラムを都度読み込みます。

### (モデル学習を行う場合のみ実行してください) Stage 3: AutoVCモデル学習

`train.pkl`を用いてAutoVCを学習します。学習途中のチェックポイントは `output/checkpoint/` に10000イテレーションごとに保存されます。

出力ファイル:

```
output/checkpoint/
└── autovc_{iter}.ckpt        # 学習途中のチェックポイント（10000イテレーションごと）
```

### Stage 4: 推論用メタデータ生成

Stage 1 で生成したメルスペクトログラムから、事前学習済みの話者エンコーダ（d-vector）を用いて話者埋め込みベクトルを抽出します。各話者について全発話の埋め込みを平均化し、スペクトログラム1つとともに推論用のメタデータファイル（`eval.pkl`）を生成します。

> **注意**: このスクリプトの実行には `model/3000000-BL.ckpt` が必要です。

出力ファイル:

```
output/feature/{tag}/
└── eval.pkl                  # メタデータ（話者名, 話者埋め込み, スペクトログラム）
```

`eval.pkl` の各エントリは `[話者名, 話者埋め込み(256,), スペクトログラム(T,80)]` の固定3要素リストです。Stage 2 の学習用メタデータとは異なり、スペクトログラムの実データを直接保持しており、Stage 5 の変換処理でそのまま使用されます。

### Stage 5: AutoVCによる声質変換（推論）

学習済みのAutoVCモデルを使って、話者間の音声変換を実行します。変換元話者のメルスペクトログラムをエンコーダに入力し、変換先話者の埋め込みを結合してデコーダで復元することで、変換先話者の声質を持つメルスペクトログラムを生成します。

> **注意**: このスクリプトの実行には学習済みモデル `model/autovc.ckpt` が必要です。Stage 3 で自分で学習したモデルを使用する場合は、設定ファイル `config/*.yaml` の `path.autovc_checkpoint` を学習済みチェックポイントのパスに変更してください。

出力ファイル:

```
output/feature/{tag}/
└── results.pkl               # 変換後のメルスペクトログラム（全話者ペア）
```

### Stage 6: WaveNetボコーダによる波形合成

Stage 5 で得られた変換後のメルスペクトログラム（`results.pkl`）から、事前学習済みのWaveNetボコーダを用いて音声波形を合成します。

> **注意**: このスクリプトの実行には `model/checkpoint_step001000000_ema.pth` が必要です。

出力ファイル:

```
output/eval/{tag}/
├── p225xp228.wav             # 変換音声（話者ペアごとに1ファイル）
├── p226xp225.wav
└── ...
```
