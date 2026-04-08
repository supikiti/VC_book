# SeedVC

本ディレクトリでは、フローマッチングに基づくゼロショット音声変換モデル「Seed-VC」を学びます。

本コードは、[Seed-VC公式実装リポジトリ](https://github.com/Plachtaa/seed-vc)をベースにしています。

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

`uv sync` により、Python 3.10 のインストールと `pyproject.toml` に記載された全依存パッケージ（PyTorch, transformers, gradio 等）のインストールが自動で行われます。

**PyTorchのインストール元がOS環境によって異なるため、以下のように `--extra` オプションを指定してください。**

```bash
# macOS の場合
uv sync --extra mac

# Windows / Linux（CUDA GPU搭載）の場合
uv sync --extra cuda
```

> **注意**: `--extra mac` と `--extra cuda` は排他的です。同時に指定することはできません。

### 3. JVSコーパスの配置

JVSコーパスが `../corpus/jvs_ver1/` に配置されていることを確認してください。配置されていない場合は、[こちらのガイド](../README.md#3-jvsコーパスのダウンロードと配置)に従い、配置を完了してください。

## 実行方法

Seed-VCでは、事前学習済みモデルが初回実行時にHuggingFaceから自動でダウンロードされます。手動でのモデルダウンロードは不要です。

> **注意**: ネットワーク環境によりHuggingFaceにアクセスできない場合は、コマンドの先頭に `HF_ENDPOINT=https://hf-mirror.com` を付けてください。

### 歌声変換デモ（Gradio UI）

ブラウザ上で操作できる歌声変換デモを起動します。ソース音声と参考音声をアップロードし、声質変換を行うことができます。

```bash
uv run python app_svc.py
```

起動後、ターミナルに表示されるURL（通常 `http://127.0.0.1:7860`）をブラウザで開いてください。

主なオプション:

| オプション | デフォルト | 説明 |
|---|---|---|
| `--fp16` | `True` | 半精度浮動小数点（FP16）の使用 |
| `--gpu` | `0` | 使用するGPUのID |
| `--share` | `False` | Gradioの共有リンクを生成 |

### リアルタイム音声変換（GUI）

デスクトップGUIによるリアルタイム音声変換を起動します。マイク入力をリアルタイムで変換して出力します。

```bash
uv run python real-time-gui.py
```

主なオプション:

| オプション | デフォルト | 説明 |
|---|---|---|
| `--fp16` | `True` | 半精度浮動小数点（FP16）の使用 |
| `--gpu` | `0` | 使用するGPUのID |

## 引用

[Seed-VCの公式リポジトリ](https://github.com/Plachtaa/seed-vc)からコードの大部分で引用を行いました。
