#!/usr/bin/env bash

set -euo pipefail

# configファイルの選択
config=${1:-jvs_w_sr}

# 開始ステージを選択(0〜7)
stage=${2:-0}

# 終了ステージを選択(0〜7)
stop_stage=${3:-7}

echo "Starting from stage ${stage}"
if [ ${stop_stage} -ne -1 ]; then
  echo "Stopping at stage ${stop_stage}"
fi

# Stage 0: （とりあえず推論）事前学習済みのFreeVCモデルを用いて推論を実行
if [ ${stage} -le 0 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 0 ]; then
  echo "===== Stage 0: 事前学習済みのFreeVCモデルを用いてとりあえず推論 ====="
  uv run python convert.py --config-name="${config}" \
    path.trained_ckpt=./model/jvs_w_sr/G_50000.pth
fi

# Stage 1: JVSコーパスのダウンサンプリング
if [ ${stage} -le 1 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 1 ]; then
  echo "===== Stage 1: JVS コーパスを2種類のサンプリングレートにダウンサンプル ====="
  uv run python downsample_jvs.py --config-name="${config}"
fi

# Stage 2: 学習・検証・テストデータのファイルパスリストを作成
if [ ${stage} -le 2 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 2 ]; then
  echo "===== Stage 2: 学習・検証・テストデータのファイルパスリストを作成 ====="
  uv run python preprocess_flist_jvs.py --config-name="${config}"
fi

# Stage 3: 話者埋め込みの生成
if [ ${stage} -le 3 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 3 ]; then
  echo "===== Stage 3: 話者埋め込みの生成 ====="
  uv run python preprocess_spk.py --config-name="${config}"
fi

# Stage 4: SSL特徴量の生成
if [ ${stage} -le 4 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 4 ]; then
  echo "===== Stage 4: SSL特徴量の生成 ====="
  uv run python preprocess_ssl_jvs.py --config-name="${config}"
fi

# Stage 5: データ拡張の前処理
if [ ${stage} -le 5 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 5 ]; then
  echo "===== Stage 5: SRベースのデータ拡張 ====="
  uv run python preprocess_sr.py --config-name="${config}" preprocess.sr_augmentation.min=68 preprocess.sr_augmentation.max=72
  uv run python preprocess_sr.py --config-name="${config}" preprocess.sr_augmentation.min=73 preprocess.sr_augmentation.max=76
  uv run python preprocess_sr.py --config-name="${config}" preprocess.sr_augmentation.min=77 preprocess.sr_augmentation.max=80
  uv run python preprocess_sr.py --config-name="${config}" preprocess.sr_augmentation.min=81 preprocess.sr_augmentation.max=84
  uv run python preprocess_sr.py --config-name="${config}" preprocess.sr_augmentation.min=85 preprocess.sr_augmentation.max=88
  uv run python preprocess_sr.py --config-name="${config}" preprocess.sr_augmentation.min=89 preprocess.sr_augmentation.max=92
fi

# Stage 6: 音声変換モデル FreeVC の学習
if [ ${stage} -le 6 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 6 ]; then
  echo "===== Stage 6: 音声変換モデル FreeVC の学習 ====="
  uv run python train.py --config-name="${config}"
fi

# Stage 7: 学習したFreeVCモデルを用いて推論
if [ ${stage} -le 7 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 7 ]; then
  echo "===== Stage 7: 学習したFreeVCモデルを用いて推論 ====="
  uv run python convert.py --config-name="${config}"
fi

echo "Processing completed!"