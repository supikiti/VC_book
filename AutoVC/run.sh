#!/usr/bin/env bash

set -euo pipefail

# コーパスの選択(jvs or vctk)
corpus=${1:-vctk}

# 開始ステージを選択(1〜5)
stage=${2:-1}

# 終了ステージを選択(1〜5)
stop_stage=${3:-5}

# Stage 1: VCTKコーパスかJVSコーパスの波形データからメルスペクトログラムを取得
if [ ${stage} -le 1 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 1 ]; then
    uv run python make_spect.py --config-name="${corpus}"
fi

# 学習を行いたい場合は、Stage 2・3 を実行してください。

# Stage 2: Stage1 で抽出した特徴量に対して、学習用のメタデータを作成
if [ ${stage} -le 2 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 2 ]; then
    uv run python make_metadata_for_train.py --config-name="${corpus}"
fi

# Stage 3: Stage2で作成したメタデータを用いて、AutoVCを学習
if [ ${stage} -le 3 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 3 ]; then
    uv run python main.py --config-name="${corpus}"
fi

# ここから、推論用のStage

# Stage 4: 評価用のメタデータを作成
if [ ${stage} -le 4 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 4 ]; then
    uv run python make_metadata_for_eval.py --config-name="${corpus}"
fi

# Stage 5: Stage4で作成した評価用メタデータを用いて、AutoVCによる声質変換（推論）
if [ ${stage} -le 5 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 5 ]; then
    uv run python conversion.py --config-name="${corpus}"
fi

# Stage 6: Stage5で変換したスペクトログラムを波形に変換
if [ ${stage} -le 6 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 6 ]; then
    uv run python vocoder.py --config-name="${corpus}"
fi
