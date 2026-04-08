
#!/usr/bin/env bash

set -euo pipefail

# 開始ステージを選択(1〜5)
stage=${1:-1}

# 終了ステージを選択(1〜5)
stop_stage=${2:-5}

# Stage 1: JVSコーパスの2話者の全データからWorld特徴量抽出
if [ ${stage} -le 1 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 1 ]; then
    uv run python extract_features.py
fi

# Stage 3: stage1で抽出した特徴量に対して、f0変換を実施
if [ ${stage} -le 2 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 2 ]; then
    uv run python calculate_stats.py
fi

# Stage 2: stage1で抽出した特徴量に対して、DTWを適用
if [ ${stage} -le 3 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 3 ]; then
    uv run python dtw_alignment.py
fi

# Stage 4: 学習
if [ ${stage} -le 4 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 4 ]; then
    uv run python train.py
fi

# Stage 5: 推論
if [ ${stage} -le 5 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 5 ]; then
    uv run python eval.py
fi