
#!/usr/bin/env bash

set -euo pipefail

# 開始ステージを選択(1〜5)
stage=1

# 終了ステージを選択(1〜5)
stop_stage=1

# Stage 1: JVSコーパスの2話者の全データからWorld特徴量抽出
if [ ${stage} -le 1 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 1 ]; then
    python3 extract_features.py
fi

# Stage 3: stage1で抽出した特徴量に対して、f0変換を実施
if [ ${stage} -le 2 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 2 ]; then
    python3 calculate_stats.py
fi

# Stage 2: stage1で抽出した特徴量に対して、DTWを適用
if [ ${stage} -le 3 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 3 ]; then
    python3 dtw_alignment.py
fi

# Stage 4: 学習
if [ ${stage} -le 4 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 4 ]; then
    python3 train.py
fi

# Stage 5: 推論
if [ ${stage} -le 5 ] && [ ${stop_stage} -eq -1 -o ${stop_stage} -ge 5 ]; then
    python3 eval.py
fi