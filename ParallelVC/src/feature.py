from collections import defaultdict
from pathlib import Path

import numpy as np

EPS = 1e-8


def normalize_features(
    features: np.ndarray, stats: dict[str, float] | None = None
) -> tuple[np.ndarray, dict[str, float]]:
    """
    特徴量の正規化

    Args:
        features: 特徴量（T x D）
        stats: 正規化統計量（平均・標準偏差）

    Returns:
        normalized_features: 正規化後特徴量
        stats: 正規化統計量
    """
    if stats is None:
        stats = {"mean": np.mean(features, axis=0), "std": np.std(features, axis=0)}
    normalized_features = (features - stats["mean"]) / (stats["std"] + EPS)
    return normalized_features, stats


def denormalize_features(
    normalized_features: np.ndarray, stats: dict[str, float]
) -> np.ndarray:
    """
    特徴量の逆正規化

    Args:
        normalized_features: 正規化済み特徴量 （T x D）
        stats: 正規化統計量

    Returns:
        denormalized_features: 逆正規化後特徴量
    """
    return normalized_features * stats["std"] + stats["mean"]


def load_features(
    input_path: Path,
    split_char: str = "/",
) -> dict[str, dict[str, np.ndarray]]:
    """
    NPZ形式の特徴量を読み込み

    Args:
        input_path: 入力ファイルパス
        split_char: キーの分割文字

    Returns:
        features: 特徴量辞書
    """
    loaded_data = np.load(input_path, allow_pickle=True)
    features = defaultdict(dict)
    for flat_key in loaded_data.files:
        # キーを "file_name/feature_key" の形式から分割
        file_name, feature_key = flat_key.split(split_char)
        features[file_name][feature_key] = loaded_data[flat_key]
    return features


def save_features(
    output_path: Path,
    features: dict[str, dict[str, np.ndarray]],
    split_char: str = "/",
) -> None:
    """
    特徴量をNPZ形式で保存

    Args:
        output_path: 出力ファイルパス
        features: 特徴量辞書
        split_char: キーの分割文字
    """
    flat_data = {}
    for file_name, feats in features.items():
        for feature_key, data in feats.items():
            # キーを "file_name/feature_key" の形式にフラット化
            # 例: "jvs001_001.wav/f0"
            flat_key = f"{file_name}{split_char}{feature_key}"
            flat_data[flat_key] = data
    np.savez_compressed(output_path, **flat_data)
