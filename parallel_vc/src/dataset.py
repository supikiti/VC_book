from pathlib import Path

import numpy as np
import torch
from src.feature import load_features, normalize_features
from torch.utils.data import Dataset


class VoiceConversionDataset(Dataset):
    """
    音声変換用のデータセットクラス
    DTW済みメルケプストラム特徴量をフレーム単位で提供します
    """

    def __init__(
        self,
        source_features_path: Path,
        target_features_path: Path,
        source_features_stats_path: Path,
        target_features_stats_path: Path,
    ):
        """
        Args:
            source_features_path: 変換元話者特徴量のNPZファイルパス
            target_features_path: 変換先話者特徴量のNPZファイルパス
            source_features_stats_path: 変換元話者特徴量の統計量ファイルパス
            target_features_stats_path: 変換先話者特徴量の統計量ファイルパス
        """
        # 特徴量の読み込み
        source_features = load_features(source_features_path)
        target_features = load_features(target_features_path)

        # 全発話のmcepを連結
        self.source_mcep = np.concatenate(
            [feats["mcep"] for feats in source_features.values()], axis=0
        )
        self.target_mcep = np.concatenate(
            [feats["mcep"] for feats in target_features.values()], axis=0
        )
        assert (
            len(self.source_mcep) == len(self.target_mcep)
        ), "元話者のMCEPと目標話者のMCEPのフレーム数が一致しません\nDTWアライメントを確認してください"

        # 統計量の読み込み
        self.source_stats = load_features(source_features_stats_path)
        self.target_stats = load_features(target_features_stats_path)

    def __len__(self) -> int:
        """データセットサイズ"""
        return len(self.source_mcep)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        正規化不適用および正規化適用済みメルケプストラムを返す

        Args:
            idx: データインデックス

        Returns:
            source_mcep: 変換元話者MCEPテンソル
            target_mcep: 変換先話者MCEPテンソル
            source_mcep_normalized: 正規化済み変換元話者MCEPテンソル
            target_mcep_normalized: 正規化済み変換先話者MCEPテンソル
        """
        # データ取得
        source_mcep = self.source_mcep[idx]
        target_mcep = self.target_mcep[idx]

        # 正規化
        source_mcep_normalized, _ = normalize_features(
            source_mcep, self.source_stats["mcep"]
        )
        target_mcep_normalized, _ = normalize_features(
            target_mcep, self.target_stats["mcep"]
        )

        # PyTorchのテンソルに変換
        source_mcep = torch.tensor(source_mcep, dtype=torch.float32)
        target_mcep = torch.tensor(target_mcep, dtype=torch.float32)
        source_mcep_normalized = torch.tensor(
            source_mcep_normalized, dtype=torch.float32
        )
        target_mcep_normalized = torch.tensor(
            target_mcep_normalized, dtype=torch.float32
        )

        return source_mcep, target_mcep, source_mcep_normalized, target_mcep_normalized
