import torch
import torch.nn as nn


class MLPConversionModel(nn.Module):
    """
    3層のMLPで構成される音声変換モデル。
    メルケプストラム特徴量をフレーム単位で変換します。
    時系列情報は考慮しない基本的な実装です。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.2,
    ):
        """
        Args:
            input_dim: 入力次元（メルケプストラム次数-1）
            hidden_dim: 隠れ層次元数
            output_dim: 出力次元（メルケプストラム次数-1）
            dropout_rate: ドロップアウト率
        """
        super().__init__()
        self.layers = nn.Sequential(
            # 第1層: 入力 → 隠れ層1
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            # 第2層: 隠れ層1 → 隠れ層2
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            # 第3層: 隠れ層2 → 出力
            nn.Linear(hidden_dim, output_dim),
        )

        # パラメータ初期化
        self._initialize_weights()

    def _initialize_weights(self):
        """重み初期化"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Xavier uniform初期化
                nn.init.xavier_uniform_(module.weight)
                # バイアスを0で初期化
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 入力メルケプストラム

        Returns:
            output: 変換後メルケプストラム
        """
        return self.layers(x)
