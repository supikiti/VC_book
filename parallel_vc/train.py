import random
import warnings
from copy import deepcopy
from pathlib import Path

warnings.filterwarnings("ignore")

from datetime import datetime

import hydra
import numpy as np

# モデルモジュール
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from src.dataset import VoiceConversionDataset
from src.feature import denormalize_features

# 自作モジュール
from src.mcd import calculate_mcd
from src.model import MLPConversionModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class VoiceConversionTrainer:
    """
    音声変換モデルの学習クラス
    """

    def __init__(self, config: DictConfig):
        """
        Args:
            config: Hydra設定
        """
        self.config = config
        self.seed_everything(config.random_seed)

        # デバイス設定
        self.device = config.train.device
        if self.device == "cuda" and not torch.cuda.is_available():
            print("CUDAが利用できません。CPUにフォールバックします。")
            self.device = "cpu"
        if self.device == "mps" and not torch.backends.mps.is_available():
            print("MPSが利用できません。CPUにフォールバックします。")
            self.device = "cpu"

        # 出力ディレクトリを作成
        self.feature_dir = (
            Path(config.path.output_base_dir) / config.path.output_tag.feature
        )
        self.checkpoint_dir = (
            Path(config.path.output_base_dir) / config.path.output_tag.checkpoint
        )
        self.train_log_dir = (
            Path(config.path.output_base_dir)
            / config.path.output_tag.train_log
            / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.train_log_dir.mkdir(parents=True, exist_ok=True)
        print(f"モデルは: {self.checkpoint_dir} に保存されます")
        print(f"学習時のログは: {self.train_log_dir} に保存されます")

        # TensorBoard設定
        self.writer = SummaryWriter(self.train_log_dir)

        # データセット作成
        train_dataset = VoiceConversionDataset(
            source_features_path=(
                self.feature_dir / config.source_speaker / "train_features_aligned.npz"
            ),
            target_features_path=(
                self.feature_dir / config.target_speaker / "train_features_aligned.npz"
            ),
            source_features_stats_path=(
                self.feature_dir / config.source_speaker / "stats.npz"
            ),
            target_features_stats_path=(
                self.feature_dir / config.target_speaker / "stats.npz"
            ),
        )
        val_dataset = VoiceConversionDataset(
            source_features_path=(
                self.feature_dir / config.source_speaker / "train_features_aligned.npz"
            ),
            target_features_path=(
                self.feature_dir / config.target_speaker / "train_features_aligned.npz"
            ),
            source_features_stats_path=(
                self.feature_dir / config.source_speaker / "stats.npz"
            ),
            target_features_stats_path=(
                self.feature_dir / config.target_speaker / "stats.npz"
            ),
        )

        # MCD計算用にターゲット話者の統計量を保存
        # 0次元目は不要なので削除
        self.target_stats = deepcopy(val_dataset.target_stats)
        self.target_stats["mcep"]["mean"] = self.target_stats["mcep"]["mean"][1:]
        self.target_stats["mcep"]["std"] = self.target_stats["mcep"]["std"][1:]

        # データローダー作成
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=self.config.train.shuffle,
            num_workers=self.config.train.num_workers,
            pin_memory=self.config.train.pin_memory,
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=128,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )

        # モデルのインスタンス化
        self.model = MLPConversionModel(
            input_dim=self.config.data.mcep_order - 1,  # 0次元目を除外
            output_dim=self.config.data.mcep_order - 1,  # 0次元目を除外
            hidden_dim=self.config.train.model.hidden_dim,
            dropout_rate=self.config.train.model.dropout_rate,
        ).to(self.device)

        # オプティマイザの設定（Adam）
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.train.optimizer.learning_rate,
            weight_decay=config.train.optimizer.weight_decay,
            betas=config.train.optimizer.betas,
        )

        # 損失関数は平均二乗誤差（MSE）
        self.criterion = nn.MSELoss()

        # 学習状態
        self.epoch = 0
        self.step = 0

    def seed_everything(self, seed: int = 42):
        """
        乱数シードの固定

        Args:
            seed: 乱数シード値
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def train_epoch(self) -> float:
        """
        1エポックの学習

        Returns:
            損失の1エポック平均
        """
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch_idx, (
            *_,
            source_mcep_normalized,
            target_mcep_normalized,
        ) in enumerate(pbar):
            # 入力データ、ターゲットデータをデバイスに転送
            source_mcep_normalized = source_mcep_normalized.to(self.device)
            target_mcep_normalized = target_mcep_normalized.to(self.device)

            # mcepの0次元目を除外
            source_mcep_normalized = source_mcep_normalized[:, 1:]
            target_mcep_normalized = target_mcep_normalized[:, 1:]

            # 勾配リセット
            self.optimizer.zero_grad()

            # 順伝播
            converted_mcep = self.model(source_mcep_normalized)

            # 損失計算
            loss = self.criterion(converted_mcep, target_mcep_normalized)

            # 逆伝播
            loss.backward()

            # パラメータ更新
            self.optimizer.step()

            # 記録
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})

            # TensorBoardにステップごとの損失を記録
            self.writer.add_scalar("train/loss(step)", loss.item(), self.step)

            self.step += 1

        avg_loss = total_loss / len(self.train_loader)

        return avg_loss

    @torch.inference_mode()
    def validate(self) -> dict[str, float]:
        """
        検証データで評価

        Returns:
            avg_loss: 検証損失の平均
            avg_mcd: 検証MCDの平均
        """
        self.model.eval()
        total_loss = 0.0
        mcd_values = []

        # 検証データで評価
        for (
            _,
            target_mcep,
            source_mcep_normalized,
            target_mcep_normalized,
        ) in self.val_loader:
            # デバイスに転送
            source_mcep_normalized = source_mcep_normalized.to(self.device)
            target_mcep_normalized = target_mcep_normalized.to(self.device)

            # mcepの0次元目は学習対象からもMCD計算からも除外
            target_mcep = target_mcep[:, 1:]
            source_mcep_normalized = source_mcep_normalized[:, 1:]
            target_mcep_normalized = target_mcep_normalized[:, 1:]

            # 推論
            converted_mcep = self.model(source_mcep_normalized)

            # 損失計算
            loss = self.criterion(converted_mcep, target_mcep_normalized)
            total_loss += loss.item()

            # MCD計算
            converted_mcep_denormalized = denormalize_features(
                converted_mcep.detach().cpu().numpy(), self.target_stats["mcep"]
            )
            mcd = calculate_mcd(
                target_mcep.detach().cpu().numpy(),
                converted_mcep_denormalized,
                use_dtw_alignment=False,
            )
            mcd_values.append(mcd)

        avg_loss = total_loss / len(self.val_loader)
        avg_mcd = np.mean(mcd_values)

        return avg_loss, avg_mcd

    def save_checkpoint(self) -> None:
        """チェックポイント保存"""
        checkpoint = {
            "epoch": self.epoch,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": OmegaConf.to_container(self.config, resolve=True),
        }

        # 最新モデル保存
        if self.config.train.logging.keep_latest:
            path = self.checkpoint_dir / "latest.pth"
            torch.save(checkpoint, path)
            print(f"最新モデルを保存しました: {path}")

        # 定期保存
        if self.epoch % self.config.train.logging.interval == 0:
            path = self.checkpoint_dir / f"epoch_{self.epoch:04d}.pth"
            torch.save(checkpoint, path)
            print(f"epoch {self.epoch} のモデルを保存しました: {path}")

    def train(self):
        """学習メインループ"""
        print("=== 学習開始 ===")

        for self.epoch in range(1, self.config.train.epoch + 1):
            # 学習
            train_loss = self.train_epoch()

            # ロギング
            self.writer.add_scalar("train/loss(epoch)", train_loss, self.epoch)

            # 検証（評価間隔ごと）
            if self.epoch % self.config.train.eval.interval == 0:
                val_loss, val_mcd = self.validate()

                # ログ出力
                print(
                    f"Epoch {self.epoch}: train_loss={train_loss:.6f}, "
                    f"val_loss={val_loss:.6f}, "
                    f"val_mcd={val_mcd:.3f}"
                )

                # ロギング
                self.writer.add_scalar("val/loss", val_loss, self.epoch)
                self.writer.add_scalar("val/mcd", val_mcd, self.epoch)

            # チェックポイント保存
            if self.epoch % self.config.train.logging.interval == 0:
                self.save_checkpoint()

        # TensorBoard終了
        self.writer.close()
        print("=== 学習完了 ===\n")


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    print("########## 音声変換モデルの学習 ##########\n")
    trainer = VoiceConversionTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
