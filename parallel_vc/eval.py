import warnings

warnings.filterwarnings("ignore")

from pathlib import Path

import hydra
import numpy as np
import torch

# 自作
from omegaconf import DictConfig, OmegaConf
from src.audio import normalize_audio, save_audio
from src.feature import denormalize_features, load_features, normalize_features
from src.mcd import calculate_mcd
from src.model import MLPConversionModel
from src.world import WORLDProcessor

EPS = 1e-8


@hydra.main(version_base=None, config_path="config", config_name="config")
@torch.inference_mode()
def main(config: DictConfig):
    print("########## 音声変換モデルの評価 ##########\n")

    # デバイス設定（モデルが小さいため、評価データの推論はCPUで実施）
    device = "cpu"

    # パス設定
    checkpoint_path = (
        Path(config.path.output_base_dir)
        / config.path.output_tag.checkpoint
        / config.eval.checkpoint_name
    )
    feature_dir = Path(config.path.output_base_dir) / config.path.output_tag.feature
    eval_dir = Path(config.path.output_base_dir) / config.path.output_tag.eval
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 合成のために必要なWORLD Processorのインスタンス化
    world_processor = WORLDProcessor(
        sample_rate=config.data.sample_rate,
        frame_period=config.data.frame_period,
        f0_floor=config.data.f0_floor,
        f0_ceil=config.data.f0_ceil,
        mcep_order=config.data.mcep_order,
    )

    # チェックポイント読み込み
    print(f"チェックポイントを読み込み中: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # モデル作成と重み読み込み
    train_config = OmegaConf.create(checkpoint["config"])
    model = MLPConversionModel(
        input_dim=train_config.data.mcep_order - 1,  # 0次元目を除外
        output_dim=train_config.data.mcep_order - 1,  # 0次元目を除外
        hidden_dim=train_config.train.model.hidden_dim,
        dropout_rate=train_config.train.model.dropout_rate,
    )
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    print(f"モデル読み込み完了 (エポック: {checkpoint['epoch']})")

    # 特徴量読み込み
    source_features = load_features(
        input_path=feature_dir / f"{config.source_speaker}" / "test_features.npz"
    )
    source_features_aligned = load_features(
        input_path=feature_dir
        / f"{config.source_speaker}"
        / "test_features_aligned.npz"
    )
    target_features_aligned = load_features(
        input_path=feature_dir
        / f"{config.target_speaker}"
        / "test_features_aligned.npz"
    )

    # 統計量読み込み
    source_stats = load_features(
        input_path=feature_dir / f"{config.source_speaker}" / "stats.npz"
    )
    target_stats = load_features(
        input_path=feature_dir / f"{config.target_speaker}" / "stats.npz"
    )

    # 評価ループ
    for file_name in source_features.keys():
        ##### mcdの計算 #####
        # アラインメント済みMCEPを取得
        source_mcep_aligned = source_features_aligned[file_name]["mcep"]
        target_mcep_aligned = target_features_aligned[file_name]["mcep"]

        # 変換元話者の特徴量を正規化
        source_mcep_aligned_normalized, _ = normalize_features(
            source_mcep_aligned, source_stats["mcep"]
        )

        # 0次元目を除外
        source_mcep_aligned_normalized = source_mcep_aligned_normalized[:, 1:]
        target_mcep_aligned = target_mcep_aligned[:, 1:]

        # 変換
        source_mcep_aligned_normalized = torch.tensor(
            source_mcep_aligned_normalized,
            dtype=torch.float32,
        ).to(device)
        converted_mcep_aligned = model(source_mcep_aligned_normalized).cpu().numpy()

        # 逆正規化
        converted_mcep_aligned_denormalized = denormalize_features(
            converted_mcep_aligned,
            stats={
                "mean": target_stats["mcep"]["mean"][1:],
                "std": target_stats["mcep"]["std"][1:],
            },
        )

        # mcdの計算
        mcd_ = calculate_mcd(converted_mcep_aligned_denormalized, target_mcep_aligned)
        print(f"{file_name}のMCD: {mcd_:.4f} dB")

        ##### 変換音声の合成と保存 #####
        # アラインメント未処理の特徴量を取得
        source_f0 = source_features[file_name]["f0"]
        source_mcep = source_features[file_name]["mcep"]
        source_ap = source_features[file_name]["ap"]

        # 変換元話者の特徴量を正規化
        source_mcep_normalized, _ = normalize_features(
            source_mcep, source_stats["mcep"]
        )

        # 0次元目を除外（0次元目は変換せず、統計量で変換して使用するため保存）
        source_mcep_normalized_0dim = source_mcep_normalized[:, 0:1]
        source_mcep_normalized = source_mcep_normalized[:, 1:]

        # 変換
        source_mcep_normalized = torch.tensor(
            source_mcep_normalized,
            dtype=torch.float32,
        ).to(device)
        converted_mcep = model(source_mcep_normalized).cpu().numpy()

        # 0次元目と結合する
        converted_mcep = np.concatenate(
            [source_mcep_normalized_0dim, converted_mcep], axis=1
        )

        # 逆正規化
        converted_mcep = denormalize_features(converted_mcep, target_stats["mcep"])

        # 変換元話者のF0を対数領域で正規化
        converted_lf0, _ = normalize_features(
            np.log(source_f0 + EPS), source_stats["lf0"]
        )

        # 変換先話者のF0に変換
        converted_lf0 = denormalize_features(converted_lf0, target_stats["lf0"])
        converted_f0 = np.exp(converted_lf0)
        converted_f0 = np.where(source_f0 > 0, converted_f0, 0.0)

        # WORLDを用いて音声を合成
        converted_audio = world_processor.synthesize_audio(
            f0=converted_f0,
            mcep=converted_mcep,
            ap=source_ap,
        )

        # 保存
        converted_audio = normalize_audio(
            audio=converted_audio,
            max_value=config.data.audio_max_value,
        )
        save_audio(
            output_path=eval_dir
            / file_name.replace(
                ".wav", f"_{config.source_speaker}_to_{config.target_speaker}.wav"
            ),
            audio=converted_audio,
            sample_rate=config.data.sample_rate,
        )


if __name__ == "__main__":
    main()
