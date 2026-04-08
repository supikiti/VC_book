import warnings

# 不要な警告を抑制
warnings.filterwarnings("ignore")

from pathlib import Path

import hydra

# 自作モジュールのインポート
from omegaconf import DictConfig
from src.data_split import load_split_info
from src.dtw import align_features
from src.feature import load_features, save_features
from tqdm import tqdm


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    print("########## DTWによる特徴量のアラインメント処理 ##########\n")

    # データセット分割情報の読み込み
    split_info = load_split_info(
        Path(config.path.output_base_dir) / config.path.output_tag.split_info
    )

    # 分割ごとにアライメント実行
    feature_dir = Path(config.path.output_base_dir) / config.path.output_tag.feature
    for split_name, file_names in split_info.items():
        aligned_source_features = {}
        aligned_target_features = {}

        # 特徴量の読み込み
        source_features = load_features(
            input_path=Path(config.path.output_base_dir)
            / config.path.output_tag.feature
            / f"{config.source_speaker}"
            / f"{split_name}_features.npz"
        )
        target_features = load_features(
            input_path=Path(config.path.output_base_dir)
            / config.path.output_tag.feature
            / f"{config.target_speaker}"
            / f"{split_name}_features.npz"
        )

        # 各ファイルをアライメント
        for file_name in tqdm(file_names, desc=f"{split_name}データアライメント"):
            # mcepを取得
            source_mcep = source_features[file_name]["mcep"]
            target_mcep = target_features[file_name]["mcep"]

            # DTWアライメントの実行
            # mcepの0次元目はエネルギーに対応しているため、アライメント計算に使用しない
            aligned_source_features[file_name], aligned_target_features[file_name] = (
                align_features(
                    source_mcep=source_mcep,
                    target_mcep=target_mcep,
                    source_features=source_features[file_name],
                    target_features=target_features[file_name],
                    use_mcep_0dim=False,
                )
            )

        # アライメント済みデータを保存
        output_file_source = (
            feature_dir
            / f"{config.source_speaker}"
            / f"{split_name}_features_aligned.npz"
        )
        output_file_target = (
            feature_dir
            / f"{config.target_speaker}"
            / f"{split_name}_features_aligned.npz"
        )
        save_features(
            output_path=output_file_source,
            features=aligned_source_features,
        )
        save_features(
            output_path=output_file_target,
            features=aligned_target_features,
        )

        print(
            f"アライメント済みデータを保存: {output_file_source}, {output_file_target}\n"
        )


if __name__ == "__main__":
    main()
