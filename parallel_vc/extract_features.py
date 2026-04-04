import warnings

# 不要な警告を抑制
warnings.filterwarnings("ignore")

from pathlib import Path

import hydra

# 自作モジュールのインポート
from omegaconf import DictConfig
from src.audio import load_audio, preprocess_audio
from src.data_split import save_split_info, train_val_test_split
from src.feature import save_features
from src.world import WORLDProcessor
from tqdm import tqdm


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    print("########## WORLD特徴量抽出 ##########\n")

    print("---------- 1. JVSコーパスのファイル名を取得 ----------")
    # JVSコーパスのファイル名を取得
    source_speaker_file_names = [
        p.name
        for p in (
            Path(config.path.corpus_dir) / config.source_speaker / "parallel100/"
        ).glob("**/*.wav")
    ]
    target_speaker_file_names = [
        p.name
        for p in (
            Path(config.path.corpus_dir) / config.target_speaker / "parallel100/"
        ).glob("**/*.wav")
    ]

    # jvs030・074・089は99文しかないため、共通のファイルのみ使用するようにする
    file_names = sorted(
        list(set(source_speaker_file_names) & set(target_speaker_file_names))
    )
    print("\n")

    print("---------- 2. データ分割の作成 ----------")
    # データ分割の作成
    train_file_names, val_file_names, test_file_names = train_val_test_split(
        file_names=file_names,
        val_size=config.data.validation_size,
        test_size=config.data.test_size,
        split_mode=config.data.split_mode,
        random_seed=config.random_seed,
    )
    split_info = {
        "train": train_file_names,
        "val": val_file_names,
        "test": test_file_names,
    }

    # 分割情報の保存
    split_info_path = (
        Path(config.path.output_base_dir) / config.path.output_tag.split_info
    )
    split_info_path.parent.mkdir(parents=True, exist_ok=True)
    save_split_info(
        output_path=split_info_path,
        split_info=split_info,
    )

    print(f"データ分割情報を保存: {split_info_path}\n")

    print("---------- 3. WORLD特徴量の抽出 ----------")
    # WORLD Processorの初期化
    world_processor = WORLDProcessor(
        sample_rate=config.data.sample_rate,
        frame_period=config.data.frame_period,
        f0_floor=config.data.f0_floor,
        f0_ceil=config.data.f0_ceil,
        mcep_order=config.data.mcep_order,
    )

    # 変換元話者と変換先話者の両方で特徴量抽出を実行
    for speaker in [config.source_speaker, config.target_speaker]:
        for split_name, file_names in split_info.items():
            features = {}

            # 発話ごとに特徴量抽出
            for file_name in tqdm(
                file_names, desc=f"{speaker}: {split_name}特徴量抽出"
            ):
                # 音声の読み込み
                audio = load_audio(
                    audio_path=(
                        Path(config.path.corpus_dir)
                        / speaker
                        / "parallel100/wav24kHz16bit"
                        / file_name
                    ),
                    sample_rate=config.data.sample_rate,
                )

                # 前処理
                audio = preprocess_audio(
                    audio=audio,
                    top_db=config.data.top_db,
                    max_value=config.data.audio_max_value,
                )

                # WORLD特徴量抽出
                f0, sp, mcep, ap = world_processor.extract_features(audio)

                # 特徴量を保存
                features[file_name] = {
                    "f0": f0,
                    "sp": sp,
                    "mcep": mcep,
                    "ap": ap,
                }

            # 特徴量の保存
            feature_output_path = (
                Path(config.path.output_base_dir)
                / config.path.output_tag.feature
                / speaker
                / f"{split_name}_features.npz"
            )
            feature_output_path.parent.mkdir(parents=True, exist_ok=True)
            save_features(
                output_path=feature_output_path,
                features=features,
            )
            print(f"特徴量を保存: {feature_output_path}")
        print("\n")


if __name__ == "__main__":
    main()
