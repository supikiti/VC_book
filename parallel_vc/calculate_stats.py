from pathlib import Path

import hydra
import numpy as np
from omegaconf import DictConfig

# 自作モジュールのインポート
from src.feature import load_features, save_features


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(config: DictConfig):
    print("########## 特徴量の統計量を計算 ##########\n")

    # 元話者と目標話者に対して学習データから統計量を計算
    for speaker in [config.source_speaker, config.target_speaker]:
        print(f"---------- {speaker}の統計量計算 ----------")

        # 特徴量の読み込み
        features = load_features(
            input_path=Path(config.path.output_base_dir)
            / config.path.output_tag.feature
            / speaker
            / "train_features.npz"
        )

        # 全発話の特徴量を連結
        f0_all = np.concatenate([feats["f0"] for feats in features.values()], axis=0)
        mcep_all = np.concatenate(
            [feats["mcep"] for feats in features.values()], axis=0
        )

        # 統計量の計算
        stats = {}
        # F0はF0>0（有声音）部分から計算する
        # 人間の聴覚特性（対数知覚）に合わせて対数F0を使用する
        f0_mask = f0_all > 0
        stats["lf0"] = {
            "mean": np.mean(np.log(f0_all[f0_mask])),
            "std": np.std(np.log(f0_all[f0_mask])),
        }
        # MCEPは0次元目が閾値以上（発話区間）の部分から計算する
        # 0次元目は包絡のエネルギーに対応しているため、それを利用して無声音区間を除去する
        mcep_mask = (
            mcep_all[:, 0] >= np.max(mcep_all[:, 0]) + config.data.mcep_0dim_threshold
        )
        stats["mcep"] = {
            "mean": np.mean(mcep_all[mcep_mask], axis=0),
            "std": np.std(mcep_all[mcep_mask], axis=0),
        }

        # 統計量の保存
        stats_output_path = (
            Path(config.path.output_base_dir)
            / config.path.output_tag.feature
            / speaker
            / "stats.npz"
        )
        save_features(
            output_path=stats_output_path,
            features=stats,
        )

        # 計算結果の表示
        with np.printoptions(formatter={"float": "{:.4g}".format}, linewidth=200):
            print(
                f"""
対数F0 ({np.sum(f0_mask)}フレームから計算):
    平均: {stats['lf0']['mean']:.4g} ({np.exp(stats['lf0']['mean']):.4g}Hz)
    標準偏差: {stats['lf0']['std']:.4g}
    最大: {np.max(f0_all[f0_mask]):.4g}Hz
    最小: {np.min(f0_all[f0_mask]):.4g}Hz

MCEP ({np.sum(mcep_mask)}フレームから計算):
    平均（0次元目～{config.data.mcep_order-1}次元目）: {stats['mcep']['mean']}
    標準偏差（0次元目～{config.data.mcep_order-1}次元目）: {stats['mcep']['std']}
                """
            )


if __name__ == "__main__":
    main()
