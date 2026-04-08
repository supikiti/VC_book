from pathlib import Path

import japanize_matplotlib
import matplotlib.pyplot as plt
import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# 明示的に日本語対応を有効化
japanize_matplotlib.japanize()


def align_mcep(
    source_mcep: np.ndarray, target_mcep: np.ndarray, use_mcep_0dim: bool = False
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """
    DTWアライメントを実行

    Args:
        source_mcep: 変換元話者のメルケプストラム
        target_mcep: 変換先話者のメルケプストラム
        use_mcep_0dim: メルケプストラムの0次元目をアライメント計算に使用するかどうか

    Returns:
        ource_mcep_aligned: アライメント済み変換元メルケプストラム
        target_mcep_aligned: アライメント済み変換先メルケプストラム
        source_indices: 変換元メルケプストラムのアライメントインデックス
        target_indices: 変換先メルケプストラムのアライメントインデックス
    """
    # DTWアライメントを実行
    if use_mcep_0dim:
        distance, path = fastdtw(source_mcep, target_mcep, dist=euclidean)
    else:
        distance, path = fastdtw(source_mcep[:, 1:], target_mcep[:, 1:], dist=euclidean)

    # アライメントパスからインデックスを抽出
    source_indices = [p[0] for p in path]
    target_indices = [p[1] for p in path]

    # インデックスを用いて各データをアライメント
    source_mcep_aligned = source_mcep[source_indices]
    target_mcep_aligned = target_mcep[target_indices]

    return (
        source_mcep_aligned,
        target_mcep_aligned,
        source_indices,
        target_indices,
    )


def align_features(
    source_mcep: np.ndarray,
    target_mcep: np.ndarray,
    source_features: dict,
    target_features: dict,
    use_mcep_0dim: bool = False,
) -> tuple[dict, dict]:
    """
    メルケプストラムを用いたDTWアライメント実行

    Args:
        source_mcep: 変換元話者のメルケプストラム
        target_mcep: 変換先話者のメルケプストラム
        source_features: 変換元話者の特徴量辞書
        target_features: 変換先話者の特徴量辞書
        use_mcep_0dim: メルケプストラムの0次元目をアライメント計算に使用するかどうか

    Returns:
        aligned_source_features: アライメント済み変換元特徴量
        aligned_target_features: アライメント済み変換先特徴量
    """
    # MCEPに対してDTWアライメントを実行
    *_, source_indices, target_indices = align_mcep(
        source_mcep, target_mcep, use_mcep_0dim
    )

    # インデックスを用いて各特徴量をアライメントして辞書に格納
    aligned_source_features = {
        "f0": source_features["f0"][source_indices],
        "mcep": source_features["mcep"][source_indices],
        "ap": source_features["ap"][source_indices],
    }
    aligned_target_features = {
        "f0": target_features["f0"][target_indices],
        "mcep": target_features["mcep"][target_indices],
        "ap": target_features["ap"][target_indices],
    }

    return aligned_source_features, aligned_target_features


def visualize_dtw_alignment(
    self,
    output_path: Path,
    file_name: str,
    source_mcep: np.ndarray,
    target_mcep: np.ndarray,
    use_mcep_0dim: bool = False,
    mcep_dim: int = 1,
    frame_period: float = 5.0,
    figsize: tuple = (12, 8),
    dpi: int = 300,
):
    """
    DTWアライメントの可視化

    Args:
        output_path: 出力ファイルパス
        file_name: 処理対象ファイル名
        source_mcep: 変換元話者のメルケプストラム
        target_mcep: 変換先話者のメルケプストラム
        use_mcep_0dim: メルケプストラムの0次元目をアライメント計算に使用するかどうか
        mcep_dim: 可視化するメルケプストラムの次元
        frame_period: フレーム周期（ms）
        figsize: 図のサイズ
        dpi: 図の解像度
    """
    # 出力ディレクトリを作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # DTWアライメントを実行
    source_mcep_aligned, target_mcep_aligned, *_ = align_mcep(
        source_mcep, target_mcep, use_mcep_0dim
    )

    # MCEPの特定の次元を抽出
    source_mcep = source_mcep[:, mcep_dim]
    target_mcep = target_mcep[:, mcep_dim]
    source_mcep_aligned = source_mcep_aligned[:, mcep_dim]
    target_mcep_aligned = source_mcep_aligned[:, mcep_dim]

    # 時間軸（横軸）の作成
    time_source = np.arange(len(source_mcep)) * frame_period / 1000.0
    time_target = np.arange(len(target_mcep)) * frame_period / 1000.0
    time_aligned = np.arange(len(source_mcep_aligned)) * frame_period / 1000.0

    def plot_mcep_trajectory(
        ax: plt.Axes,
        t1: np.ndarray,
        d1: np.ndarray,
        t2: np.ndarray,
        d2: np.ndarray,
        title: str,
        xlabel: str,
    ):
        """
        メルケプストラム軌跡のプロット補助関数

        Args:
            ax: プロット用Axes
            t1: データ1の時間軸
            d1: データ1のメルケプストラム
            t2: データ2の時間軸
            d2: データ2のメルケプストラム
            title: グラフタイトル
            xlabel: x軸ラベル
        """
        ax.plot(t1, d1, "b-", linewidth=1, label=self.config.SOURCE_SPEAKER, alpha=0.8)
        ax.plot(t2, d2, "r-", linewidth=1, label=self.config.TARGET_SPEAKER, alpha=0.8)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(f"メルケプストラム {mcep_dim}次元係数", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)

    # 描画
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=figsize)
    plot_mcep_trajectory(
        ax_top,
        time_source,
        source_mcep,
        time_target,
        target_mcep,
        f"(a) DTW前のメルケプストラム {mcep_dim}次元目",
        "",
    )
    plot_mcep_trajectory(
        ax_bottom,
        time_aligned,
        source_mcep_aligned,
        time_aligned,
        target_mcep_aligned,
        f"(b) DTW後のメルケプストラム {mcep_dim}次元目",
        "時間 [s]",
    )
    fig.suptitle(f"DTWアライメント比較 ({file_name})", fontsize=16, fontweight="bold")
    plt.tight_layout()

    # 保存
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"DTWアライメント結果を保存: {output_path}")
