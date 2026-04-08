import numpy as np
from src.dtw import align_mcep


def calculate_mcd(
    mcep_1: np.ndarray,
    mcep_2: np.ndarray,
    use_dtw_alignment: bool = False,
) -> float:
    """
    Mel-Cepstral Distortion (MCD) を計算

    Args:
        mcep_1: 1つ目のメルケプストラム (T, D) or (T1, D)
        mcep_2: 2つ目のメルケプストラム (T, D) or (T2, D)
        use_dtw_alignment: DTWを使用して時間整合を取るか

    Returns:
        mcd: MCD値 [dB]
    """
    if use_dtw_alignment:
        # DTWを用いてアラインメントを行う場合
        # 0次元目はアラインメント時には必ず使用しない
        mcep_1, mcep_2, *_ = align_mcep(mcep_1, mcep_2, use_mcep_0dim=False)
    else:
        # DTWアラインメントを行わない場合、フレーム数が一致することを確認
        assert (
            mcep_1.shape[0] == mcep_2.shape[0]
        ), "メルケプストラムのフレーム数が一致しません\nDTWアライメントを確認してください"

    # MCDを計算
    diff = mcep_1 - mcep_2
    squared_diff = diff**2
    sum_squared_diff = np.sum(squared_diff, axis=-1)  # 次元方向に合計
    mcd_frame = (10.0 / np.log(10.0)) * np.sqrt(2.0 * sum_squared_diff)
    mcd = np.mean(mcd_frame)  # 全フレームの平均

    return mcd
