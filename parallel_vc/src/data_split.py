import json
from pathlib import Path

import numpy as np


def train_val_test_split(
    file_names: list,
    val_size: int = 2,
    test_size: int = 2,
    split_mode: str = "fixed",
    random_seed: int | None = None,
) -> dict:
    """
    ファイル名リストを学習・検証・評価データに分割します。

    Args:
        file_names: 分割対象の全てのファイル名リスト
        total_size: 総発話数
        val_size: 検証データの発話数
        test_size: 評価データの発話数
        random_seed: ランダムシード値
        split_mode: 分割モード（"fixed" or "random"）

    Returns:
        train_files: 学習データのファイル名リスト
        val_files: 検証データのファイル名リスト
        test_files: 評価データのファイル名リスト
    """
    total_size = len(file_names)
    assert val_size >= 0, "検証データサイズは0以上である必要があります"
    assert test_size >= 0, "評価データサイズは0以上である必要があります"
    assert (
        val_size + test_size < total_size
    ), f"検証({val_size})+評価({test_size})が総数({total_size})を超えています"
    assert split_mode in [
        "fixed",
        "random",
    ], f"split_modeは'fixed'または'random'である必要があります: {split_mode}"
    if split_mode == "random":
        assert (
            random_seed is not None
        ), "ランダム分割モードではrandom_seedを指定する必要があります"

    # ファイル番号でソート
    file_names = sorted(file_names)

    # ランダム分割の場合はfile_namesをシャッフル
    if split_mode == "random":
        np.random.seed(random_seed)
        file_names = np.random.permutation(file_names).tolist()

    # file_namesの先頭から評価、検証、残りを学習
    test_files = file_names[:test_size]
    val_files = file_names[test_size : test_size + val_size]
    train_files = file_names[test_size + val_size :]

    return train_files, val_files, test_files


def save_split_info(output_path: Path, split_info: dict, indent: int = 2) -> None:
    """
    分割情報をJSON形式で保存します。

    Args:
        output_path: 出力ファイルパス
        split_info: 分割情報の辞書
        indent: JSONファイルのインデント幅
    """
    assert (
        output_path.parent.exists()
    ), f"出力ディレクトリ: {output_path.parent}が存在しません"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_info, f, indent=2, ensure_ascii=False)


def load_split_info(split_info_path: Path) -> dict:
    """
    JSON形式で保存された分割情報を読み込みます。

    Args:
        split_info_path: 分割情報が保存されたファイルのパス

    Returns:
        分割情報の辞書
    """
    assert split_info_path.exists(), f"split_info_path: {split_info_path}が存在しません"
    return json.loads(split_info_path.read_text(encoding="utf-8"))
