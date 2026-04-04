from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

EPS = 1e-8


def load_audio(audio_path: Path, sample_rate: int = 24000) -> np.ndarray:
    """
    音声ファイルを読み込み

    Args:
        audio_path: 音声ファイルパス
        sample_rate: サンプリング周波数

    Returns:
        audio: 音声信号
    """
    assert audio_path.exists(), f"audio_path: {audio_path}が存在しません"
    assert sample_rate > 0, "sample_rateは正の整数である必要があります"
    return librosa.load(audio_path, sr=sample_rate)[0]


def save_audio(output_path: Path, audio: np.ndarray, sample_rate: int) -> None:
    """
    音声ファイルを保存

    Args:
        output_path: 出力ファイルパス
        audio: 音声信号
        sample_rate: サンプリング周波数
    """
    assert (
        output_path.parent.exists()
    ), f"出力ディレクトリ: {output_path.parent}が存在しません"
    assert sample_rate > 0, "sample_rateは正の整数である必要があります"
    sf.write(output_path, audio, sample_rate)


def trim_silence(audio: np.ndarray, top_db: int = 60) -> np.ndarray:
    """
    音声信号の前後の無音区間をトリミング

    Args:
        audio: 入力音声信号
        top_db: 無音とみなすデシベル閾値

    Returns:
        trimmed_audio: 無音区間をトリミングした音声
    """
    return librosa.effects.trim(audio, top_db=top_db)[0]


def normalize_audio(audio: np.ndarray, max_value: float = 0.95) -> np.ndarray:
    """
    音声信号の正規化

    Args:
        audio: 入力音声信号
        max_value: 正規化後の最大振幅値

    Returns:
        normalized_audio: 正規化済み音声
    """
    assert 0 < max_value <= 1.0, "max_valueは0より大きく1以下である必要があります"
    return audio / (np.max(np.abs(audio)) + EPS) * max_value


def preprocess_audio(
    audio: np.ndarray, top_db: int = 60, max_value: float = 0.95
) -> np.ndarray:
    """
    音声の前処理
    - 無音区間の除去
    - 正規化

    Args:
        audio: 入力音声信号

    Returns:
        processed_audio: 前処理済み音声
    """
    processed_audio = trim_silence(audio, top_db=top_db)
    processed_audio = normalize_audio(processed_audio, max_value=max_value)
    return processed_audio
