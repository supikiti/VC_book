"""
JVSコーパスを使用した音声変換のための基本的な音声処理機能を提供します。
WORLD vocoder を使用した特徴量抽出、音声の前処理、正規化等を実装しています。
"""

import warnings

# 不要な警告を抑制
warnings.filterwarnings("ignore")

import argparse
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf


class AudioProcessor:
    """
    音声の読み込み、前処理、WORLD特徴量抽出、音声合成などの基本的な音声処理機能を提供します。
    """

    def __init__(
        self,
        sample_rate: int = 24000,
        frame_period: float = 5.0,
        f0_floor: float = 60.0,
        f0_ceil: float = 500.0,
        mcep_order: int = 40,
    ):
        """
        Args:
            sample_rate: サンプリング周波数（JVSコーパスのデフォルト: 24000Hz）
            frame_period: WORLD vocoderのフレーム周期（ms）
            f0_floor: F0の下限値（Hz）
            f0_ceil: F0の上限値（Hz）
            mcep_order: メルケプストラム次数
        """
        self.sample_rate = sample_rate
        self.frame_period = frame_period
        self.f0_floor = f0_floor
        self.f0_ceil = f0_ceil
        self.mcep_order = mcep_order
        self.eps = 1e-8  # 数値安定性のための微小値

    def extract_world_features(
        self, audio: np.ndarray | None = None, audio_path: Path | None = None
    ) -> dict[str, np.ndarray]:
        """
        WORLD vocoderを使用した音響特徴量抽出

        Args:
            audio: 音声信号
            audio_path: 音声ファイルパス

        Returns:
            features: 特徴量辞書
                - f0: 基本周波数
                - sp: スペクトル包絡
                - mcep: メルケプストラム
                - ap: 非周期性指標
        """
        # audioまたはaudio_pathのいずれかを指定
        assert (audio is not None) ^ (
            audio_path is not None
        ), "audioまたはaudio_pathのいずれかを指定してください"

        # 音声ファイルから読み込み
        if audio_path is not None:
            audio = self.load_audio(audio_path)

        # 前処理
        audio = self.preprocess_audio(audio)

        # WORLD特徴量抽出
        audio = audio.astype(
            np.float64
        )  # WORLDは64bit浮動小数点を使用するため型キャストが必要
        f0_, t = pw.dio(
            audio,
            self.sample_rate,
            f0_floor=self.f0_floor,
            f0_ceil=self.f0_ceil,
            frame_period=self.frame_period,
        )
        f0 = pw.stonemask(audio, f0_, t, self.sample_rate)
        sp = pw.cheaptrick(audio, f0, t, self.sample_rate)
        mcep = pw.code_spectral_envelope(sp, self.sample_rate, self.mcep_order)
        ap = pw.d4c(audio, f0, t, self.sample_rate)

        return {"f0": f0, "sp": sp, "mcep": mcep, "ap": ap}

    def synthesize_world_audio(
        self,
        f0: np.ndarray,
        ap: np.ndarray,
        sp: np.ndarray | None = None,
        mcep: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        WORLD vocoderを使用した音声合成

        Args:
            f0: 基本周波数
            ap: 非周期性指標
            sp: スペクトル包絡
            mcep: メルケプストラム

        Returns:
            synthesized_audio: 合成音声
        """
        # spがNoneの場合、mcepからspを復元
        if sp is None and mcep is not None:
            sp = pw.decode_spectral_envelope(
                mcep, self.sample_rate, fft_size=(ap.shape[1] - 1) * 2 + 1
            )

        # WORLD合成
        synthesized_audio = pw.synthesize(
            f0, sp, ap, self.sample_rate, frame_period=self.frame_period
        )
        synthesized_audio = synthesized_audio.astype(np.float32)  # float32に型変換

        # 振幅正規化
        synthesized_audio = (
            synthesized_audio / np.max(np.abs(synthesized_audio) + self.eps) * 0.95
        )

        return synthesized_audio

if __name__ == "__main__":
    # argparserに置き換える
    parser = argparse.ArgumentParser(description="AudioProcessorの使用例とテスト")
    parser.add_argument(
        "--audio_path",
        type=str,
        default="./test_audio.wav",
        help="テスト用音声ファイルパス",
    )
    args = parser.parse_args()

    # AudioProcessorのインスタンス化
    print("AudioProcessorのテストを開始します...")
    processor = AudioProcessor()

    # テスト用音声の確認
    assert Path(args.audio_path).exists(), "テスト用音声ファイルが存在しません"

    # 音声読み込みテスト
    audio = processor.load_audio(args.audio_path)
    print(f"読み込み完了: {len(audio)} samples")
    print(f"音声ファイルの長さ: {librosa.get_duration(path=args.audio_path):.2f}秒")

    # 前処理テスト
    processed_audio = processor.preprocess_audio(audio)
    print(f"前処理完了: {len(processed_audio)} samples")

    # WORLD特徴量抽出テスト
    features = processor.extract_world_features(processed_audio)
    print(f"F0: {features['f0'].shape}")
    print(f"SP: {features['sp'].shape}")
    print(f"MCEP: {features['mcep'].shape}")
    print(f"AP: {features['ap'].shape}")

    # 音声合成テスト
    synthesized_wo_mcep = processor.synthesize_world_audio(
        f0=features["f0"], sp=features["sp"], ap=features["ap"]
    )
    synthesized_w_mcep = processor.synthesize_world_audio(
        f0=features["f0"], mcep=features["mcep"], ap=features["ap"]
    )
    print(f"合成完了: {len(synthesized_wo_mcep)} samples")

    # 保存テスト
    processor.save_audio(synthesized_wo_mcep, "./test_synthesized_wo_mcep.wav")
    processor.save_audio(synthesized_w_mcep, "./test_synthesized_w_mcep.wav")
    print("音声保存完了")
