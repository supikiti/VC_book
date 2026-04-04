import warnings

# 不要な警告を抑制
warnings.filterwarnings("ignore")


import numpy as np
import pyworld as pw

EPS = 1e-8


class WORLDProcessor:
    """
    WORLD vocoderを使用した音響特徴量抽出および音声合成を提供
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

    def extract_features(self, audio: np.ndarray) -> dict[str, np.ndarray]:
        """
        WORLD vocoderを使用した音響特徴量抽出

        Args:
            audio: 音声信号

        Returns:
            f0: 基本周波数
            sp: スペクトル包絡
            mcep: メルケプストラム
            ap: 非周期性指標
        """
        # WORLDは64bit浮動小数点を使用するため型変換が必要
        audio = audio.astype(np.float64)

        # WORLDによる特徴量抽出
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

        return f0, sp, mcep, ap

    def synthesize_audio(
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
        assert (
            sp is not None or mcep is not None
        ), "spまたはmcepのいずれかを指定してください"

        # spがNoneの場合、mcepからspを復元
        if sp is None and mcep is not None:
            sp = pw.decode_spectral_envelope(
                mcep, self.sample_rate, fft_size=(ap.shape[1] - 1) * 2 + 1
            )

        # WORLDによる音声の合成
        synthesized_audio = pw.synthesize(
            f0, sp, ap, self.sample_rate, frame_period=self.frame_period
        )

        # float64からfloat32に型変換
        synthesized_audio = synthesized_audio.astype(np.float32)

        return synthesized_audio
