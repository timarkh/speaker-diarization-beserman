import numpy as np
import librosa
from pathlib import Path
import subprocess
import tempfile

TARGET_SR = 16000


def load_audio(audio_path: Path, sr=16000):
    """
    Load audio, convert to mono and resample.
    """
    if str(audio_path).lower().endswith(".mp4"):
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            subprocess.run([
                "ffmpeg",
                "-y",
                "-i", str(audio_path),
                "-ac", "1",
                "-ar", str(sr),
                tmp.name
            ], check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
            waveform, sr = librosa.load(tmp.name, sr=None)
    else:
        waveform, sr = librosa.load(
            audio_path,
            sr=sr,
            mono=True
        )
    return waveform.astype(np.float32)

def extract_features(
    waveform,
    sr=16000,
    n_fft=400,          # 25 ms
    hop_length=160,     # 10 ms
    n_mels=40,
):
    """
    Extract acoustic features.

    Parameters
    ----------
    waveform : np.ndarray
        Mono waveform.
    sr : int
        Sample rate.
    n_fft : int
        FFT window size.
    hop_length : int
        Hop size.
    n_mels : int
        Number of Mel filters.

    Returns
    -------
    features : np.ndarray
        Shape (n_frames, 124):
            40 log-Mel
            40 delta
            40 delta-delta
             1 RMS
             1 log(F0)
             1 Δlog(F0)
             1 voiced probability
    """

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )

    # Log-Mel
    logmel = librosa.power_to_db(mel, ref=np.max)
    # or maybe: logmel = librosa.power_to_db(mel, ref=1.0) -- remember absolute loudness, normalize over the whole corpus later

    # First and second derivatives
    delta = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)

    # RMS energy
    rms = librosa.feature.rms(
        y=waveform,
        frame_length=n_fft,
        hop_length=hop_length,
    )

    # F0-related features
    f0, voiced_flag, voiced_prob = librosa.pyin(
        waveform,
        fmin=70,
        fmax=400,
        sr=sr,
        frame_length=n_fft,
        hop_length=hop_length,
    )
    f0_clean = np.nan_to_num(f0, nan=1.0)
    log_f0 = np.log(f0_clean)
    log_f0[np.isnan(f0)] = 0.0
    valid = ~np.isnan(f0)

    # Interpolate values for unvoiced segments
    if valid.any():
        log_f0 = np.interp(
            np.arange(len(f0)),
            np.where(valid)[0],
            np.log(f0[valid])
        )
    else:
        log_f0 = np.zeros_like(f0)

    delta_f0 = librosa.feature.delta(
        log_f0.reshape(1, -1)
    )

    # All feature matrices should have the same number of frames,
    # but trim to the shortest just in case.
    n_frames = min(
        logmel.shape[1],
        delta.shape[1],
        delta2.shape[1],
        rms.shape[1],
    )

    logmel = logmel[:, :n_frames]
    delta = delta[:, :n_frames]
    delta2 = delta2[:, :n_frames]
    rms = rms[:, :n_frames]
    log_f0 = log_f0[:n_frames, None].T
    voiced_prob = voiced_prob[:n_frames, None].T
    delta_f0 = delta_f0[:, :n_frames]

    # for name, feat in [
    #     ("mel", mel),
    #     ("delta", delta),
    #     ("delta2", delta2),
    #     ("rms", rms),
    #     ("log_f0", log_f0),
    #     ("delta_f0", delta_f0),
    #     ("voiced_prob", voiced_prob),
    # ]:
    #     print(f"{name:12} {feat.shape}")

    # Stack into (frames, features)
    features = np.concatenate(
        [
            logmel,
            delta,
            delta2,
            rms,
            log_f0,
            delta_f0,
            voiced_prob
        ],
        axis=0,
    ).T.astype(np.float32)

    frame_times = librosa.frames_to_time(
        np.arange(n_frames),
        sr=sr,
        hop_length=hop_length,
        n_fft=n_fft
    )
    frame_times *= 1000


    return features, frame_times
