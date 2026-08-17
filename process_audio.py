import numpy as np
import librosa
from scipy.signal import find_peaks
from pathlib import Path
import subprocess
import tempfile
import random

TARGET_SR = 16000


def augment_waveform(waveform, sr=16000, segment_seconds=120):
    """
    Randomly augment a waveform.

    Returns
    -------
    waveform : np.ndarray
        Augmented waveform.
    """
    segment_length = int(segment_seconds * sr)
    augmented = np.empty_like(waveform)
    for start in range(0, len(waveform), segment_length):
        end = min(start + segment_length, len(waveform))
        segment = waveform[start:end].copy()

        # ----------------------------------------------------------
        # Noise
        # ----------------------------------------------------------

        if random.random() < 0.30:
            snr_db = random.uniform(15, 30)
            signal_power = np.mean(segment ** 2)
            if signal_power > 0:
                noise_power = signal_power / (
                        10 ** (snr_db / 10)
                )
                noise = np.random.normal(
                    0,
                    np.sqrt(noise_power),
                    size=segment.shape
                )
                segment += noise

        # ----------------------------------------------------------
        # Echo
        # ----------------------------------------------------------

        if random.random() < 0.15:
            delay_ms = random.uniform(20, 80)
            attenuation = random.uniform(0.1, 0.4)
            delay = int(sr * delay_ms / 1000)
            if 0 < delay < len(segment):
                segment[delay:] += (
                        attenuation * segment[:-delay]
                )

        # ----------------------------------------------------------
        # Pitch shift
        # ----------------------------------------------------------

        if random.random() < 0.20:
            n_steps = random.uniform(-0.6, 0.6)
            segment = librosa.effects.pitch_shift(
                segment,
                sr=sr,
                n_steps=n_steps
            )
        augmented[start:end] = segment

    # ------------------------------------------------------------
    # Prevent clipping
    # ------------------------------------------------------------

    max_abs = np.max(np.abs(waveform))
    if max_abs > 1.0:
        waveform = waveform / max_abs
    return waveform

def extract_two_f0(
        waveform,
        sr=16000,
        n_fft=2048,
        hop_length=160,
        win_length=400,
        fmin=79,
        fmax=380,
        n_harmonics=6,
        min_f0_distance=30,
):
    """
    Estimate the two strongest F0 candidates for each frame.

    Returns
    -------
    f0_1 : ndarray, shape (n_frames,)
        Strongest F0 candidate in Hz. NaN if none detected.

    f0_2 : ndarray, shape (n_frames,)
        Second strongest F0 candidate in Hz. NaN if none detected.

    salience_1 : ndarray, shape (n_frames,)
        Salience of the strongest candidate.

    salience_2 : ndarray, shape (n_frames,)
        Salience of the second candidate.
    """

    # ------------------------------------------------------------
    # STFT magnitude
    # ------------------------------------------------------------

    S = np.abs(
        librosa.stft(
            waveform,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            window="hann",
            center=True,
        )
    )
    S = np.asarray(S, dtype=np.float32)

    freqs = librosa.fft_frequencies(
        sr=sr,
        n_fft=n_fft,
    )
    print('Frequencies calculated.')

    # ------------------------------------------------------------
    # Restrict spectrum to plausible F0 range
    # ------------------------------------------------------------

    # We still need the full spectrum for harmonic lookup,
    # so don't slice S here.

    harmonics = np.arange(1, n_harmonics + 1)

    # Harmonic salience for every frequency bin and frame
    # salience = librosa.salience(
    #     S,
    #     freqs=freqs,
    #     harmonics=harmonics,
    #     filter_peaks=True,
    # )
    # Calculate salience in chunks to reduce memory consumption
    chunk_size = 5000
    salience_chunks = []
    for start in range(0, S.shape[1], chunk_size):
        end = min(start + chunk_size, S.shape[1])
        S_chunk = S[:, start:end]
        salience_chunk = librosa.salience(
            S_chunk,
            freqs=freqs,
            harmonics=harmonics,
            filter_peaks=True,
        )
        salience_chunks.append(salience_chunk)
    salience = np.concatenate(salience_chunks, axis=1)
    print('Salience calculated.')

    # salience has shape (frequency_bins, frames)
    n_frames = salience.shape[1]

    f0_1 = np.full(n_frames, np.nan)
    f0_2 = np.full(n_frames, np.nan)

    salience_1 = np.zeros(n_frames)
    salience_2 = np.zeros(n_frames)

    # Frequency-bin range corresponding to possible F0
    f0_mask = (freqs >= fmin) & (freqs <= fmax)
    f0_indices = np.where(f0_mask)[0]

    if len(f0_indices) == 0:
        return f0_1, f0_2, salience_1, salience_2

    # Number of FFT bins corresponding approximately to
    # the minimum allowed distance between the two F0s.
    freq_resolution = sr / n_fft
    min_distance_bins = max(
        1,
        int(round(min_f0_distance / freq_resolution))
    )

    # ------------------------------------------------------------
    # Find two strongest salience peaks per frame
    # ------------------------------------------------------------
    print('Looking for strongest salience peaks.')
    for frame in range(n_frames):
        values = salience[f0_indices, frame]
        if not np.any(np.isfinite(values)):
            continue

        values = np.nan_to_num(values, nan=0.0)
        peaks, properties = find_peaks(
            values,
            distance=min_distance_bins,
        )

        if len(peaks) == 0:
            continue

        peak_values = values[peaks]
        # Sort peaks by salience, strongest first
        order = np.argsort(peak_values)[::-1]
        peaks = peaks[order]
        peak_values = peak_values[order]

        # Strongest candidate
        p = peaks[0]
        f0_1[frame] = freqs[f0_indices[p]]
        salience_1[frame] = peak_values[0]

        # Second candidate
        if len(peaks) > 1:
            p = peaks[1]
            f0_2[frame] = freqs[f0_indices[p]]
            salience_2[frame] = peak_values[1]

    return f0_1, f0_2, salience_1, salience_2


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
        Shape (n_frames, 128):
            40 log-Mel
            40 delta
            40 delta-delta
             1 RMS
             1 log(F0)
             1 Δlog(F0)
             1 log(F0_2) -- potential second speaker
             1 second_f0_confidence
             1 salience(F0_2)
             1 voiced probability
             1 centroid
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
    print('mel features calculated.')

    # Log-Mel
    logmel = librosa.power_to_db(mel, ref=np.max)
    # or maybe: logmel = librosa.power_to_db(mel, ref=1.0) -- remember absolute loudness, normalize over the whole corpus later

    # First and second derivatives
    delta = librosa.feature.delta(logmel)
    delta2 = librosa.feature.delta(logmel, order=2)
    print('derivative features calculated.')

    # RMS energy
    rms = librosa.feature.rms(
        y=waveform,
        frame_length=n_fft,
        hop_length=hop_length,
    )
    print('RMS features calculated.')

    # F0-related features
    f0, voiced_flag, voiced_prob = librosa.pyin(
        waveform,
        fmin=80,
        fmax=380,
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
    print('pYIN F0 calculated.')

    f0_1, f0_2, sal1, sal2 = extract_two_f0(
        waveform,
        sr=16000,
    )
    invalid = (
            np.isnan(f0)
            | (sal2 <= 0.3)
            | (
                    ~np.isnan(f0)
                    & (f0_2 >= 1.9 * f0)
                    & (f0_2 <= 2.1 * f0)
            )
    )
    f0_2[invalid] = np.nan
    sal2[invalid] = 0.0

    # print(
    #     f"{'Frame':>5}  "
    #     f"{'Time(ms)':>10}  "
    #     f"{'pYIN F0':>8}  "
    #     f"{'F0_1':>8}  "
    #     f"{'F0_2':>8}  "
    #     f"{'sal_1':>8}  "
    #     f"{'sal_2':>8}"
    # )
    # for i, (a, b, c, d, e) in enumerate(
    #         zip(f0, f0_1, f0_2, sal1, sal2)
    # ):
    #     print(
    #         f"{i:5d}  "
    #         f"{a:8.2f}  "
    #         f"{b:8.2f}  "
    #         f"{c:8.2f}  "
    #         f"{d:8.2f}  "
    #         f"{e:8.2f}"
    #     )

    f0_2_clean = np.nan_to_num(f0_2, nan=1.0)
    log_f0_2 = np.log(f0_2_clean)
    log_f0_2[np.isnan(f0_2)] = 0.0
    second_f0_confidence = sal2 / (sal1 + 1e-8)
    print('F0_2 calculated.')

    # All feature matrices should have the same number of frames,
    # but trim to the shortest just in case.
    n_frames = min(
        logmel.shape[1],
        delta.shape[1],
        delta2.shape[1],
        rms.shape[1],
    )

    centroid = librosa.feature.spectral_centroid(
        y=waveform,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    centroid = centroid[:, :n_frames]
    print('Centroid calculated.')

    logmel = logmel[:, :n_frames]
    delta = delta[:, :n_frames]
    delta2 = delta2[:, :n_frames]
    rms = rms[:, :n_frames]
    log_f0 = log_f0[:n_frames, None].T
    log_f0_2 = log_f0_2[:n_frames, None].T
    sal2 = sal2[:n_frames, None].T
    second_f0_confidence = second_f0_confidence[:n_frames, None].T
    voiced_prob = voiced_prob[:n_frames, None].T
    delta_f0 = delta_f0[:, :n_frames]
    print('Arrays reshaped.')

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
            log_f0_2,
            sal2,
            second_f0_confidence,
            voiced_prob,
            centroid
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
