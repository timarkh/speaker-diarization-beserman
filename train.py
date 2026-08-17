import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def prepare_labels(
    annotations,
    frame_times,
    speaker_map,
    boundary_width_ms=20,
):
    """
    Convert ELAN annotations into frame-level labels.

    Parameters
    ----------
    annotations : dict
        {speaker_name: [annotation, ...]}

    frame_times : ndarray
        Frame centre times in milliseconds.

    speaker_map : dict
        Speaker name -> integer ID.

    boundary_width_ms : float
        Mark boundaries within ±boundary_width_ms.

    Returns
    -------
    speech : uint8
    speaker : uint16
    boundary : uint8
    """

    n_frames = len(frame_times)

    speech = np.zeros(n_frames, dtype=np.uint8)
    speaker = np.zeros(n_frames, dtype=np.uint16)
    speaker2 = np.zeros(n_frames, dtype=np.uint16)
    boundary = np.zeros(n_frames, dtype=np.uint8)

    annotations_shuffled = []

    for speaker_name in sorted(speaker_map):
        if speaker_name not in annotations:
             continue
        for ann in annotations[speaker_name]:
            annotations_shuffled.append((speaker_map[speaker_name], ann["start_ms"], ann["end_ms"]))
    random.shuffle(annotations_shuffled)

    for ann in annotations_shuffled:
        spID = ann[0]
        start = ann[1]
        end = ann[2]

        # frames belonging to this clause
        mask = (frame_times >= start) & (frame_times < end)
        speech[mask] = 1
        speaker[mask] = spID
        # print(mask, speaker[mask])

        # # boundary window
        # bmask = (
        #     frame_times >= end - boundary_width_ms
        # ) & (
        #     frame_times <= end + boundary_width_ms
        # )
        #
        # boundary[bmask] = 1

        boundary_frame = np.argmin(np.abs(frame_times - end))
        for iFrame in range(max(0, boundary_frame - 5), min(len(boundary), boundary_frame + 5)):
            boundary[iFrame] = 1.0 / (1 + abs(iFrame - boundary_frame))

    #Now, fill speaker2 (second speaker talking at the same time, if any)
    for ann in annotations_shuffled[::-1]:
        spID = ann[0]
        start = ann[1]
        end = ann[2]
        mask = (frame_times >= start) & (frame_times < end)
        speaker2[mask] = spID

    return speech, speaker, speaker2, boundary

def write_training_dataset(
    outfile,
    features,
    speech,
    speaker,
    speaker2,
    boundary,
    chunk_size=1000,
    overlap=500,
):
    """
    Split a recording into overlapping chunks and save as a compressed NPZ.

    Parameters
    ----------
    outfile : str or Path

    features : ndarray
        (frames, feature_dim)

    speech : ndarray
        (frames,)

    speaker : ndarray
        (frames,)

    speaker2 : ndarray
        (frames,)

    boundary : ndarray
        (frames,)

    chunk_size : int
        Frames per training example.

    overlap : int
        Number of overlapping frames.

    Returns
    -------
    Number of chunks written.
    """

    step = chunk_size - overlap

    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    n_frames = features.shape[0]

    X = []
    Y_speech = []
    Y_speaker = []
    Y_speaker2 = []
    Y_boundary = []

    start = 0
    while start < n_frames:
        end = start + chunk_size
        x = features[start:end]
        s = speech[start:end]
        sp = speaker[start:end]
        sp2 = speaker2[start:end]
        b = boundary[start:end]

        # pad the final chunk if necessary
        if len(x) < chunk_size:
            pad = chunk_size - len(x)
            x = np.pad(
                x,
                ((0, pad), (0, 0)),
                mode="constant",
            )
            s = np.pad(s, (0, pad))
            sp = np.pad(sp, (0, pad))
            sp2 = np.pad(sp2, (0, pad))
            b = np.pad(b, (0, pad))

        X.append(x)
        Y_speech.append(s)
        Y_speaker.append(sp)
        Y_speaker2.append(sp2)
        Y_boundary.append(b)
        start += step

    X = np.asarray(X, dtype=np.float32)
    Y_speech = np.asarray(Y_speech, dtype=np.uint8)
    Y_speaker = np.asarray(Y_speaker, dtype=np.uint16)
    Y_speaker2 = np.asarray(Y_speaker2, dtype=np.uint16)
    Y_boundary = np.asarray(Y_boundary, dtype=np.uint8)

    np.savez_compressed(
        outfile,
        features=X,
        speech=Y_speech,
        speaker=Y_speaker,
        speaker2=Y_speaker2,
        boundary=Y_boundary,
        sample_rate=16000,
        hop_length=160,
        n_fft=400,
        n_mels=40,
        chunk_size=chunk_size,
        overlap=overlap
    )

    return len(X)


class ClauseSegmentationModel(nn.Module):
    def __init__(
        self,
        input_size=128,
        hidden_size=96,
        num_layers=2,
        num_speakers=11,
        dropout=0.42,
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_size, hidden_size)

        self.encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        encoder_dim = hidden_size * 2

        self.speaker_head = nn.Linear(encoder_dim, num_speakers)
        self.speech_head = nn.Linear(encoder_dim, 2)
        self.boundary_head = nn.Linear(encoder_dim, 1)

    def forward(self, x):
        """
        Parameters
        ----------
        x : FloatTensor
            (batch, frames, 128)

        Returns
        -------
        dict containing logits
        """

        x = self.input_projection(x)
        x, _ = self.encoder(x)

        return {
            "speaker": self.speaker_head(x),
            "speech": self.speech_head(x),
            "boundary": self.boundary_head(x),
        }

class SpeechDataset(Dataset):
    def __init__(self, directory):
        stats = np.load("normalization.npz")
        self.normMean = stats["mean"]
        self.normStd = stats["std"]

        self.samples = []
        for file in sorted(Path(directory).glob("*.npz")):
            data = np.load(file)
            X = data["features"]
            speech = data["speech"]
            speaker = data["speaker"]
            speaker2 = data["speaker2"]
            boundary = data["boundary"]

            mask = data["mask"] if "mask" in data else None     # mask silence when a segment is at the beginning or the end of a recording

            for i in range(X.shape[0]):
                self.samples.append((
                    X[i],
                    speech[i],
                    speaker[i],
                    speaker2[i],
                    boundary[i],
                    None if mask is None else mask[i]
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, speech, speaker, speaker2, boundary, mask = self.samples[idx]
        x = (x - self.normMean) / self.normStd
        sample = {
            "features": torch.from_numpy(x).float(),
            "speech": torch.from_numpy(speech).long(),
            "speaker": torch.from_numpy(speaker).long(),
            "speaker2": torch.from_numpy(speaker2).long(),
            "boundary": torch.from_numpy(boundary).float(),
        }
        if mask is not None:
            sample["mask"] = torch.from_numpy(mask).bool()
        return sample


def calculate_mean_std():
    sum_x = None
    sum_x2 = None
    count = 0

    for file in Path("prepared/train").glob("*.npz"):
        data = np.load(file)
        X = data["features"]
        X = X.reshape(-1, X.shape[-1])
        if sum_x is None:
            sum_x = np.zeros(X.shape[1], dtype=np.float64)
            sum_x2 = np.zeros(X.shape[1], dtype=np.float64)
        sum_x += X.sum(axis=0)
        sum_x2 += (X * X).sum(axis=0)
        count += X.shape[0]

    mean = sum_x / count
    var = sum_x2 / count - mean ** 2
    std = np.sqrt(np.maximum(var, 1e-8))
    np.savez(
        "normalization.npz",
        mean=mean,
        std=std
    )


def plot_learning_curve(train_losses, valid_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(8, 5))

    plt.plot(
        epochs,
        train_losses,
        marker='o',
        label='Training loss'
    )

    plt.plot(
        epochs,
        valid_losses,
        marker='s',
        label='Validation loss'
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig("loss.png", dpi=150)


def soft_speaker_loss(logits, speaker, speaker2, secondary_weight=0.2):
    log_probs = torch.log_softmax(logits, dim=-1)
    primary_loss = -log_probs[
        torch.arange(logits.shape[0], device=logits.device),
        speaker
    ]
    secondary_loss = -log_probs[
        torch.arange(logits.shape[0], device=logits.device),
        speaker2
    ]

    # If speaker2 == speaker, don't count the secondary speaker separately.
    has_second = speaker2 != speaker
    loss = primary_loss.clone()
    loss[has_second] += (
        secondary_weight * secondary_loss[has_second]
    )

    # Normalize so that a single-speaker frame has weight 1
    # and a two-speaker frame has weight 1 + secondary_weight.
    weights = torch.ones_like(loss)
    weights[has_second] += secondary_weight

    return (loss / weights).mean()


def train_model(model, num_speakers, max_epochs=100, early_stop=5, min_delta=1e-3):
    trainLoader = DataLoader(
        SpeechDataset("prepared/train"),
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    validLoader = DataLoader(
        SpeechDataset("prepared/validation"),
        batch_size=16,
        shuffle=True,
        num_workers=0
    )
    # speaker_loss_fn = torch.nn.CrossEntropyLoss()
    speech_loss_fn = torch.nn.CrossEntropyLoss()
    boundary_loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight = torch.tensor([30]))  # Assuming non-boundary frames occur 30 times more often
    # boundary_loss_fn = torch.nn.MSELoss() # For Gaussian-like distributions of probabilities around peaks
    # boundary_loss_fn = torch.nn.SmoothL1Loss() # An alternative
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=2e-5,
        weight_decay=1e-5,
    )

    train_losses = []
    valid_losses = []
    bestValidLoss = 9999999.0
    badValidEpochs = 0
    prevValidLoss = 0
    model.train()

    for epoch in range(max_epochs):
        # Train step, train loss
        print('Start epoch', epoch, '...')
        epoch_loss = 0.0
        epoch_speaker_loss = 0.0
        epoch_speech_loss = 0.0
        epoch_boundary_loss = 0.0

        for batch in trainLoader:
            optimizer.zero_grad()

            outputs = model(batch["features"])

            speaker_loss = soft_speaker_loss(
                outputs["speaker"].reshape(-1, num_speakers),
                batch["speaker"].reshape(-1),
                batch["speaker2"].reshape(-1)
            )

            speech_loss = speech_loss_fn(
                outputs["speech"].reshape(-1, 2),
                batch["speech"].reshape(-1)
            )

            boundary_loss = boundary_loss_fn(
                outputs["boundary"].reshape(-1),
                batch["boundary"].reshape(-1)
            )

            loss = (
                    1.0 * speaker_loss
                    + 0.7 * speech_loss
                    + 2.0 * boundary_loss
            )

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_speaker_loss += speaker_loss.item()
            epoch_speech_loss += speech_loss.item()
            epoch_boundary_loss += boundary_loss.item()

        # Calculate validation loss
        model.eval()
        valid_epoch_loss = 0.0
        valid_epoch_speaker_loss = 0.0
        valid_epoch_speech_loss = 0.0
        valid_epoch_boundary_loss = 0.0

        with torch.no_grad():
            for batch in validLoader:
                outputs = model(batch["features"])

                speaker_loss = soft_speaker_loss(
                    outputs["speaker"].reshape(-1, num_speakers),
                    batch["speaker"].reshape(-1),
                    batch["speaker2"].reshape(-1)
                )

                speech_loss = speech_loss_fn(
                    outputs["speech"].reshape(-1, 2),
                    batch["speech"].reshape(-1)
                )

                boundary_loss = boundary_loss_fn(
                    outputs["boundary"].reshape(-1),
                    batch["boundary"].reshape(-1)
                )

                loss = (
                        1.0 * speaker_loss
                        + 0.7 * speech_loss
                        + 2.0 * boundary_loss
                )

                valid_epoch_loss += loss.item()
                valid_epoch_speaker_loss += speaker_loss.item()
                valid_epoch_speech_loss += speech_loss.item()
                valid_epoch_boundary_loss += boundary_loss.item()

        print(
            epoch + 1,
            '\n======  TRAIN LOSS  ======\n',
            epoch_loss / len(trainLoader),
            '\nSpeaker loss:', epoch_speaker_loss / len(trainLoader),
            '\nSpeech / no speech loss:', epoch_speech_loss / len(trainLoader),
            '\nSegment boundary loss:', epoch_boundary_loss / len(trainLoader),
            '\n===== VALIDATION LOSS =====\n',
            valid_epoch_loss / len(validLoader),
            '\nSpeaker loss:', valid_epoch_speaker_loss / len(validLoader),
            '\nSpeech / no speech loss:', valid_epoch_speech_loss / len(validLoader),
            '\nSegment boundary loss:', valid_epoch_boundary_loss / len(validLoader)
        )
        train_losses.append(epoch_loss / len(trainLoader))
        valid_losses.append(valid_epoch_loss / len(validLoader))
        plot_learning_curve(train_losses, valid_losses)
        if valid_epoch_loss < bestValidLoss:
            if valid_epoch_loss < bestValidLoss - min_delta:
                badValidEpochs = 0
            bestValidLoss = valid_epoch_loss
            torch.save(
                model.state_dict(),
                "best_model.pt"
            )
        else:
            if valid_epoch_loss > prevValidLoss:
                print('Validation loss did not decrease. :(')
                badValidEpochs += 1
                if badValidEpochs >= early_stop:
                    print('Stopping the training early.')
                    break
        prevValidLoss = valid_epoch_loss