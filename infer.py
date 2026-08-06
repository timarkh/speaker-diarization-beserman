import math

from process_audio import load_audio, extract_features
import numpy as np
import torch
from scipy.signal import find_peaks


def create_chunks(features, frame_times, chunk_size=1000, overlap=500):
    """
    Split a recording into overlapping chunks.

    Parameters
    ----------
    features : ndarray
        Shape (n_frames, feature_dim)

    frame_times : ndarray
        Shape (n_frames,)

    chunk_size : int

    overlap : int

    Returns
    -------
    feature_chunks : ndarray
        Shape (n_chunks, chunk_size, feature_dim)

    time_chunks : ndarray
        Shape (n_chunks, chunk_size)

    start_frames : ndarray
        Shape (n_chunks,)
    """
    stats = np.load("normalization.npz")
    mean = stats["mean"]
    std = stats["std"]
    features = (features - mean) / std

    step = chunk_size - overlap

    if step <= 0:
        raise ValueError("overlap must be smaller than chunk_size")

    n_frames = features.shape[0]
    feat_dim = features.shape[1]

    feature_chunks = []
    time_chunks = []
    start_frames = []

    start = 0

    while start < n_frames:
        end = start + chunk_size
        feat = features[start:end]
        times = frame_times[start:end]

        if len(feat) < chunk_size:
            pad = chunk_size - len(feat)
            feat = np.pad(
                feat,
                ((0, pad), (0, 0)),
                mode="constant",
            )
            times = np.pad(
                times,
                (0, pad),
                mode="edge",
            )

        feature_chunks.append(feat)
        time_chunks.append(times)
        start_frames.append(start)

        start += step

    return (
        np.asarray(feature_chunks, dtype=np.float32),
        np.asarray(time_chunks),
        np.asarray(start_frames, dtype=np.int32),
    )


def get_segments(outputs, timeChunk, speakerCodes, minBoundaryDist, minClauseMs, chunk_size=1000, allowedSpeakers=None):
    """
    Get segments for one chunk of frames.
    :return: {speaker: [segments in seconds]}
    """
    minFramesBreak = 10
    segStarts = {speakerCodes[s - 1]: -1 for s in allowedSpeakers if s != 0}
    segEnds = {speakerCodes[s - 1]: -1 for s in allowedSpeakers if s != 0}
    framesSinceLastSeen = {speakerCodes[s - 1]: minFramesBreak + 1 for s in allowedSpeakers if s != 0}
    segments = {speakerCodes[s - 1]: [] for s in allowedSpeakers if s != 0}

    if allowedSpeakers:
        speaker_logits = outputs["speaker"].clone()
        mask = torch.ones(len(speakerCodes) + 1, dtype=torch.bool)
        mask[allowedSpeakers] = False
        speaker_logits[:, :, mask] = -1e9
        speaker = speaker_logits.argmax(dim=2).squeeze(0).cpu().numpy()
    else:
        speaker = outputs["speaker"].argmax(dim=2).squeeze(0).cpu().numpy()
    speech = outputs["speech"].argmax(dim=2).squeeze(0).cpu().numpy()
    # boundary = (
    #         torch.sigmoid(outputs["boundary"]) > 0.02
    # ).squeeze(0).cpu().numpy()

    boundary = torch.sigmoid(outputs["boundary"]).squeeze().cpu().numpy()
    bndPeaks, bndProps = find_peaks(
        boundary,
        height=0.02,
        distance=minBoundaryDist
    )
    # print([(t[0].item(), t[1].item())
    #        for t in zip(speech, speaker)])

    for iFrame in range(chunk_size):
        curSpeaker = speaker[iFrame]
        if curSpeaker > 0:
            curSpeaker = speakerCodes[curSpeaker - 1]
        curSpeech = speech[iFrame]
        curTime = timeChunk[iFrame]
        curBoundary = (iFrame in bndPeaks)
        noSpeech = (curSpeaker == 0 or curSpeech == 0)
        for spSeg in segStarts:
            if segStarts[spSeg] >= 0 and (noSpeech or spSeg != curSpeaker):
                # Maybe it's time to close this segment. Or maybe multiple speakers overlap,
                # so that the classifier produces different outputs intermittently
                if framesSinceLastSeen[spSeg] <= 0:
                    # If the previous frame was still attributed to a speaker,
                    # then remember this timestamp as a potential end of the segment
                    segEnds[spSeg] = curTime
                framesSinceLastSeen[spSeg] += 1
                if framesSinceLastSeen[spSeg] > minFramesBreak or (noSpeech and framesSinceLastSeen[spSeg] > 2):
                    framesSinceLastSeen[spSeg] = minFramesBreak + 1
                    segments[spSeg].append([segStarts[spSeg], segEnds[spSeg]])
                    segStarts[spSeg] = -1
                    segEnds[spSeg] = -1
            elif spSeg == curSpeaker:
                segEnds[spSeg] = curTime
                if segStarts[spSeg] < 0:
                    segStarts[spSeg] = curTime
                elif (curBoundary and framesSinceLastSeen[spSeg] == 0
                      and segEnds[spSeg] - segStarts[spSeg] >= minClauseMs):
                    # Clause boundary within one person's speech
                    segments[spSeg].append([segStarts[spSeg], segEnds[spSeg]])
                    segStarts[spSeg] = curTime
                framesSinceLastSeen[spSeg] = 0
    for spSeg in segStarts:
        if segEnds[spSeg] > segStarts[spSeg] > 0:
            segments[spSeg].append([segStarts[spSeg], segEnds[spSeg]])
    for sp in segments:
        for i in range(len(segments[sp])):
            segments[sp][i] = [int(round(seg.item())) for seg in segments[sp][i]]
    return segments


def overlap(seg1, seg2, intersectionThreshold=10, shortThreshold=500):
    if seg1[0] <= seg2[0] <= seg1[1] - intersectionThreshold:
        # Overlap
        return True
    elif (seg1[1] + 0.5 * shortThreshold >= seg2[0] > seg1[1] - intersectionThreshold
          and (seg2[1] - seg2[0] < shortThreshold or seg1[1] - seg1[0] < shortThreshold)):
        # Very short segment adjacent to another one
        return True
    return False


def find_intersections(segments, wavLen):
    """
    Join overlapping segments
    """
    wavLen = math.ceil(wavLen)
    minSegLen = 500
    segments.sort()
    usedSegIndexes = set()
    newSegments = []
    for i in range(len(segments)):
        curSeg = segments[i]
        if i in usedSegIndexes or curSeg[0] > wavLen - minSegLen:
            continue
        if curSeg[1] > wavLen:
            curSeg[1] = wavLen
        bJoined = False
        for j in range(i + 1, len(segments)):
            if j in usedSegIndexes:
                continue
            if segments[j][1] > wavLen:
                segments[j][1] = wavLen
            if overlap(curSeg, segments[j]):
                curSeg = [min(curSeg[0], segments[j][0]), max(curSeg[1], segments[j][1])]
                newSegments.append(curSeg)
                usedSegIndexes.add(i)
                usedSegIndexes.add(j)
                bJoined = True
        if not bJoined:
            if curSeg[1] - curSeg[0] < minSegLen:
                delta = minSegLen - (curSeg[1] - curSeg[0])
                curSeg[0] = max(0, int(round(curSeg[0] - delta / 2)))
                curSeg[1] = min(wavLen, int(round(curSeg[1] + delta / 2)))
            newSegments.append(curSeg)
    if len(newSegments) < len(segments):
        # Repeat until there is nothing to join anymore
        return find_intersections(newSegments, wavLen)
    return newSegments


def diarize_wav(model,
                fnameWav,
                speakerCodes,
                chunk_size=1000,
                overlap=500,
                hop_length=160,
                sr=16000,
                minClauseMs=400,
                allowedSpeakers=None):
    if 0 not in allowedSpeakers:
        allowedSpeakers.append(0)   # Silence

    frame_ms = hop_length * 1000 / sr
    minBoundaryDist = int(round(minClauseMs / frame_ms))
    waveform = load_audio(fnameWav, sr=sr)
    duration = 1000 * len(waveform) / sr
    features, frame_times = extract_features(waveform, hop_length=hop_length)
    feature_chunks, time_chunks, start_frames = create_chunks(features, frame_times,
                                                              chunk_size=chunk_size, overlap=overlap)
    # all_outputs = []
    segments = {speakerCodes[s - 1]: [] for s in allowedSpeakers if s != 0}

    with torch.no_grad():
        for iChunk in range(len(feature_chunks)):
            chunk = feature_chunks[iChunk]

            x = torch.from_numpy(chunk).unsqueeze(0)
            outputs = model(x)
            # all_outputs.append(outputs)
            curSegments = get_segments(
                outputs,
                time_chunks[iChunk],
                speakerCodes,
                minBoundaryDist,
                minClauseMs,
                chunk_size=chunk_size,
                allowedSpeakers=allowedSpeakers
            )
            for sp in segments:
                segments[sp] += curSegments[sp]
    for sp in segments:
        segments[sp] = find_intersections(segments[sp], duration)
        segments[sp].sort()
    # segmentsSec = {
    #     sp: [[seg[0] / 1000, seg[1] / 1000] for seg in segments[sp]]
    #     for sp in segments
    # }
    # for sp, segSp in segmentsSec.items():
    #     print(f"      {sp:20s} {str(segSp)}")

    return segments
