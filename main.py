import re
import os
import random
import shutil
from pathlib import Path
import pympi
import torch

from process_audio import extract_features, load_audio, augment_waveform
from train import prepare_labels, write_training_dataset, ClauseSegmentationModel, train_model, calculate_mean_std
from infer import diarize_wav


# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

CORPUS_DIR = Path("./corpus")

AUDIO_EXTENSIONS = [
    ".wav",
    ".WAV",
    ".mp3",
    ".MP3",
    ".ogg",
    ".OGG",
    ".m4a",
    ".M4A",
    ".wma",
    ".M4A",
    ".mp4",
    ".MP4"
]

SPEAKER_CODES = []
with open('speaker_codes.txt', 'r', encoding='utf-8') as fIn:
    for line in fIn:
        line = line.strip()
        if not line:
            continue
        SPEAKER_CODES.append(line)

speakerMap = {
    '<silence>': 0
}
for i in range(len(SPEAKER_CODES)):
    speakerMap[SPEAKER_CODES[i]] = i + 1

# ----------------------------------------------------------------------
# Utilities
# ----------------------------------------------------------------------

def find_audio_file(fnameEaf):
    """
    Find an audio file with the same basename as the ELAN file.
    """
    fnameEaf = re.sub('/eaf/', '/sound/', fnameEaf)
    for ext in AUDIO_EXTENSIONS:
        candidate = fnameEaf[:-4] + ext
        if os.path.exists(candidate):
            return Path(candidate)
    return None


def parse_eaf(eaf_path: Path):
    """
    Read an ELAN file and return a simple representation.
    """

    eaf = pympi.Elan.Eaf(str(eaf_path))
    tiers = {}
    for tier_name in eaf.get_tier_names():
        tier_name = re.sub('@MBe?\\b', '@MAB81', tier_name)
        if not tier_name.startswith('tx@'):
            continue
        speakerCode = tier_name[3:]
        if not tier_name.endswith(tuple(['@' + sc for sc in SPEAKER_CODES])):
            speakerCode = 'other'
        if speakerCode not in tiers:
            tiers[speakerCode] = []
        annots = eaf.get_annotation_data_for_tier(tier_name)
        for begin, end, value in annots:
            tiers[speakerCode].append({
                "start_ms": begin,
                "end_ms": end,
                "text": value
            })
    print('EAF parsing done. Tiers: ', [sp for sp in sorted(tiers)])
    return tiers

def create_eaf(segments, fnameOut, fnameWav):
    """
    Create an ELAN file based on the inferred segments.

    Parameters
    ----------
    segments : dict
        Example:
        {
            "AB": [[1000, 1400], [1450, 1700]],
            "CD": [[1800, 2500]]
        }

    output_file : str
    """
    if not fnameOut.endswith('.eaf'):
        return
    eaf = pympi.Elan.Eaf()
    eaf.add_linguistic_type(
        "transcription",
        timealignable=True
    )
    eaf.add_linguistic_type(
        "translation",
        constraints="Symbolic_Association",
        timealignable=False
    )
    eaf.add_linguistic_type(
        "privacy",
        timealignable=True
    )
    eaf.add_linked_file(
        file_path='./' + re.sub('.*[/\\\\]', '', fnameWav),
        relpath='./' + re.sub('.*[/\\\\]', '', fnameWav),
        mimetype='audio/x-wav'
    )
    for speaker in sorted(segments, key=lambda s: (-len(s), s)):
        intervals = segments[speaker]
        if len(intervals) <= 0:
            continue
        tx = f"tx@{speaker}"
        en = f"ft_en@{speaker}"
        ru = f"ft_ru@{speaker}"

        eaf.add_tier(
            tx,
            ling="transcription",
            part=speaker
        )
        eaf.add_tier(
            en,
            ling="translation",
            parent=tx,
            part=speaker
        )
        eaf.add_tier(
            ru,
            ling="translation",
            parent=tx,
            part=speaker
        )

        for start_ms, end_ms in intervals:
            eaf.add_annotation(
                tx,
                start_ms,
                end_ms,
                value=""
            )
            eaf.add_ref_annotation(
                en,
                tx,
                start_ms,
                value=""
            )
            eaf.add_ref_annotation(
                ru,
                tx,
                start_ms,
                value=""
            )
    eaf.to_file(fnameOut)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def preprocess_eaf(sr=16000, overwrite=False):
    if overwrite:
        if os.path.exists('prepared/train'):
            shutil.rmtree('prepared/train')
        os.makedirs('prepared/train')
        if os.path.exists('prepared/validation'):
            shutil.rmtree('prepared/validation')
        os.makedirs('prepared/validation')

    for root, dirs, files in os.walk(os.path.join(CORPUS_DIR, 'eaf')):
        for fnameEaf in files:
            if not fnameEaf.lower().endswith('.eaf'):
                continue
            fnameEafFull = os.path.join(root, fnameEaf)
            print("=" * 70)
            print(fnameEafFull)
            pathSound = find_audio_file(fnameEafFull)

            if pathSound is None:
                print("  ERROR: matching audio file not found.")
                continue

            print(f"  Audio : {pathSound.name}")
            tiers = parse_eaf(Path(fnameEafFull))
            fnameNpzTrain = os.path.join('prepared/train', fnameEaf[:-3] + 'npz')
            fnameNpzValid = os.path.join('prepared/validation', fnameEaf[:-3] + 'npz')
            if os.path.exists(fnameNpzTrain) or os.path.exists(fnameNpzValid):
                print('Skipping this file.')
                continue
            if 'other' in tiers and len(tiers['other']) > 0.8 * sum(len(t) for t in tiers):
                print('Skipping this file because it contains too many unknown speakers.')
                continue

            if random.random() < 0.13 and len(tiers) >= 2:
                outPath = 'prepared/validation'
            else:
                outPath = 'prepared/train'
            fnameNpz = os.path.join(outPath, fnameEaf[:-3] + 'npz')

            waveform = load_audio(pathSound, sr=sr)
            print('Waveform loaded, size: ', waveform.size)
            if fnameNpz == fnameNpzTrain:
                waveform = augment_waveform(waveform)
                # Only augment training data

            duration = len(waveform) / (sr * 60)

            print(f"  Duration    : {duration:.1f} min")
            print(f"  Samples     : {len(waveform)}")
            print(f"  Tiers       : {len(tiers)}")

            features, frameTimes = extract_features(waveform)
            speech, speaker, speaker2, boundary = prepare_labels(tiers, frameTimes, speakerMap)
            n = write_training_dataset(
                fnameNpz,
                features,
                speech,
                speaker,
                speaker2,
                boundary,
            )
            print(f"{n} training examples written.")

            for tier_name, annots in tiers.items():
                print(f"      {tier_name:20s} {len(annots):6d} annotations")

    calculate_mean_std()
    print("\nPreprocessing done.")


def main(preprocess=True, overwrite=False):
    if preprocess:
        preprocess_eaf(overwrite=overwrite)
    model = ClauseSegmentationModel(num_speakers=len(speakerMap))
    train_model(model, num_speakers=len(speakerMap))
    print("\nTraining done.")


def load_model():
    device = torch.device("cpu")
    model = ClauseSegmentationModel(
        input_size=128,
        hidden_size=96,
        num_layers=2,
        num_speakers=len(speakerMap),
    )
    model.load_state_dict(
        torch.load("best_model.pt", map_location=device)
    )
    model.eval()
    return model


def process_wav(model, fnameWav, allowedSpeakers=None, speakerCollate=None):
    segments = diarize_wav(model, fnameWav, SPEAKER_CODES,
                           allowedSpeakers=allowedSpeakers, speakerCollate=speakerCollate)
    fnameElan = re.sub('\\.[^.]*$', '.eaf', fnameWav)
    create_eaf(segments, fnameElan, fnameWav)


def process_wav_files(model, others=False):
    """
    Generate EAF for all WAV files in ./wav.
    Allow only speakers whose codes are extracted from the filename,
    e.g., AL_RA-xxxxx-MU-xxxx.wav translates to AL, RA, Interviewer_MU.
    If others == True, then also allow the 'other' class.
    """
    for fname in os.listdir('wav'):
        if not fname.lower().endswith('.wav'):
            continue
        m = re.search('^([^-]+)-[^-]+-([^-]+).*?\\.wav', fname, flags=re.I)
        if not m:
            print('Bad file name:', fname)
            continue
        participantCodes = list(m.group(1).split('_')) + ['Interviewer_' + pn for pn in m.group(2).split('_')]
        participantIdx = set()
        if others:
            participantIdx.add('other')
        for pc in participantCodes:
            if pc not in SPEAKER_CODES:
                pc = 'other'
            participantIdx.add(1 + SPEAKER_CODES.index(pc))
        allowedSpeakers = [pi for pi in sorted(participantIdx)]
        print('Processing', fname, ', allowed speakers:', allowedSpeakers)
        process_wav(model, os.path.join('wav', fname), allowedSpeakers=allowedSpeakers)


if __name__ == '__main__':
    main(preprocess=True, overwrite=True)
    model = load_model()
    process_wav(model, 'prepared/test/SN-2016.01.06-TA-germany.wav', allowedSpeakers=[0, 2, 12])
    process_wav(model, 'prepared/test/RA-2021.06-MU-borba_s_vreditelyami.wav', allowedSpeakers=[0, 1, 5])
    # process_wav_files(model)
    # process_wav(model, 'prepared/test/RSK67-2026.08.03-MU-perepesj_piroshki_kwarnjanj.wav',
    #             allowedSpeakers=[0, 1, 17],
    #             speakerCollate={
    #                 'AS': 'other',
    #                 'AL': 'other',
    #                 'RA': 'other',
    #                 'TS': 'other',
    #                 'MAB81': 'other',
    #                 'AVB62': 'other',
    #                 'NV': 'other',
    #                 'IM': 'other',
    #                 'AVS': 'other',
    #                 'ANS': 'other'
    #             })
    # diarize_wav(model, 'corpus/sound/2022/RA_TS-2022.07.13-MU-experiment_partitives_anim.wav', SPEAKER_CODES)
