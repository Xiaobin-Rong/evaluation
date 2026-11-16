"""
Calculate SpkSim using WavLM-BASE+-SV
"""

from pathlib import Path

import numpy as np
import soundfile as sf
import soxr
import torch
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

METRICS = ("SpeakerSimilarity",)
TARGET_FS = 16000


################################################################
# Definition of metrics
################################################################
def speaker_similarity_metric(processor, model, ref, inf, fs=16000):
    """Calculate the cosine similarity between ref and inf speaker embeddings.

    Args:
        processor (WavLMProcessor): processor for WavLM-BASE+-SV
        model (WavLMModel): WavLM-BASE+-SV model
        ref (np.ndarray): reference signal (time,)
        inf (np.ndarray): enhanced signal (time,)
        fs (int): sampling rate in Hz
    Returns:
        similarity (float): cosine similarity value between [0, 1]
    """
    if fs != TARGET_FS:
        ref = soxr.resample(ref, fs, TARGET_FS)
        inf = soxr.resample(inf, fs, TARGET_FS)

    # Convert numpy arrays to torch tensors and process
    ref_tensor = processor(ref, padding=True, return_tensors="pt", sampling_rate=TARGET_FS).to(next(model.parameters()).device)
    inf_tensor = processor(inf, padding=True, return_tensors="pt", sampling_rate=TARGET_FS).to(next(model.parameters()).device)

    with torch.no_grad():
        ref_emb = model(**ref_tensor).embeddings
        inf_emb = model(**inf_tensor).embeddings

        similarity = torch.cosine_similarity(ref_emb, inf_emb).item()

    return similarity


################################################################
# Main entry
################################################################
def main(args):
    refs = {}
    with open(args.ref_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            refs[uid] = audio_path

    data_pairs = []
    with open(args.inf_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            data_pairs.append((uid, refs[uid], audio_path))

    size = len(data_pairs)
    assert 1 <= args.job <= args.nsplits <= size
    interval = size // args.nsplits
    start = (args.job - 1) * interval
    end = size if args.job == args.nsplits else start + interval
    data_pairs = data_pairs[start:end]
    print(
        f"[Job {args.job}/{args.nsplits}] Processing ({len(data_pairs)}/{size}) samples",
        flush=True,
    )
    suffix = "" if args.nsplits == args.job == 1 else f".{args.job}"

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    writers = {
        metric: (outdir / f"{metric}{suffix}.scp").open("w") for metric in METRICS
    }

    # Load WavLM-BASE+-SV model and processor
    processor = Wav2Vec2FeatureExtractor.from_pretrained("/data/hdd0/Evaluation/lib/SpeakerSimilarity/wavlm-base-plus-sv/")
    model = WavLMForXVector.from_pretrained("/data/hdd0/Evaluation/lib/SpeakerSimilarity/wavlm-base-plus-sv/").to(args.device)
    model.eval()

    ret = []
    for uid, ref_audio, inf_audio in tqdm(data_pairs):
        _, score = process_one_pair((uid, ref_audio, inf_audio), processor=processor, model=model)
        ret.append((uid, score))
        for metric, value in score.items():
            writers[metric].write(f"{uid} {value}\n")

    for metric in METRICS:
        writers[metric].close()

    if args.nsplits == args.job == 1:
        with (outdir / "RESULTS.txt").open("w") as f:
            for metric in METRICS:
                mean_score = np.nanmean([score[metric] for uid, score in ret])
                f.write(f"{metric}: {mean_score:.4f}\n")
        print(
            f"Overall results have been written in {outdir / 'RESULTS.txt'}", flush=True
        )


def process_one_pair(data_pair, processor, model):
    uid, ref_path, inf_path = data_pair
    ref, fs = sf.read(ref_path, dtype="float32")
    inf, fs2 = sf.read(inf_path, dtype="float32")
    assert fs == fs2, (fs, fs2)
    assert ref.shape == inf.shape, (ref.shape, inf.shape)
    assert ref.ndim == 1, ref.shape

    scores = {}
    for metric in METRICS:
        if metric == "SpeakerSimilarity":
            scores[metric] = speaker_similarity_metric(processor, model, ref, inf, fs=fs)
        else:
            raise NotImplementedError(metric)

    return uid, scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref_scp",
        type=str,
        required=True,
        help="Path to the scp file containing reference signals",
    )
    parser.add_argument(
        "--inf_scp",
        type=str,
        required=True,
        help="Path to the scp file containing enhanced signals",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output directory for writing metrics",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for running speaker embedding extraction",
    )
    parser.add_argument(
        "--nsplits",
        type=int,
        default=1,
        help="Total number of computing nodes to speed up evaluation",
    )
    parser.add_argument(
        "--job",
        type=int,
        default=1,
        help="Index of the current node (starting from 1)",
    )
    args = parser.parse_args()

    main(args)
