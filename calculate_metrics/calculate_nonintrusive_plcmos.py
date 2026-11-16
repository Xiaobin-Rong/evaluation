from distutils.util import strtobool
from pathlib import Path
import soxr
import librosa
import numpy as np
import torch
from tqdm import tqdm
import soundfile as sf
from plcmos_utils import PLCMOSEstimator

METRICS = ("PLCMOS",)
TARGET_FS = 16000   

def str2bool(value: str) -> bool:
    return bool(strtobool(value))


################################################################
# Definition of metrics
################################################################
def plcmos_metric(model, audio_path):
    """Calculate the PLCMOS metric.

    Args:
        model (torch.nn.Module): PLCMOS model
        audio_path: path to the enhanced signal
    Returns:
        plcmos (float): PLCMOS value between [1, 5]
    """
    data, sr = sf.read(audio_path)
    if sr != TARGET_FS:
        data = soxr.resample(data, sr, TARGET_FS)
        sr = TARGET_FS
    plcmos_score = model.run(data, sr)
    return plcmos_score


################################################################
# Main entry
################################################################
def main(args):
    data_pairs = []
    with open(args.inf_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            data_pairs.append((uid, audio_path))

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

    plcmos_model = PLCMOSEstimator()
    
    ret = []
    for uid, inf_audio in tqdm(data_pairs):
        _, score = process_one_pair(
            (uid, inf_audio), plcmos_model=plcmos_model
        )
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


def process_one_pair(data_pair, plcmos_model=None):
    uid, inf_path = data_pair

    scores = {}
    for metric in METRICS:
        if metric == "PLCMOS":
            scores[metric] = plcmos_metric(plcmos_model, inf_path)
        else:
            raise NotImplementedError(metric)

    return uid, scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
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
        help="Device for running non-instrusive MOS calculation",
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

    group = parser.add_argument_group("MOS model related")
    group.add_argument(
        "--plcmos_tag",
        type=str,
        default="2",
        help="Tag of the PLCMOS model to be used",
    )
    args = parser.parse_args()

    main(args)
