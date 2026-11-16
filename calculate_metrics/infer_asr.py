"""
Infer ASR model to get transcripts
"""

from pathlib import Path

import json
import numpy as np
import soundfile as sf
import soxr
import torch
from Levenshtein import opcodes
from tqdm import tqdm

import whisper


################################################################
# Definition of metrics
################################################################
def infer_asr(model, inf_file):
    """Calculate the Levenshtein distance between ref and inf ASR results.

    Args:
        model (torch.nn.Module): ASR model
        inf_file (str): audio file
    Returns:
        inf_txt (string): inferred transcript
    """

    with torch.no_grad():
        results = model.transcribe(
            inf_file,
            task="transcribe",
            beam_size=5,
            hallucination_silence_threshold=2.0,
            verbose=None,
        )
        inf_lang = results['language']
        inf_txt = results['text']

    return inf_txt, inf_lang


################################################################
# Main entry
################################################################
def main(args):
    print("""
          ======================================================================
          NOTE: please only run this on CLEAN speech to obtain reference transcripts. 
          If you want to calculate WER of enhanced speech, please directly run `calculate_wer.py` or `calculate_wer_whisper.py`!
          ======================================================================
          """)

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
    writer = {'text': (outdir / "text").open("w"),
              'lang': (outdir / "lang").open("w"),
              'note': (outdir / "_Note_These_results_are_calculated_by_Whisper_Large_v3_Not_Ground_Truth").open("w")}

    # Load Whisper-large-v3
    model = whisper.load_model(
        name="large-v3",
        device=args.device,
        download_root="./lib/WER/openai-whisper",
    )
    
    for uid, inf_audio in tqdm(data_pairs):
        _, inf_txt, inf_lang = process_one_pair(
            (uid, inf_audio), model=model
        )
        writer['text'].write(f"{uid} {inf_txt}\n")
        writer['lang'].write(f"{uid} {inf_lang}\n")

def process_one_pair(data_pair, model):
    uid, inf_path = data_pair

    inf_txt, inf_lang = infer_asr(model, inf_path)

    return uid, inf_txt, inf_lang


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
        help="Path to the output directory for writing inferred transcripts",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for running ASR inference",
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