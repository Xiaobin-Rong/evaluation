"""
English only, using Whisper ASR model (Whisper-large-v3) via OpenAI's repo.
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
from espnet2.text.cleaner import TextCleaner

METRICS = ("WER",)
TARGET_FS = 16000


################################################################
# Definition of metrics
################################################################
def levenshtein_metric(model, textcleaner, ref_txt, inf_file):
    """Calculate the Levenshtein distance between ref and inf ASR results."""
    if ref_txt == "<not-available>":
        return {}
    
    with torch.no_grad():
        results = model.transcribe(
            inf_file,
            task="transcribe",
            language="en",
            beam_size=5,
            hallucination_silence_threshold=2.0,
            verbose=None,
        )
        inf_txt = results['text']

    ref_txt = textcleaner(ref_txt)
    inf_txt = textcleaner(inf_txt)
    
    ref_words = ref_txt.split()
    inf_words = inf_txt.split()

    ret = {
        "hyp_text": inf_txt,
        "ref_text": ref_txt,
        "delete": 0,
        "insert": 0,
        "replace": 0,
        "equal": 0,
    }
    for op, ref_st, ref_et, inf_st, inf_et in opcodes(ref_words, inf_words):
        if op == "insert":
            ret[op] = ret[op] + inf_et - inf_st
        else:
            ret[op] = ret[op] + ref_et - ref_st
    total = ret["delete"] + ret["replace"] + ret["equal"]
    assert total == len(ref_words), (total, len(ref_words))
    total = ret["insert"] + ret["replace"] + ret["equal"]
    assert total == len(inf_words), (total, len(inf_words))
    return ret


################################################################
# Main entry
################################################################
def main(args):
    transcripts = {}
    if args.meta_tsv.endswith(".tsv"):
        with open(args.meta_tsv, "r") as f:
            headers = next(f).strip().split("\t")
            uid_idx = headers.index("id")
            txt_idx = headers.index("text")
            for line in f:
                tup = line.strip().split("\t")
                uid, txt = tup[uid_idx], tup[txt_idx]
                transcripts[uid] = txt
    else:
        with open(args.meta_tsv, "r") as f:
            for line in f:
                uid, txt = line.strip().split(maxsplit=1)
                transcripts[uid] = txt

    data_pairs = []
    with open(args.inf_scp, "r") as f:
        for line in f:
            uid, audio_path = line.strip().split()
            data_pairs.append((uid, transcripts[uid], audio_path))

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    writers = {metric: (outdir / f"{metric}.scp").open("w") for metric in METRICS}

    # Load Whisper-large-v3
    model = whisper.load_model(
        name="large-v3",
        device=args.device,
        download_root="./lib/WER/openai-whisper",
    )
    textcleaner = TextCleaner("whisper_en")
            
    ret = []
    for uid, ref_text, inf_audio in tqdm(data_pairs):
        uid, score = process_one_pair((uid, ref_text, inf_audio), model=model, textcleaner=textcleaner)
        ret.append((uid, score))
        for metric, value in score.items():
            if metric == "WER":
                writers[metric].write(f"{uid} {json.dumps(value)}\n")

    for metric in METRICS:
        writers[metric].close()

    # Compute overall WER
    with (outdir / "RESULTS.txt").open("w") as f:
        for metric in METRICS:
            if metric == "WER":
                dic = {"delete": [], "insert": [], "replace": [], "equal": []}
                for uid, score in ret:
                    if len(score[metric]) == 0:
                        continue
                    for op in dic.keys():
                        dic[op].append(score[metric][op])
                dic = {op: sum(count) for op, count in dic.items()}
                numerator = dic["replace"] + dic["delete"] + dic["insert"]
                denominator = dic["replace"] + dic["delete"] + dic["equal"]
                wer = numerator / denominator
                f.write(f"{metric}: {wer:.4f}\n")
                for op, count in dic.items():
                    f.write(f"    {op}: {count}\n")


def process_one_pair(data_pair, model, textcleaner):
    uid, ref_txt, inf_path = data_pair
    scores = {metric: levenshtein_metric(model,textcleaner, ref_txt, inf_path) for metric in METRICS}
    return uid, scores


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--meta_tsv",
        type=str,
        required=True,
        help="Path to the tsv file containing meta information about the data "
        "(including transcripts)\n"
        "Alternatively, this can also be a scp file containing transcript per sample",
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
        help="Device for running ASR inference",
    )
    args = parser.parse_args()

    main(args)
