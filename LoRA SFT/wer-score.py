#!/usr/bin/env python3
"""
Compute WER for three hypotheses (ASR, random, exhaustive) against the ground truth,
using jiwer as requested.

Defaults expect files at:
- GT:        "aprocsa1944a GT.txt"
- ASR:       "aprocsa1944a Azure.txt"
- Random:    "outputs/qwen2p5-14b-random/aprocsa1944a_ASR_response_4k.txt" (falls back if missing)
- Exhaustive: "outputs/qwen2p5-14b-exhaustive/aprocsa1944a_ASR_response_4k.txt" (falls back if missing)

Outputs a readable summary and writes JSON to outputs/wer_scores.json
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict

import jiwer


def compute_wer_percent(gt_text: str, hyp_text: str) -> float:
    """Compute WER (%) using jiwer with the requested transforms.

    Steps:
    1) Remove punctuation from both reference (GT) and hypothesis strings.
    2) Call jiwer.wer using jiwer.wer_standardize for both reference and hypothesis.

    Returns a percentage value (0–100).
    """
    # Apply the exact sequence the user requested:
    # - Remove punctuation from both reference and hypothesis
    # - Use jiwer.wer with wer_standardize on both sides
    gt_proc = jiwer.RemovePunctuation()(gt_text)
    hy_proc = jiwer.RemovePunctuation()(hyp_text)
    wer_val = jiwer.wer(
        gt_proc,
        hy_proc,
        reference_transform=jiwer.wer_standardize,
        hypothesis_transform=jiwer.wer_standardize,
    )
    return float(wer_val) * 100.0


def pick_first_existing(paths: List[str]) -> str:
    """Return the first path that exists from a list; otherwise return first item.

    Helps choose between several candidate output file names that may or may not
    exist depending on which generation command was run.
    """
    for p in paths:
        if os.path.exists(p):
            return p
    # If none exist, return the first so the subsequent file read fails clearly.
    return paths[0]


def main():
    """CLI entry-point: resolves file paths, computes WERs, prints and saves JSON."""
    ap = argparse.ArgumentParser(description="Compute WER of ASR, random, and exhaustive vs GT")
    ap.add_argument("--gt", default="aprocsa1944a GT.txt")
    ap.add_argument("--asr", default="aprocsa1944a Azure.txt")
    ap.add_argument("--random", dest="random_path", default=None, help="Path to random adapter output")
    ap.add_argument("--exhaustive", dest="exhaustive_path", default=None, help="Path to exhaustive adapter output")
    ap.add_argument("--out_json", default=os.path.join("outputs", "wer_scores.json"))
    args = ap.parse_args()

    # Resolve defaults for model outputs if not provided
    random_candidates = [
        os.path.join("outputs", "qwen2p5-14b-random", "aprocsa1944a_Azure_response_4k.txt"),
        os.path.join("outputs", "qwen2p5-14b-random", "aprocsa1944a Azure_response.txt"),
        os.path.join("outputs", "qwen2p5-14b-random", "aprocsa1944a_random_response.txt"),
    ]
    exhaustive_candidates = [
        os.path.join("outputs", "qwen2p5-14b-exhaustive", "aprocsa1944a_Azure_response_4k.txt"),
        os.path.join("outputs", "qwen2p5-14b-exhaustive", "aprocsa1944a Azure_response.txt"),
    ]
    random_path = args.random_path or pick_first_existing(random_candidates)
    exhaustive_path = args.exhaustive_path or pick_first_existing(exhaustive_candidates)

    # Read files
    def readf(p: str) -> str:
        with open(p, "r", encoding="utf-8") as f:
            return f.read()

    gt_txt = readf(args.gt)
    asr_txt = readf(args.asr)
    rand_txt = readf(random_path)
    exh_txt = readf(exhaustive_path)

    # Compute WERs (percent)
    asr_wer = compute_wer_percent(gt_txt, asr_txt)
    rand_wer = compute_wer_percent(gt_txt, rand_txt)
    exh_wer = compute_wer_percent(gt_txt, exh_txt)

    # Prepare results
    results: Dict[str, Dict[str, object]] = {
        "paths": {
            "gt": os.path.abspath(args.gt),
            "asr": os.path.abspath(args.asr),
            "random": os.path.abspath(random_path),
            "exhaustive": os.path.abspath(exhaustive_path),
        },
        "wer": {
            "asr": round(asr_wer, 3),
            "random": round(rand_wer, 3),
            "exhaustive": round(exh_wer, 3),
        },
    }

    # Print concise summary
    print("WER (%) vs GT")
    print(f"- ASR:       {results['wer']['asr']}")
    print(f"- Random:    {results['wer']['random']}")
    print(f"- exhaustive: {results['wer']['exhaustive']}")

    # Save JSON
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON to: {args.out_json}")


if __name__ == "__main__":
    main()
