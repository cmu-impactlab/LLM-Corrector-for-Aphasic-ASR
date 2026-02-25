#!/usr/bin/env python3
"""
Compare model outputs against a ground-truth transcript and write CSV/JSON.

Defaults target the four adapters in this repo (random, exhaustive,
random-small, exhaustive-small) and assume greedy decoding (temperature 0).

Usage examples:

  # Use defaults and GT
  python3 scripts/compare_models.py --gt "aprocsa1944a GT.txt"

  # Explicitly pass model outputs
  python3 scripts/compare_models.py --gt GT.txt \
    --model random=outputs/qwen2p5-14b-random/aprocsa1944a_Azure_response_4k.txt \
    --model exhaustive=outputs/qwen2p5-14b-exhaustive/aprocsa1944a_Azure_response_4k.txt \
    --model random-small="outputs/qwen2p5-14b-random-small/aprocsa1944a Azure_response.txt" \
    --model exhaustive-small="outputs/qwen2p5-14b-exhaustive-small/aprocsa1944a Azure_response.txt"

Outputs:
  - JSON: outputs/compare_models.json (path configurable)
  - CSV : outputs/compare_models.csv (path configurable)
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import jiwer


@dataclass
class ModelEntry:
    tag: str
    path: str
    exists: bool = False
    wer_percent: Optional[float] = None


def compute_wer_percent(gt_text: str, hyp_text: str) -> float:
    """Compute WER (%) with punctuation removal and jiwer standardization."""
    gt_proc = jiwer.RemovePunctuation()(gt_text)
    hy_proc = jiwer.RemovePunctuation()(hyp_text)
    wer_val = jiwer.wer(
        gt_proc,
        hy_proc,
        reference_transform=jiwer.wer_standardize,
        hypothesis_transform=jiwer.wer_standardize,
    )
    return float(wer_val) * 100.0


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_models(args_models: List[str] | None) -> List[ModelEntry]:
    if args_models:
        result: List[ModelEntry] = []
        for spec in args_models:
            if "=" not in spec:
                raise SystemExit(f"--model entries must be in tag=path form: got {spec}")
            tag, path = spec.split("=", 1)
            result.append(ModelEntry(tag=tag.strip(), path=path.strip()))
        return result
    # Defaults for this repo
    return [
        ModelEntry(
            tag="random",
            path=os.path.join("outputs", "qwen2p5-14b-random", "aprocsa1944a_Azure_response_4k.txt"),
        ),
        ModelEntry(
            tag="exhaustive",
            path=os.path.join("outputs", "qwen2p5-14b-exhaustive", "aprocsa1944a_Azure_response_4k.txt"),
        ),
        ModelEntry(
            tag="random-small",
            path=os.path.join(
                "outputs", "qwen2p5-14b-random-small", "aprocsa1944a Azure_response.txt"
            ),
        ),
        ModelEntry(
            tag="exhaustive-small",
            path=os.path.join(
                "outputs", "qwen2p5-14b-exhaustive-small", "aprocsa1944a Azure_response.txt"
            ),
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare WER for multiple model outputs vs ground truth")
    ap.add_argument("--gt", required=True, help="Ground truth transcript path")
    ap.add_argument(
        "--model",
        action="append",
        default=None,
        help="Model spec in tag=path form. Repeatable. If omitted, uses repo defaults.",
    )
    ap.add_argument("--out_json", default=os.path.join("outputs", "compare_models.json"))
    ap.add_argument("--out_csv", default=os.path.join("outputs", "compare_models.csv"))
    ap.add_argument(
        "--decoding_note",
        default="greedy (do_sample=false, temperature=0.0, top_p=1.0)",
        help="Free-form note about decoding settings saved in JSON metadata",
    )
    args = ap.parse_args()

    if not os.path.exists(args.gt):
        raise SystemExit(f"GT not found: {args.gt}")
    gt_text = read_file(args.gt)

    models = parse_models(args.model)

    rows: List[Dict[str, object]] = []
    for m in models:
        m.exists = os.path.exists(m.path)
        if not m.exists:
            rows.append({
                "tag": m.tag,
                "wer_percent": "",
                "exists": False,
                "output_path": m.path,
            })
            continue
        hyp = read_file(m.path)
        m.wer_percent = round(compute_wer_percent(gt_text, hyp), 3)
        rows.append({
            "tag": m.tag,
            "wer_percent": m.wer_percent,
            "exists": True,
            "output_path": m.path,
        })

    summary = {
        "gt_path": os.path.abspath(args.gt),
        "decoding": args.decoding_note,
        "models": [
            {
                "tag": m.tag,
                "output_path": os.path.abspath(m.path),
                "exists": m.exists,
                "wer_percent": m.wer_percent,
            }
            for m in models
        ],
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["tag", "wer_percent", "output_path"])
        for r in rows:
            w.writerow([r["tag"], r["wer_percent"], r["output_path"]])

    # Print concise summary to stdout
    print(f"Wrote JSON: {args.out_json}")
    print(f"Wrote CSV : {args.out_csv}")
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()

