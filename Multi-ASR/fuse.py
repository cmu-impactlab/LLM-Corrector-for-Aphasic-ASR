#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fuse multiple ASR hypotheses into a single transcript, optionally post-processed
by GPT-4.1, DeepSeek R1, or Gemini-2.5-Pro.  Produces evaluation plots of WER
and semantic similarity for every raw engine and post-processing mode.

Usage (from project root with *.txt files):
    $ python fuse.py               # runs everything and saves plots
"""

# ───────── standard lib ─────────
import os, re, glob, sys
from collections import defaultdict
from pathlib import Path
import random, math, itertools
random.seed(42)

# ───────── 3rd-party ────────────

from openai import AzureOpenAI, OpenAI
from google import genai
from google.genai import types
import jiwer
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from wtpsplit import WtP
_wtp = WtP("wtp-bert-mini")
MAX_TOKENS = 30
MIN_TOKENS = 3


#####################################################################
# 0) CONFIGURATION ###################################################
#####################################################################

# 0-a) filename suffix → (canonical key, long label) for *raw* ASR engines
_SUFFIX_MAP = {
    "assemblyai":   ("AssemblyAI",  "AssemblyAI Slam-1"),
    "aws":          ("AWS",         "AWS Transcribe"),
    "azure":        ("Azure",       "Azure Speech"),
    "deepgram":     ("Deepgram",    "Deepgram Nova-3"),
    "elevenlabs":   ("ElevenLabs",  "ElevenLabs Scribe v1"),
    "gcp":          ("GCP",         "GCP Chirp 2"),
    "gemini":       ("Gemini",      "Gemini 2.5 Pro Audio"),
    "gladia":       ("Gladia",      "Gladia Solaria"),
    "speechmatics": ("Speechmatics","Speechmatics Ursa 2"),
    "whisper":      ("Whisper",     "Whisper v3"),
}

_POSTPROCS = dict()
# 0-b) LLM post-processing modes we support
# key → dict(label, filename_suffix, fuse_func)
_POSTPROCS = {
    "GPT-4.1 Post": dict(
        label="GPT-4.1 Fusion",
        filename_suffix="GPT-4.1",
        fuse_func="gpt_fuse",
    ),
    "DeepSeek Post": dict(
        label="DeepSeek R1 Fusion",
        filename_suffix="DeepSeek",
        fuse_func="deepseek_fuse",
    ),
    "Gemini Post": dict(
        label="Gemini 2.5 Pro Fusion",
        filename_suffix="Gemini POST",
        fuse_func="gemini_fuse",
    ),
}

_POSTPROC_REGEX = dict()
# 0-c) regexes to detect post-processed filenames
_POSTPROC_REGEX = {
    "GPT-4.1 Post": re.compile(r"[ _-]GPT[ _-]?4\.?1$", re.I),
    "DeepSeek Post": re.compile(r"[ _-]DeepSeek$", re.I),
    "Gemini Post":   re.compile(r"[ _-]Gemini[ _-]?Post$", re.I),
}

# 0-d) Instantiate reusable models lazily later
_ST_MODEL = None  # Sentence-Transformer for semantic similarity

# 0-e)  Map every internal key → nice display label  #################
_LABEL_MAP = {canon: long for canon, long in _SUFFIX_MAP.values()}
_LABEL_MAP.update({k: v["label"] for k, v in _POSTPROCS.items()})


#####################################################################
# 1) LOAD TRANSCRIPTS ################################################
#####################################################################

def collect_transcripts(folder="."):
    """Return dict(sample → {service: transcript})"""
    root = Path(folder)
    grouped = defaultdict(dict)

    # 1-a) Post-processed variants from evaluation_outputs/
    for path in (root / "evaluation_outputs").glob("*.txt"):
        stem = path.stem
        lowered = stem.lower()
        txt = path.read_text(encoding="utf-8").strip()
        for key, regex in _POSTPROC_REGEX.items():
            if regex.search(lowered):
                sample = regex.sub("", stem).rstrip(" _-")
                grouped[sample][key] = txt
                break

    # 1-b) Raw engine transcripts from raw_transcripts/
    for path in (root / "raw_transcripts").glob("*.txt"):
        stem = path.stem
        lowered = stem.lower()
        txt = path.read_text(encoding="utf-8").strip()
        for suff, (canon, _) in _SUFFIX_MAP.items():
            for sep in (" ", "_", "-"):
                if lowered.endswith(sep + suff):
                    sample = stem[: -len(sep + suff)].rstrip(" _-")
                    grouped[sample][canon] = txt
                    break
            else:
                continue
            break

    # 1-c) Ground truth from ground_truth/
    for path in (root / "ground_truth").glob("*.txt"):
        stem = path.stem
        txt = path.read_text(encoding="utf-8").strip()
        grouped[stem]["Ground"] = jiwer.RemovePunctuation()(txt)

    return grouped


#####################################################################
# 2) QUALITY METRICS ##################################################
#####################################################################

def split_into_sentences(text):
    """
    Use WtP to split a raw stream (no punctuation) into sentences.
    """
    return list(_wtp.split([text]))[0]

def alignment_ops(truth, hyp):
    """
    Perform word-level alignment of truth vs. ASR using jiwer.process_words.
    Returns a list of (truth_word, hyp_word, op_type) tuples.
    """
    truth_tokens = truth.split()
    hyp_tokens   = hyp.split()
    out = jiwer.process_words(truth, hyp)
    ops = []
    for sent_align in out.alignments:
        for chunk in sent_align:
            t_slice = range(chunk.ref_start_idx, chunk.ref_end_idx)
            h_slice = range(chunk.hyp_start_idx, chunk.hyp_end_idx)
            if chunk.type == "equal":
                for ti, hi in zip(t_slice, h_slice):
                    ops.append((truth_tokens[ti], hyp_tokens[hi], "correct"))
            elif chunk.type == "substitute":
                for ti, hi in zip(t_slice, h_slice):
                    ops.append((truth_tokens[ti], hyp_tokens[hi], "substitution"))
            elif chunk.type == "delete":
                for ti in t_slice:
                    ops.append((truth_tokens[ti], "", "deletion"))
            elif chunk.type == "insert":
                for hi in h_slice:
                    ops.append(("", hyp_tokens[hi], "insertion"))
    return ops

def sentence_pairs_from_alignment(truth, ops):
    """
    Break the aligned word ops into sentence-aligned pairs, using WtP
    to detect sentence boundaries in the reconstructed truth stream,
    and flushing pairs when hitting those boundaries (or MAX_TOKENS).
    Returns a list of (hyp_sentence, truth_sentence) tuples.
    """
    # rebuild the truth stream & detect sentence boundaries
    truth_stream = " ".join(w for w, _, _ in ops)
    truth_sents  = split_into_sentences(truth_stream)
    counts = [len(s.split()) for s in truth_sents]
    boundaries = set()
    cum = 0
    for c in counts:
        cum += c
        boundaries.add(cum)

    # walk through ops, flushing when we hit a boundary or token limit
    pairs = []
    current_gt, current_hyp = [], []
    ref_counter = 0

    for gt_w, hyp_w, _ in ops:
        if gt_w:
            ref_counter += 1
            current_gt.append(gt_w)
        if hyp_w:
            current_hyp.append(hyp_w)

        if ref_counter in boundaries or len(current_gt) >= MAX_TOKENS:
            if (MIN_TOKENS <= len(current_gt) <= MAX_TOKENS) or \
               (MIN_TOKENS <= len(current_hyp) <= MAX_TOKENS):
                pairs.append((" ".join(current_hyp), " ".join(current_gt)))
            current_gt, current_hyp = [], []

    # handle any trailing words
    if current_gt or current_hyp:
        if (MIN_TOKENS <= len(current_gt) <= MAX_TOKENS) or \
           (MIN_TOKENS <= len(current_hyp) <= MAX_TOKENS):
            pairs.append((" ".join(current_hyp), " ".join(current_gt)))

    return pairs

def semantic_sim(reference, hypothesis, model):
    """
    1. Normalize & remove punctuation so jiwer sees clean streams.
    2. Align at the word level (jiwer → alignment_ops).
    3. Break that alignment into WtP sentence pairs via sentence_pairs_from_alignment.
    4. SBERT-encode each aligned (truth, hyp) sentence pair.
    5. Cosine each, then average into a final percentage.
    """
    ref_clean = jiwer.RemovePunctuation()(reference.replace("\n", " "))
    hyp_clean = jiwer.RemovePunctuation()(hypothesis.replace("\n", " "))

    ops = alignment_ops(ref_clean, hyp_clean)
    pairs = sentence_pairs_from_alignment(ref_clean, ops)

    sims = []
    for truth_sent, hyp_sent in pairs:
        if not truth_sent or not hyp_sent:
            continue
        emb_truth = model.encode(truth_sent, convert_to_tensor=True)
        emb_hyp   = model.encode(hyp_sent,   convert_to_tensor=True)
        sims.append(float(util.pytorch_cos_sim(emb_truth, emb_hyp)[0][0]))

    return round(np.mean(sims) * 100, 2) if sims else 0.0


#####################################################################
# 3) PLOTTING #########################################################
#####################################################################
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines

def plot_metric(df, metric, fname, title, ymax, ymin=0, out_dir="evaluation_outputs"):
    # 1) Which display‐names are post‐processed?
    post_labels = [
        _POSTPROCS[k]["label"]
        for k in ["Gemini Post", # comment this out if needed
                  "DeepSeek Post", # comment this out if needed
                  "GPT-4.1 Post"]
    ]
    # 2) Everything else is raw
    raw_labels = [s for s in df["Service"].unique() if s not in post_labels]
    hue_order  = raw_labels + post_labels

    # 3) Split the six samples into two groups of three
    samples     = list(df["Sample"].unique())
    top_samples = samples[:3]
    bot_samples = samples[3:]

    # Use seaborn for nice grid and structure, but override fonts
    sns.set_theme(style="whitegrid", font_scale=1.6)
    
    # Override seaborn's font settings with Times New Roman (IEEE standard)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 22,
        'mathtext.fontset': 'stix',  # STIX fonts are Times-compatible for math
        'axes.unicode_minus': False,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'xtick.labelsize': 22,
        'ytick.labelsize': 22
    })
    
    # Professional monochromatic color scheme with transparency
    # Lighter blue tones for raw ASR services (all in blue family)
    blue_tones = [
        '#3182bd',  # Standard blue
        '#6baed6',  # Medium-light blue
        '#9ecae1',  # Light blue
        '#4292c6',  # Medium blue
        '#85c1e5',  # Baby blue
        '#2b8cbe',  # Sky blue
        '#74a9cf',  # Soft sky blue
        '#5eb3d6',  # Light cerulean
        '#91bfdb',  # Powder blue
        '#54a3d5',  # Bright blue
    ]
    
    # Yellow/gold tones for post-processed
    yellow_tones = [
        '#fed976',  # Light gold
        '#feb24c',  # Medium gold
        '#fd8d3c',  # Dark gold
    ]
    
    # Use the blue tones without transparency for raw ASR
    raw_colors = [blue_tones[i % len(blue_tones)] for i in range(len(raw_labels))]
    
    # Use yellow/gold tones for post-processed
    post_colors = yellow_tones[:len(post_labels)]
    
    # Combine all colors
    palette = raw_colors + post_colors

    # 4) Two stacked subplots sharing Y
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharey=True, figsize=(2 * len(raw_labels), 12))

    def _draw_panel(ax, subset):
        df_sub = df[df["Sample"].isin(subset)]
        # draw bars (no auto-legend)
        sns.barplot(
            data=df_sub, x="Sample", y=metric,
            hue="Service", hue_order=hue_order,
            ax=ax, width= 0.8,
            palette=palette, edgecolor=".25", dodge=True,
            legend=False
        )
        # 1) label every bar
        for cont in ax.containers:
            ax.bar_label(cont, fmt="%.0f", padding=40/len(hue_order), weight="bold", fontfamily="Times New Roman", fontsize=20)
        # 2) draw raw‐ASR mean lines only under raw bars
        mean_by_sample = (
            df_sub[df_sub["Service"].isin(raw_labels)]
                 .groupby("Sample")[metric]
                 .mean()
        )
        raw_containers = ax.containers[: len(raw_labels)]
        for i, sample in enumerate(subset):
            y = mean_by_sample.loc[sample]
            bars = [c.patches[i] for c in raw_containers]
            x0   = min(b.get_x() for b in bars)
            x1   = max(b.get_x() + b.get_width() for b in bars)
            ax.hlines(y, x0, x1, color="red", linestyle="--", linewidth=2, label="_nolegend_")
            ax.text(
                x0 - bars[0].get_width()*0.3, y,
                f"{y:.0f}",
                ha="right", va="center",
                weight="bold", color="red", fontfamily="Times New Roman", fontsize=20
            )
        # cosmetics
        ax.set_xticklabels(subset, fontsize=22, fontfamily="Times New Roman")
        ax.tick_params(axis='x', pad=12)  # Adjust padding between labels and axis
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("")

    # 5) draw both panels
    _draw_panel(ax_top, top_samples)
    _draw_panel(ax_bot, bot_samples)

    # 6) y‐labels and title
    ax_top.set_ylabel(metric, fontfamily="Times New Roman", fontsize=24)
    ax_bot.set_ylabel(metric, fontfamily="Times New Roman", fontsize=24)
    fig.suptitle(title, y=0.93, fontsize=30, fontfamily="Times New Roman")

    # 7) build a manual legend (services + single Raw-ASR Mean)
    # a) one Patch per service in display order
    raw_handles = [
        mpatches.Patch(facecolor=palette[i], edgecolor="black", linewidth=0.5, label=s)
        for i, s in enumerate(raw_labels)
    ]
    post_handles = [
        mpatches.Patch(facecolor=palette[i+len(raw_labels)], edgecolor="black", linewidth=0.5, label=s)
        for i, s in enumerate(post_labels)
    ]
    # b) one Line2D for the dashed‐mean
    mean_handle = mlines.Line2D([], [], color="red", linestyle="--", linewidth=2,
                                label="Raw-ASR Mean")
    all_handles = raw_handles + [mean_handle] + post_handles
    all_labels  = [h.get_label()   for h in all_handles]

    # c) place that legend below the bottom panel
    fig.legend(
        all_handles, all_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.013 * len(raw_labels)),
        bbox_transform=fig.transFigure,
        ncol=round(0.4 * len(raw_labels)), frameon=False,
        prop={'family': 'Times New Roman', 'size': 22}
    )

    # 8) finalize
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    # Save as SVG for crisp vector graphics
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    svg_fname = out_path / fname.replace('.png', '.svg')
    plt.savefig(svg_fname, format='svg', bbox_inches="tight")
    plt.close()
    print("✓", svg_fname)


def plot_metric_concatenated(values, y_label, fname, title, ymax, ymin=0, out_dir="evaluation_outputs"):
    df = (pd.Series(values, name="Value")
            .reset_index()
            .rename(columns={"index": "Service"}))
    df["Bucket"] = "All"

    post_labels = ["Gemini Post", # comment this out if needed
                   "DeepSeek Post", # comment this out if needed
                   "GPT-4.1 Post"]
    raw_labels  = [s for s in df["Service"] if s not in post_labels]
    hue_order   = raw_labels + post_labels

    # Use seaborn for nice grid and structure, but override fonts
    sns.set_theme(style="whitegrid", font_scale=1.6)

    # Override seaborn's font settings with Times New Roman (IEEE standard)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 22,
        'mathtext.fontset': 'stix',  # STIX fonts are Times-compatible for math
        'axes.unicode_minus': False,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'xtick.labelsize': 22,
        'ytick.labelsize': 22
    })

    grey_shades = [
    '#737373',
    '#A6A6A6',
    '#595959',
    '#BFBFBF',
    '#404040',
    '#8C8C8C',
    '#262626',
    '#D9D9D9',
    '#666666',
    '#999999'
    ]

    # Yellow/gold tones for post-processed
    yellow_tones = [
        '#fed976',  # Light gold
        '#feb24c',  # Medium gold
        '#fd8d3c',  # Dark gold
    ]

    # Use the blue tones without transparency for raw ASR
    raw_colors = [grey_shades[i % len(grey_shades)] for i in range(len(raw_labels))]

    # Use yellow/gold tones for post-processed
    post_colors = yellow_tones[:len(post_labels)]

    # Combine all colors
    palette = raw_colors + post_colors

    fig, ax = plt.subplots(figsize=(len(hue_order), 6.5))

    sns.barplot(data=df, x="Bucket", y="Value", hue="Service", hue_order=hue_order,
                ax=ax, palette=palette, width=.8, edgecolor=".25")

    raw_mean = df[df["Service"].isin(raw_labels)]["Value"].mean()
    patches  = ax.patches[:len(raw_labels)]
    x_left   = min(p.get_x() for p in patches)
    x_right  = max(p.get_x() + p.get_width() for p in patches)
    ax.hlines(raw_mean, x_left, x_right,
            color="red", linestyle="--", linewidth=2, label="Raw-ASR Mean")
    ax.text(x_left - patches[0].get_width()*0.1, raw_mean, f"{raw_mean:.0f}",
            ha="right", va="center", weight="bold", color="red", fontfamily="Times New Roman", fontsize=20)

    ax.set_ylabel(y_label, fontfamily="Times New Roman", fontsize=24)
    ax.set_xlabel("")
    ax.set_title(title, pad=40, fontfamily="Times New Roman", fontsize=30)  # Increased padding
    ax.set_ylim(ymin, ymax)
    ax.set_xticks([])
    ax.margins(x=0.2)
    fmt = f"%.0f"
    for c in ax.containers:
        ax.bar_label(c, fmt=fmt, padding= 40 / len(hue_order), weight="bold", fontfamily="Times New Roman", fontsize=20)

    # Build legend handles like in plot_metric
    raw_handles = [mpatches.Patch(facecolor=palette[i], edgecolor="black", linewidth=0.5, label=s)
                   for i, s in enumerate(raw_labels)]
    post_handles = [mpatches.Patch(facecolor=palette[i+len(raw_labels)], edgecolor="black", linewidth=0.5, label=s)
                    for i, s in enumerate(post_labels)]
    mean_handle = mlines.Line2D([], [], color="red", linestyle="--", linewidth=2, label="Raw-ASR Mean")

    all_handles = raw_handles + [mean_handle] + post_handles
    all_labels = [_LABEL_MAP.get(h.get_label(), h.get_label()) for h in all_handles]

    ax.legend(all_handles, all_labels,
              loc="upper center", bbox_to_anchor=(0.5, -0.1),
              ncol=3, frameon=False, prop={'family': 'Times New Roman', 'size': 22})
    plt.tight_layout()
    # Save as SVG for crisp vector graphics
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    svg_fname = out_path / fname.replace('.png', '.svg')
    plt.savefig(svg_fname, format='svg', bbox_inches="tight")
    plt.close()
    print("✓", svg_fname)


#####################################################################
# 4) SYSTEM PROMPT & HELPERS #########################################
#####################################################################

_SYS_MSG = """# task 
You will receive **multiple independent ASR hypotheses** of the same aphasic utterance. 
Fuse them into **one transcript** that minimises word-error-rate (WER) against the hidden gold reference. 
# decision rules
0. **Disordered speech** – never rewrite or “fix” ungrammatical patient language; keep it exactly as spoken.
1. **Keep word order** – never paraphrase, reorder, or add new words. 
2. **Fillers** – delete pure hesitation noises: 
*uh, um, er, ah, hmm, mm, uh-huh, uh-uh*. 
3. **Discourse markers** – *you know | yeah | okay / ok | well*: 
• keep **one copy** only if the marker appears in **close to half** of the ASR hypotheses at roughly the same position; 
• otherwise drop it. 
4. **Self-corrections** – *oh gosh | my god | i mean | sorry | i am sorry*: 
• always keep exactly one copy when any of these self-correction cues occur, regardless of how many hypotheses contain them.
5. **Repetitions** 
• if **more than half** of the hypotheses show the same word repeated back-to-back **at least 3** times, keep it **exactly twice**; 
• otherwise collapse adjacent duplicates to a single token. 
6. **Stutters** – remove truncated fragments before a hyphen 
(e.g. *thin-think* → *think*). 
7. **Non-lexical noises** – remove tokens such as *whew, sigh, laugh, cough*. 
8. **Digits → words** – spell every numeral out in plain English 
(13 → *thirteen*, 2008 → *two thousand eight*, 07 → *o seven*). 
9. **Safe substitutions only** – replace a word when **close to half** of the hypotheses agree on the same candidate; never invent unseen words. 
10. **Ties** – when choices remain, prefer the majority wording; if still tied, choose the shorter option.
11. **Colloquial Contractions** – expand all colloquial contractions such as “useta,” “hafta,” “hadta,” “gotta,” and “gonna” into their standard forms “used to,” “have to,” “had to,” and “going to,” etc.
# deliverable 
Return **only** the fused transcript text (single line, normalised spacing)."""

_SYS_MSG_SINGLE = """# task 
You will receive one ASR hypothese of an aphasic utterance. 
Postprocess it into **one transcript** that minimises word-error-rate (WER) against the hidden gold reference. 
# decision rules
0. **Disordered speech** – never rewrite or “fix” ungrammatical patient language; keep it exactly as spoken.
1. **Keep word order** – never paraphrase, reorder, or add new words. 
2. **Fillers** – delete pure hesitation noises: 
*uh, um, er, ah, hmm, mm, uh-huh, uh-uh*. 
3. **Self-corrections** – *oh gosh | my god | i mean | sorry | i am sorry*: 
• always keep exactly one copy when any of these self-correction cues occur
4. **Repetitions** 
• if the same word is repeated back-to-back **at least 3** times, keep it **exactly twice**; 
• otherwise collapse adjacent duplicates to a single token. 
5. **Stutters** – remove truncated fragments before a hyphen 
(e.g. *thin-think* → *think*). 
6. **Non-lexical noises** – remove tokens such as *whew, sigh, laugh, cough*. 
7. **Digits → words** – spell every numeral out in plain English 
(13 → *thirteen*, 2008 → *two thousand eight*, 07 → *o seven*). 
8. **Colloquial Contractions** – expand all colloquial contractions such as “useta,” “hafta,” “hadta,” “gotta,” and “gonna” into their standard forms “used to,” “have to,” “had to,” and “going to,” etc.
# deliverable 
Return **only** the fused transcript text (single line, normalised spacing)."""


def _build_user_msg(hyps_dict):
    ordered = [hyps_dict[k] for k in sorted(hyps_dict.keys())]
    n = len(ordered)
    return (
        f"Here are the {n} hypotheses, numbered 1–{n}:\n\n" +
        "\n".join(f"{i+1}. {txt}" for i, txt in enumerate(ordered)) +
        "\n\nReturn just the fused transcript."
    )


#####################################################################
# 5) LLM FUSION FUNCTIONS ############################################
#####################################################################

def gpt_fuse(hyps_dict):
    sys_msg = _SYS_MSG_SINGLE if len(hyps_dict) == 1 else _SYS_MSG
    user_msg = _build_user_msg(hyps_dict)
    client = AzureOpenAI(
        api_key="<YOUR AZURE OPENAI API KEY>",
        api_version="2024-10-21",
        azure_endpoint = "<YOUR AZURE OPENAI ENDPOINT>"
    )
    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system", "content": sys_msg},
                  {"role": "user",   "content": user_msg}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def deepseek_fuse(hyps_dict):
    """DeepSeek R1 (OpenAI-compatible).  Requires DEEPSEEK_API_KEY."""
    user_msg = _build_user_msg(hyps_dict)
    client = OpenAI(
        api_key="<YOUR DEEPSEEK API KEY>",
        base_url="https://api.deepseek.com",
    )
    resp = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role": "system", "content": _SYS_MSG},
                  {"role": "user",   "content": user_msg}],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def gemini_fuse(hyps_dict):
    """Gemini 2.5 Pro.  Requires GEMINI_API_KEY and google-generativeai≥0.4.0"""
    user_msg = _build_user_msg(hyps_dict)
    client = genai.Client(api_key="<YOUR GEMINI API KEY>")
    resp = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=user_msg,
        config=types.GenerateContentConfig(
        system_instruction=_SYS_MSG,
        temperature=0,
        ),
    )
    return resp.text.strip()


#####################################################################
# 6) PIPELINE UTILITIES ###############################################
#####################################################################

def _ensure_st_model():
    global _ST_MODEL
    if _ST_MODEL is None:
        print("⏳ Loading SentenceTransformer…", file=sys.stderr)
        _ST_MODEL = SentenceTransformer("all-mpnet-base-v2")
    return _ST_MODEL


def _save_fused(sample, cfg, text, folder="."):
    out_dir = Path(folder) / "evaluation_outputs"
    out_dir.mkdir(exist_ok=True)
    fn = out_dir / f"{sample} {cfg['filename_suffix']}.txt"
    fn.write_text(text.strip() + "\n", encoding="utf-8")
    print("✓ wrote", fn)


def run_postprocessing(grouped):
    for sample, data in grouped.items():
        # Collect the 10 raw hypotheses (keys present in _SUFFIX_MAP canon names)
        hyps_dict = {svc: txt for svc, txt in data.items() if svc in
                     {v[0] for v in _SUFFIX_MAP.values()}}
        if len(hyps_dict) != len(_SUFFIX_MAP):
            continue  # skip incomplete sample

        for key, cfg in _POSTPROCS.items():
            if key in data:
                continue  # already present
            fused_txt = globals()[cfg["fuse_func"]](hyps_dict)
            data[key] = fused_txt
            _save_fused(sample, cfg, fused_txt)


def build_metrics_df(grouped):
    rows = []
    st_model = _ensure_st_model()

    for sample, bundle in grouped.items():
        gt = bundle.get("Ground")
        if not gt:
            continue  # skip if gold reference missing

        for service, txt in bundle.items():
            if service == "Ground":
                continue
            hyp_clean = jiwer.RemovePunctuation()(txt)
            wer_val = round(
                jiwer.wer(
                    gt,
                    hyp_clean,
                    reference_transform=jiwer.wer_standardize,
                    hypothesis_transform=jiwer.wer_standardize,
                )
                * 100,
                2,
            )
            sim_val = semantic_sim(gt, txt, st_model)
            rows.append(
                dict(
                    Sample=sample,
                    Service=_LABEL_MAP.get(service, service),
                    WER=wer_val,
                    SemanticSimilarity=sim_val,
                )
            )

    return pd.DataFrame(rows)


def aggregated_sem_sims(grouped):
    """
    Return {service → SS%} computed over the 6-sample concatenation once.
    """
    gt_all   = " ".join(b["Ground"] for b in grouped.values())
    services = [k for k in next(iter(grouped.values())) if k != "Ground"]
    out = {}
    for svc in services:
        hyp_all  = " ".join(b[svc] for b in grouped.values())
        out[svc] = semantic_sim(gt_all, hyp_all, _ensure_st_model())
    return out


def aggregated_wers(grouped):
    """
    Return {service → WER%} computed over the 6-sample concatenation once.
    """
    gt_all = " ".join(b["Ground"] for b in grouped.values())
    out = {}
    example_bundle = next(iter(grouped.values()))
    services = [k for k in example_bundle if k not in ("Ground")]
    for svc in services:
        hyp_all = " ".join(b[svc] for b in grouped.values())
        out[svc] = jiwer.wer(gt_all,
                             jiwer.RemovePunctuation()(hyp_all),
                             reference_transform=jiwer.wer_standardize,
                             hypothesis_transform=jiwer.wer_standardize,
                             ) * 100
    return out


#####################################################################
# 7)  VARIABLE-ASR  EXPERIMENT  #####################################
#####################################################################

def random_combos(population, m, n):
    """
    Yield up to n unique combos of size m.
    If n exceeds C(len(population), m) we just return them all, shuffled.
    """
    combos = list(itertools.combinations(sorted(population), m))
    random.shuffle(combos)
    for combo in combos[: min(n, len(combos))]:
        yield combo


def fuse_combo(grouped, combo, tag, out_dir, reuse=True):
    """
    • For every sample (utterance) in `grouped`
        – build a hyps_dict using only the ASRs in `combo`
        – fuse with GPT-4.1 (or read per-sample cache)
    • After 6 fusions, concatenate the six fused strings
      and compute ONE WER value (same transform pipeline
      as everywhere else).
    • Return that aggregated WER %.
    """
    fused_samples = []
    for sample_name, bundle in grouped.items():
        per_sample_cache = out_dir / f"{sample_name} GPT-4.1 {tag}.txt"
        if reuse and per_sample_cache.exists():
            fused = per_sample_cache.read_text().strip()
        else:
            hyps_dict = {svc: bundle[svc] for svc in combo}
            fused = gpt_fuse(hyps_dict)
            per_sample_cache.write_text(fused + "\n", encoding="utf-8")
        fused_samples.append(fused)

    # ── aggregate over the 6 samples ─────────────────────────────────────
    gt_all    = " ".join(b["Ground"] for b in grouped.values())
    fused_all = " ".join(fused_samples)
    return jiwer.wer(gt_all,
                     jiwer.RemovePunctuation()(fused_all),
                     reference_transform=jiwer.wer_standardize,
                     hypothesis_transform=jiwer.wer_standardize,
                     ) * 100


def plot_stage(df, upto_m, out_dir):
    # Use seaborn for nice grid and structure, but override fonts
    sns.set_theme(style="whitegrid", font_scale=1.6)
    
    # Override seaborn's font settings with Times New Roman (IEEE standard)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 22,
        'mathtext.fontset': 'stix',  # STIX fonts are Times-compatible for math
        'axes.unicode_minus': False,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'xtick.labelsize': 22,
        'ytick.labelsize': 22
    })
    
    # Smaller figure size with better proportions
    plt.figure(figsize=(6, 5))
    
    # Blue color scheme matching the main plots
    box_color = '#3182bd'  # Standard blue from the main plots
    
    ax = sns.boxplot(data=df, x="m", y="rel_impr",
                     order=sorted(df["m"].unique()),
                     showfliers=False,
                     color=box_color,
                     boxprops=dict(edgecolor='#333333', linewidth=2.0),
                     whiskerprops=dict(color='#333333', linewidth=2.0),
                     capprops=dict(color='#333333', linewidth=2.0),
                     medianprops=dict(color='#333333', linewidth=2.5))
    
    ax.set_xlabel("Num of fused ASRs", labelpad=10, fontfamily="Times New Roman", fontsize=24)
    ax.set_ylabel("WER % Improvement", labelpad=10, fontfamily="Times New Roman", fontsize=24)
    
    # Set y-axis to show 0 and max clearly
    y_min, y_max = df["rel_impr"].min(), df["rel_impr"].max()
    y_max_rounded = np.ceil(y_max / 10) * 10  # Round up to nearest 10
    ax.set_ylim(0, y_max_rounded)
    ax.set_yticks(np.arange(0, y_max_rounded + 1, 10))  # Ticks every 10 units
    
    # Set tick labels with Times New Roman
    ax.tick_params(axis='both', labelsize=22)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
    
    plt.tight_layout()
    
    # Save as SVG for crisp vector graphics
    svg_out = out_dir / f"wer_upto_{upto_m}.svg"
    plt.savefig(svg_out, format='svg', bbox_inches="tight")
    plt.close()
    print(f"✓ {svg_out}")


def run_experiment(grouped, m_list=(1,2,3,5,7), n_runs=100):
    base = aggregated_wers(grouped)          # {svc: WER%}
    services = list(base.keys())
    print(f"📄 Running Variable-ASR Experiment (with GPT-4.1) ...")

    reuse = input("Reuse existing random experiment data? (y/N) ").lower().startswith("y")
    
    out_dir = Path(f"experiment_outputs (RAND {n_runs})")
    out_dir.mkdir(exist_ok=True)

    rows = []
    for m in m_list:
        for idx, combo in enumerate(random_combos(services, m, n_runs), 1):
            tag   = f"{m}-{idx}"
            b     = fuse_combo(grouped, combo, tag, out_dir, reuse)
            a     = sum(base[s] for s in combo) / m
            rel   = (a - b) / a * 100
            rows.append({"m": m, "run": idx, "avg_base": a, "rel_impr": rel})
        plot_stage(pd.DataFrame(rows), upto_m=m, out_dir=out_dir)
    df_all = pd.DataFrame(rows)
    plot_scatter(df_all, out_dir)


def plot_scatter(df, out_dir):
    """
    Scatter‐plot of avg_base vs. rel_impr with a best‐fit line.
    """
    # Use seaborn for nice grid and structure, but override fonts
    sns.set_theme(style="whitegrid", font_scale=1.6)
    
    # Override seaborn's font settings with Times New Roman (IEEE standard)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 22,
        'mathtext.fontset': 'stix',  # STIX fonts are Times-compatible for math
        'axes.unicode_minus': False,
        'axes.linewidth': 1.0,
        'grid.linewidth': 0.8,
        'axes.edgecolor': '#333333',
        'xtick.labelsize': 22,
        'ytick.labelsize': 22
    })
    
    # Smaller figure size with better proportions
    plt.figure(figsize=(6, 5))
    
    ax = sns.regplot(
        x="avg_base",
        y="rel_impr",
        data=df,
        scatter_kws={"s": 30,
                     "alpha": 0.7,
                     "color": "#005c94",
                     "linewidths": 0},
        line_kws={"lw": 3,
                  "color": "#ff9900"},
        ci=None
    )
    
    ax.set_xlabel("Initial ASR Set WER Average (%)", labelpad=10, fontfamily="Times New Roman", fontsize=24)
    ax.set_ylabel("WER % Decrease", labelpad=10, fontfamily="Times New Roman", fontsize=24)
    
    # Set x-axis to show 0 and max clearly
    x_min, x_max = df["avg_base"].min(), df["avg_base"].max()
    x_max_rounded = np.ceil(x_max / 10) * 10  # Round up to nearest 10
    ax.set_xlim(17.5, 38)
    ax.set_xticks(np.arange(20, 36, 5))  # Ticks every 10 units
    
    # Set y-axis to show 0 and max clearly
    y_min, y_max = df["rel_impr"].min(), df["rel_impr"].max()
    y_max_rounded = np.ceil(y_max / 10) * 10  # Round up to nearest 10
    ax.set_ylim(0, y_max_rounded)
    ax.set_yticks(np.arange(0, y_max_rounded + 1, 10))  # Ticks every 10 units

    # Set tick labels with Times New Roman
    ax.tick_params(axis='both', labelsize=22)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontfamily('Times New Roman')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linewidth=0.8)
    
    plt.tight_layout()
    
    # Save as SVG for crisp vector graphics
    svg_out = out_dir / "scatter_with_fit.svg"
    plt.savefig(svg_out, format='svg', bbox_inches="tight")
    plt.close()
    print(f"✓ {svg_out}")


def main():
    grouped = collect_transcripts()
    print(f"📄 Found {len(grouped)} sample bundles")
    run_postprocessing(grouped)

    df = build_metrics_df(grouped)
    df.to_csv("metrics.csv", index=False)
    print("✓ metrics.csv")

    plot_metric(
        df,
        "WER",
        "wer.svg",
        "Word-Error-Rate (lower is better)",
        ymax=60,
    )
    plot_metric(
        df,
        "SemanticSimilarity",
        "semantic_sim.png",
        "Semantic Similarity (higher is better)",
        ymax=100,
        ymin=60,
    )
    wer_vals = aggregated_wers(grouped)
    plot_metric_concatenated(
        wer_vals,
        y_label = "WER (%)",
        fname   = "wer_concat.png",
        title   = "Word-Error-Rate – concatenated across six samples (lower is better)",
        ymax    = 45,
    )
    sim_vals = aggregated_sem_sims(grouped)
    plot_metric_concatenated(
        sim_vals,
        y_label = "SS (%)",
        fname   = "sem_sim_concat.png",
        title   = "Semantic Similarity – concatenated across six samples (higher is better)",
        ymax    = 100,
        ymin    = 60,
    )
    print(f"🎉 {len(grouped)}-ASR Fusion basic evaluation done.")
    
    filtered_grouped = {
    sample: {
        k: v for k, v in bundle.items()
        if k == "Ground" or k not in _POSTPROCS
        }
        for sample, bundle in grouped.items()
    }
    run_experiment(filtered_grouped)
    print("🎉 Variable-ASR random experiment done.")


if __name__ == "__main__":
    main()