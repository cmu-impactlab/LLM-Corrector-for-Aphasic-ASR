"""
Helpers
-------
pick_exhaustive_phoneme_sentences(truth_path, asr_path,
                                  num_sentences_to_pick=5)

    • Greedy: select up to num_sentences_to_pick ground-truth / ASR
      *sentence* pairs whose union of CMU-ARPAbet phones grows fastest.
    • Only pairs where the strings differ (a real ASR error) are returned.

pick_random_error_sentences(truth_path, asr_path,
                            num_sentences_to_pick=5)

    • Uniform random sample of the same number of error pairs.

Both helpers return
    (asr_sentences, truth_sentences)      # two parallel lists


Dependencies
------------
Run in PowerShell: $Env:PYTHONUTF8 = "1"
pip install wtpsplit jiwer nltk g2p_en spacy
python -m spacy download en_core_web_sm
"""

from pathlib import Path
import re
import random

import nltk
from g2p_en import G2p
import spacy

from wtpsplit import WtP
import jiwer

# ────── resources & setup ──────
nltk.download("cmudict", quiet=True)
CMU_DICT = nltk.corpus.cmudict.dict()
G2P = G2p()

try:
    NLP = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    NLP = spacy.load("en_core_web_sm")

_wtp = WtP("wtp-bert-mini")   # sentence splitter for raw text

MAX_TOKENS = 30
MIN_TOKENS = 3

# ────── helper functions ────── #

def arpabet_phones(word):
    prons = CMU_DICT.get(word.lower(), [G2P(word)])
    phones = set()
    for pron in prons:
        for ph in pron:
            phones.add(re.sub(r"\d", "", ph))
    return phones

def sentence_phone_set(sentence):
    phones = set()
    for w in sentence.split():
        phones |= arpabet_phones(w)
    return phones

def alignment_ops(truth, asr):
    truth_tokens = truth.split()
    asr_tokens   = asr.split()
    out = jiwer.process_words(truth, asr)
    ops = []
    for sent_align in out.alignments:
        for chunk in sent_align:
            t_slice = range(chunk.ref_start_idx, chunk.ref_end_idx)
            h_slice = range(chunk.hyp_start_idx, chunk.hyp_end_idx)
            if chunk.type == "equal":
                for ti, hi in zip(t_slice, h_slice):
                    ops.append((truth_tokens[ti], asr_tokens[hi], "correct"))
            elif chunk.type == "substitute":
                for ti, hi in zip(t_slice, h_slice):
                    ops.append((truth_tokens[ti], asr_tokens[hi], "substitution"))
            elif chunk.type == "delete":
                for ti in t_slice:
                    ops.append((truth_tokens[ti], "", "deletion"))
            elif chunk.type == "insert":
                for hi in h_slice:
                    ops.append(("", asr_tokens[hi], "insertion"))
    return ops

def split_into_sentences(text):
    """
    Use WtP to split a raw stream (no punctuation) into sentences.
    """
    # WtP.split expects a list of documents, returns list of lists
    return list(_wtp.split([text]))[0]

def sentence_pairs_from_alignment(truth, ops):
    """
    Break aligned word ops into sentence-aligned pairs using WtP
    and enforce MIN/MAX token limits per pair.
    Returns list of (asr_sentence, truth_sentence).
    """
    # build the truth stream and get boundaries
    truth_stream = " ".join([w for w, _, _ in ops if w])
    truth_sents  = split_into_sentences(truth_stream)
    counts = [len(s.split()) for s in truth_sents]
    boundaries = set()
    cum = 0
    for c in counts:
        cum += c
        boundaries.add(cum)

    # walk through ops, flushing at boundaries or MAX_TOKENS
    pairs = []
    current_gt, current_asr = [], []
    ref_counter = 0

    for gt_w, asr_w, _ in ops:
        if gt_w:
            ref_counter += 1
            current_gt.append(gt_w)
        if asr_w:
            current_asr.append(asr_w)

        if ref_counter in boundaries or len(current_gt) >= MAX_TOKENS:
            if (MIN_TOKENS <= len(current_gt) <= MAX_TOKENS) or (MIN_TOKENS <= len(current_asr) <= MAX_TOKENS):
                pairs.append((" ".join(current_asr), " ".join(current_gt)))
            current_gt, current_asr = [], []

    # trailing
    if current_gt or current_asr:
        if (MIN_TOKENS <= len(current_gt) <= MAX_TOKENS) or (MIN_TOKENS <= len(current_asr) <= MAX_TOKENS):
            pairs.append((" ".join(current_asr), " ".join(current_gt)))

    return pairs

# ────── main public pickers ────── #
def pick_exhaustive_phoneme_sentences(truth_path, asr_path, num_sentences_to_pick=5):
    truth_text = Path(truth_path).read_text(encoding="utf-8", errors="ignore")
    asr_text   = Path(asr_path)  .read_text(encoding="utf-8", errors="ignore")

    truth = jiwer.RemovePunctuation()(truth_text).replace("\n", " ")
    asr   = jiwer.RemovePunctuation()(asr_text)  .replace("\n", " ")

    ops       = alignment_ops(truth, asr)
    all_pairs = sentence_pairs_from_alignment(truth, ops)
    pairs = [p for p in all_pairs if p[0] != p[1]]
    if not pairs:
        return [], []

    phone_cache = {p: sentence_phone_set(p[1]) for p in pairs}
    selected, covered = [], set()
    while pairs and len(selected) < num_sentences_to_pick:
        best, best_gain = None, set()
        for p in pairs:
            gain = phone_cache[p] - covered
            if len(gain) > len(best_gain):
                best, best_gain = p, gain
        if not best_gain:
            break
        selected.append(best)
        covered |= phone_cache[best]
        pairs.remove(best)

    if len(selected) < num_sentences_to_pick and pairs:
        need = num_sentences_to_pick - len(selected)
        for p in random.sample(pairs, need):
            selected.append(p)

    return [p[0] for p in selected], [p[1] for p in selected]

def pick_random_error_sentences(truth_path, asr_path, num_sentences_to_pick=5):
    truth_text = Path(truth_path).read_text(encoding="utf-8", errors="ignore")
    asr_text   = Path(asr_path)  .read_text(encoding="utf-8", errors="ignore")

    truth = jiwer.RemovePunctuation()(truth_text).replace("\n", " ")
    asr   = jiwer.RemovePunctuation()(asr_text)  .replace("\n", " ")

    ops       = alignment_ops(truth, asr)
    all_pairs = sentence_pairs_from_alignment(truth, ops)
    pairs     = [p for p in all_pairs if p[0] != p[1]]
    if not pairs:
        return [], []

    sampled = random.sample(pairs, min(num_sentences_to_pick, len(pairs)))
    return [p[0] for p in sampled], [p[1] for p in sampled]

# ────── optional CLI smoke test ────── #
if __name__ == "__main__":
    asr_list, gt_list = pick_exhaustive_phoneme_sentences(
        truth_path="aprocsa1554a.txt",
        asr_path="aprocsa1554a Azure.txt",
        num_sentences_to_pick=10,
    )
    print("Ground Truth:", gt_list)
    print("ASR:", asr_list)
