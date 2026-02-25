from pathlib import Path
from collections import Counter
import random
import re

import jiwer
import nltk
from g2p_en import G2p
from wtpsplit import WtP


nltk.download("cmudict", quiet=True)
CMU_DICT = nltk.corpus.cmudict.dict()
G2P = G2p()
_wtp = WtP("wtp-bert-mini")

MAX_TOKENS = 30
MIN_TOKENS = 3


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
    asr_tokens = asr.split()
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
    return list(_wtp.split([text]))[0]


def sentence_pairs_from_alignment(truth, ops):
    truth_stream = " ".join([w for w, _, _ in ops if w])
    truth_sents = split_into_sentences(truth_stream)
    counts = [len(s.split()) for s in truth_sents]
    boundaries = set()
    cum = 0
    for c in counts:
        cum += c
        boundaries.add(cum)

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
            if (MIN_TOKENS <= len(current_gt) <= MAX_TOKENS) or (
                MIN_TOKENS <= len(current_asr) <= MAX_TOKENS
            ):
                pairs.append((" ".join(current_asr), " ".join(current_gt)))
            current_gt, current_asr = [], []

    if current_gt or current_asr:
        if (MIN_TOKENS <= len(current_gt) <= MAX_TOKENS) or (
            MIN_TOKENS <= len(current_asr) <= MAX_TOKENS
        ):
            pairs.append((" ".join(current_asr), " ".join(current_gt)))

    return pairs


def pick_exhaustive_phoneme_sentences(truth_path, asr_path, num_sentences_to_pick=5):
    truth_text = Path(truth_path).read_text(encoding="utf-8", errors="ignore")
    asr_text = Path(asr_path).read_text(encoding="utf-8", errors="ignore")

    truth = jiwer.RemovePunctuation()(truth_text).replace("\n", " ")
    asr = jiwer.RemovePunctuation()(asr_text).replace("\n", " ")

    ops = alignment_ops(truth, asr)
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
        available = min(need, len(pairs))
        for p in random.sample(pairs, available):
            selected.append(p)

    return [p[0] for p in selected], [p[1] for p in selected]


def pick_random_error_sentences(truth_path, asr_path, num_sentences_to_pick=5):
    truth_text = Path(truth_path).read_text(encoding="utf-8", errors="ignore")
    asr_text = Path(asr_path).read_text(encoding="utf-8", errors="ignore")

    truth = jiwer.RemovePunctuation()(truth_text).replace("\n", " ")
    asr = jiwer.RemovePunctuation()(asr_text).replace("\n", " ")

    ops = alignment_ops(truth, asr)
    all_pairs = sentence_pairs_from_alignment(truth, ops)
    pairs = [p for p in all_pairs if p[0] != p[1]]
    if not pairs:
        return [], []

    sampled = random.sample(pairs, min(num_sentences_to_pick, len(pairs)))
    return [p[0] for p in sampled], [p[1] for p in sampled]


def discover_error_patterns(ops, asr_text, gt_text):
    inserted_words = [word for _, word, op_type in ops if op_type == "insert" and word]
    insertion_counts = Counter(inserted_words)
    discovered_insertions = set(inserted_words)

    deleted_words = [word for word, _, op_type in ops if op_type == "deletion" and word]
    deletion_counts = Counter(deleted_words)
    discovered_deletions = set(deleted_words)

    substitutions = [(ref, hyp) for ref, hyp, op_type in ops if op_type == "substitution" and ref and hyp]
    sub_counts = Counter(substitutions)
    discovered_substitutions = set(substitutions)

    discovered_repetitions = set()
    asr_words = asr_text.split()
    gt_words = gt_text.split()
    for i in range(len(asr_words) - 1):
        if asr_words[i] == asr_words[i + 1]:
            if i < len(gt_words) - 1 and gt_words[i] != gt_words[i + 1]:
                discovered_repetitions.add(asr_words[i])

    return {
        "insertions": discovered_insertions,
        "deletions": discovered_deletions,
        "substitutions": discovered_substitutions,
        "repetitions": discovered_repetitions,
        "insertion_counts": insertion_counts,
        "deletion_counts": deletion_counts,
        "substitution_counts": sub_counts,
    }


def score_sentence_pair_with_patterns(asr_sent, gt_sent, error_patterns, weights=None):
    if weights is None:
        weights = {"insertions": 3, "deletions": 2, "substitutions": 2, "repetitions": 3}

    score = 0
    asr_tokens = asr_sent.split()
    gt_tokens = gt_sent.split()

    score += weights["insertions"] * sum(
        1 for w in asr_tokens if w in error_patterns["insertions"]
    )
    score += weights["deletions"] * sum(
        1 for w in gt_tokens if w in error_patterns["deletions"]
    )

    for gt_w, asr_w in zip(gt_tokens, asr_tokens):
        if (gt_w, asr_w) in error_patterns["substitutions"]:
            score += weights["substitutions"]

    for i in range(len(asr_tokens) - 1):
        if (
            asr_tokens[i] == asr_tokens[i + 1]
            and asr_tokens[i] in error_patterns["repetitions"]
        ):
            score += weights["repetitions"]
            break

    return score


def pick_data_driven_targeted_sentences(truth_path, asr_path, num_sentences_to_pick=10):
    truth_text = Path(truth_path).read_text(encoding="utf-8", errors="ignore")
    asr_text = Path(asr_path).read_text(encoding="utf-8", errors="ignore")
    truth_clean = jiwer.RemovePunctuation()(truth_text).replace("\n", " ")
    asr_clean = jiwer.RemovePunctuation()(asr_text).replace("\n", " ")

    ops = alignment_ops(truth_clean, asr_clean)
    all_pairs = sentence_pairs_from_alignment(truth_clean, ops)
    error_pairs = [p for p in all_pairs if p[0] != p[1]]
    if not error_pairs:
        return [], []

    error_pairs.sort(key=lambda p: p[1])
    error_patterns = discover_error_patterns(ops, asr_clean, truth_clean)

    num_exhaustive_to_get = round(num_sentences_to_pick * 0.4)
    exhaustive_asr, _ = pick_exhaustive_phoneme_sentences(
        truth_path, asr_path, num_sentences_to_pick=num_exhaustive_to_get
    )
    exhaustive_set = set(exhaustive_asr)

    scored_pairs = []
    weights = {"insertions": 4, "deletions": 3, "substitutions": 2, "repetitions": 4}

    for asr_sent, gt_sent in error_pairs:
        score = 0
        if asr_sent in exhaustive_set:
            score += 10
        score += score_sentence_pair_with_patterns(asr_sent, gt_sent, error_patterns, weights)

        if score > 0:
            scored_pairs.append({"pair": (asr_sent, gt_sent), "score": score})

    scored_pairs.sort(key=lambda x: (-x["score"], x["pair"][0]))

    selected_pairs_dict = {}
    for item in scored_pairs:
        if len(selected_pairs_dict) >= num_sentences_to_pick:
            break
        asr, gt = item["pair"]
        if asr not in selected_pairs_dict:
            selected_pairs_dict[asr] = gt

    if len(selected_pairs_dict) < num_sentences_to_pick:
        needed = num_sentences_to_pick - len(selected_pairs_dict)
        remaining_pool = [p for p in error_pairs if p[0] not in selected_pairs_dict]
        if remaining_pool:
            deterministic_fill = remaining_pool[:needed]
            for asr, gt in deterministic_fill:
                selected_pairs_dict[asr] = gt

    final_pairs = list(selected_pairs_dict.items())
    asr_results = [p[0] for p in final_pairs]
    truth_results = [p[1] for p in final_pairs]
    return asr_results, truth_results
'''
'''
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
        available = min(need, len(pairs))
        for p in random.sample(pairs, available):
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



# --- Add this new DATA-DRIVEN function to your prompt.py file ---
from collections import Counter

# def pick_data_driven_targeted_sentences(truth_path, asr_path, num_sentences_to_pick=10):
#     """
#     An intelligent hybrid picker that learns a speaker's specific error
#     patterns from the data itself before selecting examples.

#     It discovers:
#     1.  Commonly inserted filler words (e.g., 'uh', 'like').
#     2.  Commonly repeated words (e.g., 'I I', 'and and').

#     It then uses this learned knowledge, along with phonetic diversity, to
#     select a highly effective and targeted set of examples.
#     """
#     # 1. Get all possible error pairs (same as before)
#     truth_text = Path(truth_path).read_text(encoding="utf-8", errors="ignore")
#     asr_text   = Path(asr_path).read_text(encoding="utf-8", errors="ignore")
#     truth_clean = jiwer.RemovePunctuation()(truth_text).replace("\n", " ")
#     asr_clean   = jiwer.RemovePunctuation()(asr_text).replace("\n", " ")

#     ops = alignment_ops(truth_clean, asr_clean)
#     all_pairs = sentence_pairs_from_alignment(truth_clean, ops)
#     error_pairs = [p for p in all_pairs if p[0] != p[1]]
#     if not error_pairs:
#         return [], []

#     # 2. Learn Speaker-Specific Error Patterns from the full alignment ops
#     #    This is the key data-driven step.
    
#     # Discover common insertions (filler words)
#     inserted_words = [word for _, word, op_type in ops if op_type == 'insert' and word]
#     insertion_counts = Counter(inserted_words)
#     # Consider any word inserted more than once as a likely filler
#     discovered_fillers = {word for word, count in insertion_counts.items() if count > 1}

#     # Discover common repetitions in the ASR text
#     discovered_repetitions = set()
#     asr_words = asr_clean.split()
#     for i in range(len(asr_words) - 1):
#         if asr_words[i] == asr_words[i+1]:
#             # Now check if this repetition is an error by looking at the GT
#             gt_words = truth_clean.split()
#             if i < len(gt_words) -1 and gt_words[i] != gt_words[i+1]:
#                  discovered_repetitions.add(asr_words[i])

#     print(f"  -> Discovered Fillers: {discovered_fillers or 'None'}")
#     print(f"  -> Discovered Repetitions: {discovered_repetitions or 'None'}")

#     # 3. Identify the top N phonetically diverse examples
#     num_exhaustive_to_get = round(num_sentences_to_pick * 0.7)
#     exhaustive_asr, _ = pick_exhaustive_phoneme_sentences(
#         truth_path, asr_path, num_sentences_to_pick=num_exhaustive_to_get
#     )
#     exhaustive_set = set(exhaustive_asr)

#     # 4. Score and rank all available error pairs based on learned patterns
#     scored_pairs = []
#     for asr_sent, gt_sent in error_pairs:
#         score = 0
#         asr_sent_words = asr_sent.split()
        
#         # High priority for being in the exhaustive set
#         if asr_sent in exhaustive_set:
#             score += 9
        
#         # High priority for containing a discovered filler word
#         if any(word in discovered_fillers for word in asr_sent_words):
#             score += 5
            
#         # High priority for containing a discovered repetition pattern
#         for i in range(len(asr_sent_words) - 1):
#             if asr_sent_words[i] == asr_sent_words[i+1] and asr_sent_words[i] in discovered_repetitions:
#                 score += 5
#                 break # Only score once per sentence for repetitions

#         if score > 0:
#             scored_pairs.append({'pair': (asr_sent, gt_sent), 'score': score + random.random()})

#     # 5. Sort and select the top N unique examples
#     scored_pairs.sort(key=lambda x: x['score'], reverse=True)

#     selected_pairs_dict = {}
#     for item in scored_pairs:
#         if len(selected_pairs_dict) >= num_sentences_to_pick:
#             break
#         asr, gt = item['pair']
#         if asr not in selected_pairs_dict:
#             selected_pairs_dict[asr] = gt
    
#     # Fallback if needed
#     if len(selected_pairs_dict) < num_sentences_to_pick:
#         needed = num_sentences_to_pick - len(selected_pairs_dict)
#         remaining_pool = [p for p in error_pairs if p[0] not in selected_pairs_dict]
#         if remaining_pool:
#             random_fill = random.sample(remaining_pool, min(needed, len(remaining_pool)))
#             for asr, gt in random_fill:
#                  selected_pairs_dict[asr] = gt

#     # 6. Convert final dictionary to lists and return
#     final_pairs = list(selected_pairs_dict.items())
    
#     asr_results = [p[0] for p in final_pairs]
#     truth_results = [p[1] for p in final_pairs]

#     return asr_results, truth_results



# A deterministic version of the exhaustive picker
# --- FINAL, FULLY DETERMINISTIC VERSION ---
# This is the only function you need to replace.

def pick_deterministic_exhaustive_phoneme_sentences(truth_path, asr_path, num_sentences_to_pick=5):
    """
    A fully deterministic version of the exhaustive picker. It uses a stable
    multi-level sort to break ties, ensuring the exact same output every run.
    """
    truth_text = Path(truth_path).read_text(encoding="utf-8", errors="ignore")
    asr_text   = Path(asr_path)  .read_text(encoding="utf-8", errors="ignore")

    truth = jiwer.RemovePunctuation()(truth_text).replace("\n", " ")
    asr   = jiwer.RemovePunctuation()(asr_text)  .replace("\n", " ")

    ops       = alignment_ops(truth, asr)
    all_pairs = sentence_pairs_from_alignment(truth, ops)
    pairs = [p for p in all_pairs if p[0] != p[1]]
    if not pairs:
        return [], []

    # Initial sort to ensure a consistent starting order before the loop
    pairs.sort(key=lambda p: p[1])

    phone_cache = {p: sentence_phone_set(p[1]) for p in pairs}
    selected, covered = [], set()
    
    # This loop is now fully deterministic
    while pairs and len(selected) < num_sentences_to_pick:
        
        # --- THE CRITICAL FIX IS HERE ---
        # We sort by three criteria to guarantee a stable order:
        #  1. Primary: Phonetic gain (descending)
        #  2. Secondary (tie-breaker): Sentence length (ascending, prefer shorter)
        #  3. Tertiary (final tie-breaker): The sentence string itself (alphabetical)
        pairs.sort(key=lambda p: (
            -len(phone_cache[p] - covered), # Negate for descending sort
            len(p[1]),                      # Ascending length
            p[1]                            # Ascending alphabetical
        ))
        
        best = pairs[0]
        best_gain = phone_cache[best] - covered

        if not best_gain and len(covered) > 0:
            # Break if no sentence can add any new phones
            # (and we've already started covering phones)
            break
            
        selected.append(best)
        covered |= phone_cache[best]
        pairs.remove(best)

    # Fallback is now also deterministic
    if len(selected) < num_sentences_to_pick and pairs:
        need = num_sentences_to_pick - len(selected)
        available = min(need, len(pairs))
        # `pairs` is already sorted deterministically, so just take the next ones
        for p in pairs[:available]:
            selected.append(p)

    return [p[0] for p in selected], [p[1] for p in selected]

# --- Use this new DETERMINISTIC function ---
from collections import Counter

def pick_filler_heavy_sentences(truth_path, asr_path, num_sentences_to_pick=5):
    """
    Deterministic filler-focused picker. Uses the helper functions to discover
    and score insertion, deletion, and substitution patterns.
    """
    # 1. Clean inputs
    truth_text = Path(truth_path).read_text(encoding="utf-8", errors="ignore")
    asr_text   = Path(asr_path)  .read_text(encoding="utf-8", errors="ignore")
    truth_clean = jiwer.RemovePunctuation()(truth_text).replace("\n", " ")
    asr_clean   = jiwer.RemovePunctuation()(asr_text).replace("\n", " ")

    # 2. Align ops + sentence pairs
    ops = alignment_ops(truth_clean, asr_clean)
    all_pairs = sentence_pairs_from_alignment(truth_clean, ops)
    error_pairs = [p for p in all_pairs if p[0] != p[1]]
    if not error_pairs:
        return [], []

    # 3. Use helper function to discover error patterns
    error_patterns = discover_error_patterns(ops, asr_clean, truth_clean)
    
    print(f"  -> Discovered Fillers (Insert): {error_patterns['insertions'] or 'None'}")
    print(f"  -> Discovered Deletes: {error_patterns['deletions'] or 'None'}")
    print(f"  -> Discovered Subs: {len(error_patterns['substitutions'])} pairs")
    print(f"  -> Discovered Repetitions: {error_patterns['repetitions'] or 'None'}")

    # 4. Score all error pairs using helper function
    # Focus heavily on insertions and deletions for filler detection
    weights = {'insertions': 5, 'deletions': 4, 'substitutions': 1, 'repetitions': 3}
    scored_pairs = []
    
    for asr_sent, gt_sent in error_pairs:
        score = score_sentence_pair_with_patterns(asr_sent, gt_sent, error_patterns, weights)
        if score > 0:
            scored_pairs.append({'pair': (asr_sent, gt_sent), 'score': score})

    # 5. Sort + trim
    scored_pairs.sort(key=lambda x: (-x['score'], x['pair'][0]))

    selected_pairs_dict = {}
    for item in scored_pairs:
        if len(selected_pairs_dict) >= num_sentences_to_pick:
            break
        asr, gt = item['pair']
        if asr not in selected_pairs_dict:
            selected_pairs_dict[asr] = gt

    # 6. Deterministic fallback if needed
    if len(selected_pairs_dict) < num_sentences_to_pick:
        needed = num_sentences_to_pick - len(selected_pairs_dict)
        remaining_pool = [p for p in error_pairs if p[0] not in selected_pairs_dict]
        if remaining_pool:
            deterministic_fill = remaining_pool[:needed]
            for asr, gt in deterministic_fill:
                selected_pairs_dict[asr] = gt

    # 7. Final return
    final_pairs = list(selected_pairs_dict.items())
    return [p[0] for p in final_pairs], [p[1] for p in final_pairs]


def pick_filler_heavy_sentences_easy(
        truth_path: str,
        asr_path: str,
        num_sentences_to_pick: int = 5,
        verbose: bool = False):
    """
    Simple, deterministic filler-insertion picker.

    • Learns *all* inserted words (no frequency threshold).
    • Scores each (ASR, GT) sentence pair by how many of those
      inserted words it contains on the ASR side.
    • Returns up to `num_sentences_to_pick` highest-scoring pairs.
    • If none score > 0, falls back to pick_random_error_sentences().
    """

    def dbg(msg):          # local tiny logger
        if verbose:
            print(msg)

    # ── 1. read & strip punctuation ──
    # errors="ignore" silences decoding problems if the transcript
    # has stray bytes; the alternative is "strict", which would crash.
    truth_text = Path(truth_path).read_text(encoding="utf-8", errors="ignore")
    asr_text   = Path(asr_path)  .read_text(encoding="utf-8", errors="ignore")
    truth_clean = jiwer.RemovePunctuation()(truth_text).replace("\n", " ")
    asr_clean   = jiwer.RemovePunctuation()(asr_text)  .replace("\n", " ")

    # ── 2. align & extract sentence pairs ──
    ops        = alignment_ops(truth_clean, asr_clean)
    all_pairs  = sentence_pairs_from_alignment(truth_clean, ops)
    error_pairs = [p for p in all_pairs if p[0] != p[1]]
    dbg(f"Total error pairs: {len(error_pairs)}")
    if not error_pairs:
        return [], []

    # ── 3. discover every inserted token ──
    inserted_tokens = [hyp for _, hyp, op in ops if op == "insert" and hyp]
    dbg(f"Inserted token sample: {inserted_tokens[:10]}")
    discovered_fillers = set(inserted_tokens)        # no freq filter
    dbg(f"Discovered fillers = {discovered_fillers or '∅'}")

    # ── 4. score each error pair ──
    scored = []
    for asr_sent, gt_sent in error_pairs:
        n = sum(1 for w in asr_sent.split() if w in discovered_fillers)
        if n:
            scored.append({'pair': (asr_sent, gt_sent), 'score': n})

    # ── 5. fallback if nothing scored ──
    if not scored:
        dbg("⚠️  No filler-heavy sentences found – using random fallback")
        return pick_random_error_sentences(
            truth_path, asr_path, num_sentences_to_pick)

    # ── 6. sort & return ──
    scored.sort(key=lambda x: (-x['score'], x['pair'][0]))   # desc score, asc text
    top = scored[:num_sentences_to_pick]
    dbg(f"Picked {len(top)} sentence pairs")

    asr_out   = [item['pair'][0] for item in top]
    truth_out = [item['pair'][1] for item in top]
    return asr_out, truth_out

def discover_error_patterns(ops, asr_text, gt_text):
    """
    Helper function to discover insertion, deletion, and substitution patterns
    from alignment operations. Used by both filler-heavy and data-driven strategies.
    
    Returns:
        dict with keys: 'insertions', 'deletions', 'substitutions', 'repetitions'
    """
    # Discover insertions (potential fillers)
    inserted_words = [word for _, word, op_type in ops if op_type == 'insert' and word]
    insertion_counts = Counter(inserted_words)
    discovered_insertions = set(inserted_words)  # All insertions, no frequency filter
    
    # Discover deletions 
    deleted_words = [word for word, _, op_type in ops if op_type == 'deletion' and word]
    deletion_counts = Counter(deleted_words)
    discovered_deletions = set(deleted_words)  # All deletions
    
    # Discover substitutions
    substitutions = [(ref, hyp) for ref, hyp, op_type in ops if op_type == 'substitution' and ref and hyp]
    sub_counts = Counter(substitutions)
    discovered_substitutions = set(substitutions)  # All substitutions
    
    # Discover repetitions in ASR that aren't in ground truth
    discovered_repetitions = set()
    asr_words = asr_text.split()
    gt_words = gt_text.split()
    for i in range(len(asr_words) - 1):
        if asr_words[i] == asr_words[i+1]:
            if i < len(gt_words) - 1 and gt_words[i] != gt_words[i+1]:
                discovered_repetitions.add(asr_words[i])
    
    return {
        'insertions': discovered_insertions,
        'deletions': discovered_deletions,
        'substitutions': discovered_substitutions,
        'repetitions': discovered_repetitions,
        'insertion_counts': insertion_counts,
        'deletion_counts': deletion_counts,
        'substitution_counts': sub_counts
    }

def score_sentence_pair_with_patterns(asr_sent, gt_sent, error_patterns, weights=None):
    """
    Helper function to score a sentence pair based on discovered error patterns.
    
    Args:
        asr_sent: ASR sentence
        gt_sent: Ground truth sentence
        error_patterns: Dict from discover_error_patterns()
        weights: Dict of weights for different error types
    
    Returns:
        int: Score for this sentence pair
    """
    if weights is None:
        weights = {'insertions': 3, 'deletions': 2, 'substitutions': 2, 'repetitions': 3}
    
    score = 0
    asr_tokens = asr_sent.split()
    gt_tokens = gt_sent.split()
    
    # Score insertions (filler words in ASR)
    score += weights['insertions'] * sum(1 for w in asr_tokens if w in error_patterns['insertions'])
    
    # Score deletions (words missing from ASR that were in GT)
    score += weights['deletions'] * sum(1 for w in gt_tokens if w in error_patterns['deletions'])
    
    # Score substitutions
    for gt_w, asr_w in zip(gt_tokens, asr_tokens):
        if (gt_w, asr_w) in error_patterns['substitutions']:
            score += weights['substitutions']
    
    # Score repetitions in ASR
    for i in range(len(asr_tokens) - 1):
        if (asr_tokens[i] == asr_tokens[i+1] and 
            asr_tokens[i] in error_patterns['repetitions']):
            score += weights['repetitions']
            break  # Only score once per sentence
    
    return score

def pick_data_driven_targeted_sentences(truth_path, asr_path, num_sentences_to_pick=10):
    """
    Enhanced data-driven picker that uses comprehensive error pattern discovery.
    Combines phonetic diversity with learned insertion/deletion/substitution patterns
    from the filler-heavy approach for maximum effectiveness.
    """
    # 1. Get all possible error pairs
    truth_text = Path(truth_path).read_text(encoding="utf-8", errors="ignore")
    asr_text   = Path(asr_path).read_text(encoding="utf-8", errors="ignore")
    truth_clean = jiwer.RemovePunctuation()(truth_text).replace("\n", " ")
    asr_clean   = jiwer.RemovePunctuation()(asr_text).replace("\n", " ")

    ops = alignment_ops(truth_clean, asr_clean)
    all_pairs = sentence_pairs_from_alignment(truth_clean, ops)
    error_pairs = [p for p in all_pairs if p[0] != p[1]]
    if not error_pairs:
        return [], []

    # Sort the initial pool for stability
    error_pairs.sort(key=lambda p: p[1])

    # 2. Discover comprehensive error patterns using the helper function
    error_patterns = discover_error_patterns(ops, asr_clean, truth_clean)
    
    print(f"  -> Discovered Insertions: {error_patterns['insertions'] or 'None'}")
    print(f"  -> Discovered Deletions: {error_patterns['deletions'] or 'None'}")
    print(f"  -> Discovered Substitutions: {len(error_patterns['substitutions'])} pairs")
    print(f"  -> Discovered Repetitions: {error_patterns['repetitions'] or 'None'}")

    # 3. Get phonetically diverse examples (reduced proportion to make room for error patterns)
    num_exhaustive_to_get = round(num_sentences_to_pick * 0.4)  # Reduced from 0.7
    exhaustive_asr, _ = pick_deterministic_exhaustive_phoneme_sentences(
        truth_path, asr_path, num_sentences_to_pick=num_exhaustive_to_get
    )
    exhaustive_set = set(exhaustive_asr)

    # 4. Score all error pairs using both phonetic and error pattern criteria
    scored_pairs = []
    weights = {'insertions': 4, 'deletions': 3, 'substitutions': 2, 'repetitions': 4}
    
    for asr_sent, gt_sent in error_pairs:
        score = 0
        
        # High priority for phonetic diversity
        if asr_sent in exhaustive_set:
            score += 10
        
        # Score based on error patterns using helper function
        pattern_score = score_sentence_pair_with_patterns(asr_sent, gt_sent, error_patterns, weights)
        score += pattern_score

        if score > 0:
            scored_pairs.append({'pair': (asr_sent, gt_sent), 'score': score})

    # 5. Sort by score (desc), then by ASR sentence (asc) for stable tie-breaking
    scored_pairs.sort(key=lambda x: (-x['score'], x['pair'][0]))

    selected_pairs_dict = {}
    for item in scored_pairs:
        if len(selected_pairs_dict) >= num_sentences_to_pick:
            break
        asr, gt = item['pair']
        if asr not in selected_pairs_dict:
            selected_pairs_dict[asr] = gt
    
    # 6. Deterministic fallback if needed
    if len(selected_pairs_dict) < num_sentences_to_pick:
        needed = num_sentences_to_pick - len(selected_pairs_dict)
        remaining_pool = [p for p in error_pairs if p[0] not in selected_pairs_dict]
        if remaining_pool:
            deterministic_fill = remaining_pool[:needed]
            for asr, gt in deterministic_fill:
                selected_pairs_dict[asr] = gt

    # 7. Final conversion and return
    final_pairs = list(selected_pairs_dict.items())
    
    asr_results = [p[0] for p in final_pairs]
    truth_results = [p[1] for p in final_pairs]

    return asr_results, truth_results


def pick_data_driven_targeted_sentences_simple(truth_path, asr_path, num_sentences_to_pick=10):
    """
    A fully deterministic and quantitative picker. Given the same input files,
    it will ALWAYS return the exact same list of sentences in the same order.
    
    It removes all randomness for stable, repeatable results.
    """
    # 1. Get all possible error pairs (same as before)
    truth_text = Path(truth_path).read_text(encoding="utf-8", errors="ignore")
    asr_text   = Path(asr_path).read_text(encoding="utf-8", errors="ignore")
    truth_clean = jiwer.RemovePunctuation()(truth_text).replace("\n", " ")
    asr_clean   = jiwer.RemovePunctuation()(asr_text).replace("\n", " ")

    ops = alignment_ops(truth_clean, asr_clean)
    all_pairs = sentence_pairs_from_alignment(truth_clean, ops)
    error_pairs = [p for p in all_pairs if p[0] != p[1]]
    if not error_pairs:
        return [], []

    # Sort the initial pool for stability
    error_pairs.sort(key=lambda p: p[1]) # Sort by ground-truth

    # 2. Learn Speaker-Specific Error Patterns (this part is already deterministic)
    inserted_words = [word for _, word, op_type in ops if op_type == 'insert' and word]
    insertion_counts = Counter(inserted_words)
    discovered_fillers = {word for word, count in insertion_counts.items() if count > 1}
    
    discovered_repetitions = set()
    asr_words = asr_clean.split()
    for i in range(len(asr_words) - 1):
        if asr_words[i] == asr_words[i+1]:
            gt_words = truth_clean.split()
            if i < len(gt_words) -1 and gt_words[i] != gt_words[i+1]:
                 discovered_repetitions.add(asr_words[i])
    
    print(f"  -> Discovered Fillers: {discovered_fillers or 'None'}")
    print(f"  -> Discovered Repetitions: {discovered_repetitions or 'None'}")

    # 3. MODIFICATION: Call the new deterministic exhaustive picker
    num_exhaustive_to_get = round(num_sentences_to_pick * 0.7)
    exhaustive_asr, _ = pick_deterministic_exhaustive_phoneme_sentences(
        truth_path, asr_path, num_sentences_to_pick=num_exhaustive_to_get
    )
    exhaustive_set = set(exhaustive_asr)

    # 4. Score and rank all available error pairs
    scored_pairs = []
    for asr_sent, gt_sent in error_pairs:
        score = 0
        asr_sent_words = asr_sent.split()
        
        if asr_sent in exhaustive_set: score += 9
        if any(word in discovered_fillers for word in asr_sent_words): score += 5
        for i in range(len(asr_sent_words) - 1):
            if asr_sent_words[i] == asr_sent_words[i+1] and asr_sent_words[i] in discovered_repetitions:
                score += 5
                break

        if score > 0:
            # MODIFICATION: Removed the random tie-breaker
            scored_pairs.append({'pair': (asr_sent, gt_sent), 'score': score})

    # 5. Sort by score (desc), then by ASR sentence (asc) for stable tie-breaking
    scored_pairs.sort(key=lambda x: (-x['score'], x['pair'][0]))

    selected_pairs_dict = {}
    for item in scored_pairs:
        if len(selected_pairs_dict) >= num_sentences_to_pick: break
        asr, gt = item['pair']
        if asr not in selected_pairs_dict:
            selected_pairs_dict[asr] = gt
    
    # 6. MODIFICATION: Replace random fallback with a deterministic fill
    if len(selected_pairs_dict) < num_sentences_to_pick:
        needed = num_sentences_to_pick - len(selected_pairs_dict)
        # Create a pool of remaining sentences that is already sorted
        remaining_pool = [p for p in error_pairs if p[0] not in selected_pairs_dict]
        if remaining_pool:
            # Take the first N items from the stable list
            deterministic_fill = remaining_pool[:needed]
            for asr, gt in deterministic_fill:
                 selected_pairs_dict[asr] = gt

    # 7. Final conversion and return
    # The dictionary preserves insertion order in modern Python, but we can sort for safety
    final_pairs = list(selected_pairs_dict.items())
    
    asr_results = [p[0] for p in final_pairs]
    truth_results = [p[1] for p in final_pairs]

    return asr_results, truth_results
