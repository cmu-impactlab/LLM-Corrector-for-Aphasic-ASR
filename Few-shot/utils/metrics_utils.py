import numpy as np
import jiwer
import re
from jiwer import wer, cer
from sentence_transformers import SentenceTransformer, util
from wtpsplit import WtP

# Initialize models
model = SentenceTransformer("all-MiniLM-L6-v2")
_wtp = WtP("wtp-bert-mini")

# Constants
MAX_TOKENS = 30
MIN_TOKENS = 3

def clean_text(text):
    """
    Clean text by removing punctuation, newlines, and special characters.
    Uses the project's original cleaning logic plus jiwer for comprehensive removal.
    """
    # Original cleaning logic
    text = text.replace("\n\n", " ").replace("\n", " ")
    text = text.replace("—", " ")
    text = text.replace("[", "").replace("]", "")
    text = text.replace(",", "")
    text = text.replace(".", "")
    text = text.replace("?", "")
    
    # Use jiwer to remove any remaining punctuation
    text = jiwer.RemovePunctuation()(text)
    text = text.lower()
    
    # Clean up multiple spaces and strip
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_into_sentences(text):
    """
    Use WtP to split a raw stream (no punctuation) into sentences.
    """
    return list(_wtp.split([text]))[0]

def alignment_ops(truth, hyp):
    """
    Perform word‐level alignment of truth vs. ASR using jiwer.process_words.
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
    Break the aligned word ops into sentence‐aligned pairs, using WtP
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
    4. SBERT‐encode each aligned (truth, hyp) sentence pair.
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

def calculate_semantic_similarity(reference, hypothesis, model=model):
    """
    Calculate semantic similarity using the new alignment-based approach.
    """
    return semantic_sim(reference, hypothesis, model)

def calculate_wer(reference, hypothesis):
    """
    Calculate Word Error Rate between reference and hypothesis texts.
    Uses the same clean_text function for consistent preprocessing.
    """
    ref_clean = clean_text(reference)
    hyp_clean = clean_text(hypothesis)
    return wer(ref_clean, hyp_clean)

def calculate_cer(reference, hypothesis):
    """
    Calculate Character Error Rate between reference and hypothesis texts.
    Uses the same clean_text function for consistent preprocessing.
    """
    ref_clean = clean_text(reference)
    hyp_clean = clean_text(hypothesis)
    return cer(ref_clean, hyp_clean)

def calculate_all_metrics(reference, hypothesis, model=model):
    """
    Calculate all metrics (WER, CER, Semantic Similarity) in one function.
    Returns a dictionary with all metrics.
    """
    return {
        "WER": calculate_wer(reference, hypothesis),
        "CER": calculate_cer(reference, hypothesis),
        "Semantic_Similarity": calculate_semantic_similarity(reference, hypothesis, model)
    } 