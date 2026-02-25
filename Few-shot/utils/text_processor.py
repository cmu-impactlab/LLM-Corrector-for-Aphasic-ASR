import jiwer

def find_example_indices_in_alignment(gt_words, alignment_chunks, gt_example_sentences):
    """
    Find indices of GT words that are part of example sentences.
    Handles cases where sentences are not contiguous due to deletions.
    """
    indices_to_remove = set()

    gt_alignment_trace = []
    for chunk in alignment_chunks:
        if chunk.type in ["equal", "substitute", "delete"]:
            gt_alignment_trace.extend(range(chunk.ref_start_idx, chunk.ref_end_idx))
    
    aligned_gt_words = [gt_words[i] for i in gt_alignment_trace]
    
    for sentence in gt_example_sentences:
        sentence_words = sentence.split()
        if not sentence_words:
            continue
        
        for i in range(len(aligned_gt_words) - len(sentence_words) + 1):
            candidate = aligned_gt_words[i:i + len(sentence_words)]
            if candidate == sentence_words:
                original_indices_for_match = gt_alignment_trace[i:i + len(sentence_words)]
                indices_to_remove.update(original_indices_for_match)
                break
    
    return indices_to_remove


def remove_aligned_segments(full_gt, full_hyp, gt_sentences_to_remove):
    """
    Remove corresponding segments from GT and hypothesis text using alignment-aware method.
    """
    if not gt_sentences_to_remove:
        return full_gt, full_hyp

    gt_words = full_gt.split()
    hyp_words = full_hyp.split()

    try:
        alignment_output = jiwer.process_words(full_gt, full_hyp)
        if not alignment_output.alignments:
            raise ValueError("Jiwer alignment returned empty.")
        alignment_chunks = alignment_output.alignments[0]
    except (ValueError, IndexError) as e:
        print(f"Warning: Jiwer alignment failed ({e}). Could not perform aligned removal.")
        return full_gt, full_hyp

    gt_indices_to_remove = find_example_indices_in_alignment(
        gt_words,
        alignment_chunks,
        gt_sentences_to_remove
    )
    
    new_gt_words = []
    new_hyp_words = []
    
    for chunk in alignment_chunks:
        if chunk.type in ["equal", "substitute", "delete"]:
            for i, gt_idx in enumerate(range(chunk.ref_start_idx, chunk.ref_end_idx)):
                if gt_idx not in gt_indices_to_remove:
                    new_gt_words.append(gt_words[gt_idx])
                    if chunk.type in ["equal", "substitute"]:
                        hyp_idx = chunk.hyp_start_idx + i
                        new_hyp_words.append(hyp_words[hyp_idx])

        elif chunk.type == "insert":
            keep_insertion = True
            if chunk.ref_start_idx > 0:
                preceding_gt_idx = chunk.ref_start_idx - 1
                if preceding_gt_idx in gt_indices_to_remove:
                    keep_insertion = False
            
            if keep_insertion:
                hyp_slice = hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]
                new_hyp_words.extend(hyp_slice)

    processed_gt = " ".join(new_gt_words)
    processed_hyp = " ".join(new_hyp_words)

    return processed_gt, processed_hyp