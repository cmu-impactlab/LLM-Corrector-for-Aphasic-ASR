import os
import json
from utils.metrics_utils import calculate_wer, calculate_cer, calculate_semantic_similarity, model

def calculate_dynamic_raw_metrics(ground_truth_text: str, asr_text: str):
    """
    Calculate raw ASR metrics dynamically from text inputs.

    Args:
        ground_truth_text (str): The ground truth text.
        asr_text (str): The ASR output text.

    Returns:
        dict: {"WER_ASR": float, "CER_ASR": float, "S_ASR": float} or None if an error occurs.
    """
    try:
        # Calculate raw metrics
        wer_asr = calculate_wer(ground_truth_text, asr_text)
        cer_asr = calculate_cer(ground_truth_text, asr_text)
        s_asr = calculate_semantic_similarity(ground_truth_text, asr_text, model)
        
        return {
            "WER_ASR": wer_asr,
            "CER_ASR": cer_asr,
            "S_ASR": s_asr
        }
        
    except Exception as e:
        print(f"Failed to calculate dynamic raw metrics: {e}")
        return None

# Old caching functions are removed:
# - calculate_raw_metrics_for_all_samples
# - load_raw_metrics_from_cache
# - get_raw_metrics

# The main guard is removed as this module will primarily be imported.
# if __name__ == "__main__":
# calculate_raw_metrics_for_all_samples() 