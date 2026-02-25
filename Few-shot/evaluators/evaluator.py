import os
import csv
import inspect
import jiwer
from utils.example_generator import generate_dynamic_examples, ExampleStrategy
from utils.raw_metrics_calculator import calculate_dynamic_raw_metrics
from utils.metrics_utils import calculate_wer, calculate_cer, calculate_semantic_similarity, model
from utils.file_utils import get_data_paths, get_asr_file_path, ASRSystem
from utils.text_processor import remove_aligned_segments

def read_asr_file(filepath):
    """Read ASR file content"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read().strip()

def write_per_file_evaluation_csv(output_directory: str, raw_metrics: dict, improved_metrics: dict):
    csv_filename = "evaluation_metrics.csv"
    csv_path = os.path.join(output_directory, csv_filename)
    header = ['Type', 'WER', 'CER', 'SIM']
    raw_wer = raw_metrics.get('WER_ASR', 'N/A') if raw_metrics else 'N/A'
    raw_cer = raw_metrics.get('CER_ASR', 'N/A') if raw_metrics else 'N/A'
    raw_sim = raw_metrics.get('S_ASR', 'N/A') if raw_metrics else 'N/A'
    imp_wer = improved_metrics.get('WER_IMP', 'N/A') if improved_metrics else 'N/A'
    imp_cer = improved_metrics.get('CER_IMP', 'N/A') if improved_metrics else 'N/A'
    imp_sim = improved_metrics.get('S_IMP', 'N/A') if improved_metrics else 'N/A'
    rows = [
        ['Raw ASR', raw_wer, raw_cer, raw_sim],
        ['Improved', imp_wer, imp_cer, imp_sim]
    ]
    try:
        os.makedirs(output_directory, exist_ok=True)
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
    except Exception as e:
        print(f"Failed to write CSV {csv_path}: {e}")

def evaluate_with_example_pairs(postprocess_function, experiment_folder_for_run: str, num_sentences: int, 
                               strategy: ExampleStrategy = None, asr_system: ASRSystem = ASRSystem.AZURE_SPEECH):
    data_paths = get_data_paths(asr_system)
    ground_truth_folder = data_paths["ground_truth"]
    asr_folder = data_paths["asr_raw"]
    file_bases = ["aprocsa1554a", "aprocsa1713a", "aprocsa1731a", "aprocsa1738a", "aprocsa1833a", "aprocsa1944a"]
    sig = inspect.signature(postprocess_function)
    requires_audio_file = 'audio_file_path' in sig.parameters

    print(f"Running evaluation with {asr_system.value} ASR system")
    print(f"ASR data folder: {asr_folder}")

    for file_base in file_bases:
        sample_base_output_dir = os.path.join(experiment_folder_for_run, file_base)
        final_output_dir = os.path.join(sample_base_output_dir, f"{num_sentences}_sentences")
        os.makedirs(final_output_dir, exist_ok=True)

        print(f"Processing {file_base} for {num_sentences} examples. Output to: {final_output_dir}")

        try:
            # Pass the strategy and ASR system to the example generator
            if strategy is not None:
                asr_example_sentences, gt_example_sentences, example_pairs_string_for_prompt = generate_dynamic_examples(
                    file_base, num_sentences, ground_truth_folder, asr_folder, strategy=strategy, asr_system=asr_system
                )
            else:
                # Fallback to default behavior
                asr_example_sentences, gt_example_sentences, example_pairs_string_for_prompt = generate_dynamic_examples(
                    file_base, num_sentences, ground_truth_folder, asr_folder, asr_system=asr_system
                )

            if example_pairs_string_for_prompt is None: 
                print(f"Skipping {file_base} for {num_sentences} examples: example generation failure.")
                continue
            
            if num_sentences > 0 and example_pairs_string_for_prompt:
                prompt_examples_filename = os.path.join(final_output_dir, "PROMPT_EXAMPLES.txt")
                with open(prompt_examples_filename, 'w', encoding='utf-8') as f_prompt_ex:
                    f_prompt_ex.write(example_pairs_string_for_prompt)
            
            # Use the new ASR file path function
            full_asr_text_path = get_asr_file_path(file_base, asr_system)
            full_gt_path = os.path.join(ground_truth_folder, f"{file_base}.txt")
            
            if not (os.path.exists(full_asr_text_path) and os.path.exists(full_gt_path)):
                print(f"ASR or GT file not found for {file_base}, skipping...")
                print(f"ASR path: {full_asr_text_path}")
                print(f"GT path: {full_gt_path}")
                continue
                
            full_asr_text = read_asr_file(full_asr_text_path)
            full_ground_truth = read_asr_file(full_gt_path)

            cleaned_full_asr_text = " ".join(jiwer.RemovePunctuation()(full_asr_text).split())
            cleaned_full_gt_text = " ".join(jiwer.RemovePunctuation()(full_ground_truth).split())
            
            print("Removing example sentences using alignment-aware method...")
            gt_text_with_examples_removed, asr_text_with_examples_removed = remove_aligned_segments(
                cleaned_full_gt_text,
                cleaned_full_asr_text,
                gt_example_sentences or []
            )

            # Save the correctly processed versions with examples removed
            with open(os.path.join(final_output_dir, "ASR_examples_removed.txt"), 'w', encoding='utf-8') as f_asr_rem:
                f_asr_rem.write(asr_text_with_examples_removed)
            with open(os.path.join(final_output_dir, "GT_examples_removed.txt"), 'w', encoding='utf-8') as f_gt_rem:
                f_gt_rem.write(gt_text_with_examples_removed)

            if requires_audio_file:
                reference_gt_for_metrics = cleaned_full_gt_text
                hypothesis_asr_for_raw_metrics = cleaned_full_asr_text
                print("Using FULL GT and ASR for metrics (multimodal model).")
            else:
                reference_gt_for_metrics = gt_text_with_examples_removed
                hypothesis_asr_for_raw_metrics = asr_text_with_examples_removed
                print("Using EXAMPLE-REMOVED GT and ASR for metrics (text-only model).")

            raw_asr_metrics = calculate_dynamic_raw_metrics(reference_gt_for_metrics, hypothesis_asr_for_raw_metrics)
            raw_asr_metrics = raw_asr_metrics if raw_asr_metrics else {}

            asr_input_for_llm = asr_text_with_examples_removed

            improved_text_raw = None
            if requires_audio_file:
                audio_file_path = os.path.join(data_paths["audio_files"], f"{file_base}.wav")
                if not os.path.exists(audio_file_path):
                    print(f"Audio file {audio_file_path} not found, multimodal postprocessing may fail for {file_base}.")
                improved_text_raw = postprocess_function(audio_file_path, cleaned_full_asr_text, example_pairs_string_for_prompt)
            else:
                improved_text_raw = postprocess_function(asr_input_for_llm, example_pairs_string_for_prompt)
            
            
            cleaned_improved_text_raw = " ".join(jiwer.RemovePunctuation()(improved_text_raw).split())
                
            improved_metrics_calculated = {}
            if requires_audio_file:
                improved_metrics_calculated = {
                    "WER_IMP": calculate_wer(cleaned_full_gt_text, cleaned_improved_text_raw),
                    "CER_IMP": calculate_cer(cleaned_full_gt_text, cleaned_improved_text_raw),
                    "S_IMP": calculate_semantic_similarity(cleaned_full_gt_text, cleaned_improved_text_raw, model)
                }
            else:
                improved_metrics_calculated = {
                    "WER_IMP": calculate_wer(reference_gt_for_metrics, cleaned_improved_text_raw),
                    "CER_IMP": calculate_cer(reference_gt_for_metrics, cleaned_improved_text_raw),
                    "S_IMP": calculate_semantic_similarity(reference_gt_for_metrics, cleaned_improved_text_raw, model)
                }

            # Save the improved text
            improved_filename = os.path.join(final_output_dir, "IMPROVED.txt")
            with open(improved_filename, 'w', encoding='utf-8') as f_improved:
                f_improved.write(improved_text_raw)

            write_per_file_evaluation_csv(final_output_dir, raw_asr_metrics, improved_metrics_calculated)

            print(f"Completed processing for {file_base} with {num_sentences} examples")

        except Exception as e:
            print(f"FAILED to process {file_base}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\nEvaluation loop for {num_sentences} examples completed.")