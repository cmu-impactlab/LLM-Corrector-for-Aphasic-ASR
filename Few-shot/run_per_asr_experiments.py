#!/usr/bin/env python3
"""
Multi-ASR Experiment Runner

Runs experiments for selected ASR systems.
"""

import os
import sys
import argparse
import time
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.file_utils import ASRSystem, get_available_asr_systems, get_data_paths
from utils.example_generator import generate_dynamic_examples
from evaluators.evaluator import evaluate_with_example_pairs
from models.azure_gpt import postprocess_4_1_4

def run_experiment(asr_system: str, strategy: str, run_number: int, samples: list = None, sentences: list = None):
    """
    Run a single experiment for the given ASR system and strategy.
    
    Args:
        asr_system: Name of ASR system (e.g., "Whisper", "Gemini")
        strategy: Strategy name (e.g., "data_driven", "exhaustive_phoneme")
        run_number: Run number (1-5)
        samples: List of specific samples to process (default: all available)
        sentences: List of specific sentence counts to process (default: [2,4,6,8,10])
    """
    
    print(f"\nStarting {asr_system} - {strategy} - Run {run_number}")
    print("=" * 60)
    
    timestamp = datetime.now().strftime("%m%d%H")
    experiment_id = f"{timestamp}_{strategy}_run{run_number}"
    
    available_samples = []
    try:
        gt_path = "data/ground_truth"
        if os.path.exists(gt_path):
            for file in os.listdir(gt_path):
                if file.endswith('.txt'):
                    sample_id = file.replace('.txt', '')
                    asr_file_path = os.path.join('data', 'asr_raw', asr_system, f"{sample_id} {asr_system}.txt")
                    if os.path.exists(asr_file_path):
                        available_samples.append(sample_id)
    except Exception as e:
        print(f"Error getting samples for {asr_system}: {e}")
        return False
    
    if not available_samples:
        print(f"No samples found for {asr_system}")
        return False
    
    # Use provided samples or all available samples
    if samples:
        process_samples = [s for s in samples if s in available_samples]
        if not process_samples:
            print(f"None of the specified samples are available for {asr_system}")
            return False
    else:
        process_samples = available_samples
    
    print(f"Processing {len(process_samples)} samples: {process_samples}")
    
    sentence_counts = sentences if sentences else [2, 4, 6, 8, 10]
    
    total_experiments = len(sentence_counts)
    current_experiment = 0
    
    for num_sentences in sentence_counts:
        current_experiment += 1
        print(f"\n[{current_experiment}/{total_experiments}] Processing {num_sentences} example sentences for all samples")
        
        sample_id = process_samples[0]
        
        try:
            from utils.example_generator import get_strategy_from_name
            strategy_enum = get_strategy_from_name(strategy)
            
            asr_examples, gt_examples, examples_string = generate_dynamic_examples(
                file_base=sample_id,
                num_sentences_to_pick=num_sentences,
                ground_truth_folder="data/ground_truth",
                asr_folder=f"data/asr_raw/{asr_system}",  # This gets ignored anyway
                strategy=strategy_enum,
                asr_system=ASRSystem.AZURE_SPEECH if asr_system == "Azure"
                          else ASRSystem.WHISPER if asr_system == "Whisper"
                          else ASRSystem.GEMINI if asr_system == "Gemini"
                          else ASRSystem.ASSEMBLYAI if asr_system == "AssemblyAI"
                          else ASRSystem.AWS if asr_system == "AWS"
                          else ASRSystem.DEEPGRAM if asr_system == "Deepgram"
                          else ASRSystem.ELEVENLABS if asr_system == "ElevenLabs"
                          else ASRSystem.GCP if asr_system == "GCP"
                          else ASRSystem.GLADIA if asr_system == "Gladia"
                          else ASRSystem.SPEECHMATICS if asr_system == "Speechmatics"
                          else ASRSystem.ASSEMBLYAI  # Default case
            )
            
            if not asr_examples or not gt_examples:
                print(f"    No examples generated for {num_sentences} sentences")
                continue
            
            print(f"    Generated {len(asr_examples)} example pairs using {strategy} with {asr_system}")
            
            asr_system_enum = (ASRSystem.AZURE_SPEECH if asr_system == "Azure"
                             else ASRSystem.WHISPER if asr_system == "Whisper"
                             else ASRSystem.GEMINI if asr_system == "Gemini"
                             else ASRSystem.ASSEMBLYAI if asr_system == "AssemblyAI"
                             else ASRSystem.AWS if asr_system == "AWS"
                             else ASRSystem.DEEPGRAM if asr_system == "Deepgram"
                             else ASRSystem.ELEVENLABS if asr_system == "ElevenLabs"
                             else ASRSystem.GCP if asr_system == "GCP"
                             else ASRSystem.GLADIA if asr_system == "Gladia"
                             else ASRSystem.SPEECHMATICS if asr_system == "Speechmatics"
                             else ASRSystem.ASSEMBLYAI)  # Default case
            
            experiment_folder = os.path.join("outputsnew", asr_system, strategy, experiment_id)
            os.makedirs(experiment_folder, exist_ok=True)
            
            try:
                evaluate_with_example_pairs(
                    postprocess_function=postprocess_4_1_4,
                    experiment_folder_for_run=experiment_folder,
                    num_sentences=num_sentences,
                    strategy=strategy_enum,
                    asr_system=asr_system_enum
                )
                print(f"    Successfully processed all samples for {num_sentences} sentences")
                
                sample_result_path = os.path.join(experiment_folder, sample_id, f"{num_sentences}_sentences", "evaluation_metrics.csv")
                if os.path.exists(sample_result_path):
                    import pandas as pd
                    df = pd.read_csv(sample_result_path)
                    raw_wer = df[df['Type'] == 'Raw ASR']['WER'].iloc[0]
                    imp_wer = df[df['Type'] == 'Improved']['WER'].iloc[0]
                    wer_improvement = ((raw_wer - imp_wer) / raw_wer) * 100 if raw_wer > 0 else 0
                    print(f"    Sample {sample_id}: WER {raw_wer:.3f} -> {imp_wer:.3f} ({wer_improvement:+.1f}%)")
                
            except Exception as e:
                print(f"    Evaluation failed: {e}")
                continue
                
        except Exception as e:
            print(f"    Error processing {num_sentences} sentences: {e}")
            continue
        
        time.sleep(2)
    
    print(f"\nCompleted {asr_system} - {strategy} - Run {run_number}")
    return True

def run_all_experiments(asr_systems: list = None, strategies: list = None, num_runs: int = 5, sentences: list = None):
    """
    Run all experiments for the specified ASR systems and strategies.
    """
    if asr_systems is None:
        asr_systems = ["Whisper", "Gemini"]
    
    if strategies is None:
        strategies = [
            "data_driven",
            "exhaustive_phoneme",
            "random_error"
        ]
    
    print("MULTI-ASR EXPERIMENT PLAN")
    print("=" * 50)
    print(f"ASR Systems: {asr_systems}")
    print(f"Strategies: {strategies}")
    print(f"Runs per combination: {num_runs}")
    print(f"Sentence counts: {sentences if sentences else [2,4,6,8,10]}")
    print(f"Total experiments: {len(asr_systems)} × {len(strategies)} × {num_runs} = {len(asr_systems) * len(strategies) * num_runs}")
    
    available_systems = get_available_asr_systems()
    available_system_names = [system.value for system in available_systems]
    print(f"\nAvailable ASR systems: {available_system_names}")
    
    for asr_system in asr_systems:
        if asr_system not in available_system_names:
            print(f"{asr_system} not available. Ensure ASR files exist in data/asr_raw/{asr_system}/")
            continue
        
        print(f"\nStarting experiments for {asr_system}")
        print("=" * 40)
        
        for strategy in strategies:
            print(f"\nStrategy: {strategy}")
            
            for run_num in range(1, num_runs + 1):
                success = run_experiment(asr_system, strategy, run_num, sentences=sentences)
                
                if not success:
                    print(f"Failed: {asr_system} - {strategy} - Run {run_num}")
                else:
                    print(f"Completed: {asr_system} - {strategy} - Run {run_num}")
                
                if run_num < num_runs:
                    print("Waiting 5 seconds before next run...")
                    time.sleep(5)
            
            print("Waiting 10 seconds before next strategy...")
            time.sleep(10)
        
        print(f"\nCompleted all experiments for {asr_system}")

def main():
    parser = argparse.ArgumentParser(description="Run multi-ASR experiments")
    parser.add_argument('--asr-systems', nargs='+', default=["Whisper", "Gemini"],
                       help='ASR systems to test (default: Whisper Gemini)')
    parser.add_argument('--strategies', nargs='+',
                       default=["data_driven", "exhaustive_phoneme", "random_error"],
                       help='Strategies to test')
    parser.add_argument('--runs', type=int, default=5,
                       help='Number of runs per combination (default: 5)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be run without actually running')
    parser.add_argument('--single-strategy', type=str,
                       help='Run only a single strategy (e.g., data_driven)')
    parser.add_argument('--single-asr', type=str,
                       help='Run only a single ASR system (e.g., Whisper)')
    parser.add_argument('--sentences', nargs='+', type=int,
                       help='Specific sentence counts to process (e.g., 8 10)')
    
    args = parser.parse_args()
    
    # Handle single options
    if args.single_asr:
        asr_systems = [args.single_asr]
    else:
        asr_systems = args.asr_systems
    
    if args.single_strategy:
        strategies = [args.single_strategy]
    else:
        strategies = args.strategies
    
    print("MULTI-ASR EXPERIMENT RUNNER")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if args.dry_run:
        print("\nDRY RUN MODE - No actual experiments will be run")
        print(f"Would run experiments for:")
        print(f"  ASR Systems: {asr_systems}")
        print(f"  Strategies: {strategies}")
        print(f"  Runs: {args.runs}")
        print(f"  Sentences: {args.sentences if args.sentences else [2,4,6,8,10]}")
        print(f"  Total: {len(asr_systems) * len(strategies) * args.runs} experiments")
        return
    
    # Check dependencies
    try:
        import openai
        from dotenv import load_dotenv
        load_dotenv()
        
        if not os.getenv('AZURE_OPENAI_API_KEY'):
            print("AZURE_OPENAI_API_KEY not found in environment variables")
            print("Please set up your .env file with Azure OpenAI credentials")
            return
        
        print("Dependencies and API keys verified")
        
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install openai python-dotenv")
        return
    
    # Confirm before starting
    if not args.dry_run:
        total_experiments = len(asr_systems) * len(strategies) * args.runs
        print(f"\nAbout to run {total_experiments} experiments")
        print("This may take several hours and will consume API credits.")
        
        response = input("Continue? (y/N): ").lower().strip()
        if response != 'y':
            print("Cancelled.")
            return
    
    # Run experiments
    start_time = time.time()
    
    try:
        run_all_experiments(asr_systems, strategies, args.runs, args.sentences)
        
        elapsed_time = time.time() - start_time
        print("\nALL EXPERIMENTS COMPLETED!")
        print(f"Total time: {elapsed_time/3600:.1f} hours")
        print("Run analysis with: python wer_analysis.py")
        
    except KeyboardInterrupt:
        print("\nExperiments interrupted by user")
        elapsed_time = time.time() - start_time
        print(f"Partial completion time: {elapsed_time/60:.1f} minutes")
    except Exception as e:
        print(f"\nError during experiments: {e}")

if __name__ == "__main__":
    main() 