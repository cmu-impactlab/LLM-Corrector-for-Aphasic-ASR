#!/usr/bin/env python3
"""
Advanced Experiment Runner for ASR Post-Processing

This script can run experiments across different:
- ASR systems (Azure, AssemblyAI, Whisper, Gemini)
- Strategies (data_driven, exhaustive_phoneme, random_error)
- Models (GPT-4.1)

Examples:
    # Run with Azure using data_driven strategy
    python run_experiments.py --asr-system "Azure" --strategy data_driven --model gpt4 --runs 3

    # Run with all available ASR systems
    python run_experiments.py --strategy exhaustive_phoneme --model gpt4 --runs 1

    # Run specific sentence counts
    python run_experiments.py --asr-system Whisper --strategy random_error --sentences 2 4 6 8 10
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Optional

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.file_utils import (
    ASRSystem, get_available_asr_systems, get_experiment_output_path,
    create_or_replace_experiment_folder
)
from utils.example_generator import ExampleStrategy
from evaluators.evaluator import evaluate_with_example_pairs

try:
    from models.azure_gpt import postprocess_4_1_4
    print("Azure GPT 4_1_4 model loaded successfully")
    gpt4_available = True
except ImportError as e:
    print(f"Warning: Could not import Azure GPT 4_1_4 model: {e}")
    postprocess_4_1_4 = None
    gpt4_available = False


def get_model_function(model_name: str):
    """Get the postprocessing function for the specified model."""
    model_map = {
        "gpt4": postprocess_4_1_4,
        "azure_gpt": postprocess_4_1_4,
        "gpt4_4": postprocess_4_1_4,
    }
    
    # Check if model name exists in map
    if model_name.lower() not in model_map:
        available = [name for name, func in model_map.items() if func is not None]
        raise ValueError(f"Unknown model '{model_name}'. Available: {available}")
    
    # Check if model function is actually available (not None)
    model_func = model_map[model_name.lower()]
    if model_func is None:
        available = [name for name, func in model_map.items() if func is not None]
        unavailable = [name for name, func in model_map.items() if func is None]
        
        error_msg = f"Model '{model_name}' is not available due to import errors."
        if available:
            error_msg += f"\nAvailable models: {available}"
        if unavailable:
            error_msg += f"\nUnavailable models (import failed): {unavailable}"
        error_msg += "\n\nPlease check:"
        error_msg += "\n  1. models/azure_gpt.py exists and has postprocess_4_1_4 function"
        error_msg += "\n  2. Required dependencies are installed:"
        error_msg += "\n     pip install openai nltk"
        error_msg += "\n  3. Environment variables are set (.env file with Azure OpenAI credentials)"
        
        raise ValueError(error_msg)
    
    return model_func


def run_single_experiment(
    asr_system: ASRSystem,
    strategy: ExampleStrategy, 
    model_name: str,
    run_id: int,
    sentence_counts: List[int]
):
    """Run a single experiment configuration."""
    
    print(f"\n{'='*60}")
    print("Starting Experiment")
    print(f"   ASR System: {asr_system.value}")
    print(f"   Strategy: {strategy.value.replace('_', ' ').title()}")
    print(f"   Model: {model_name}")
    print(f"   Run: {run_id}")
    print(f"   Sentence counts: {sentence_counts}")
    print(f"{'='*60}")
    
    # Get model function
    try:
        postprocess_function = get_model_function(model_name)
    except ValueError as e:
        print(f"{e}")
        return False
    
    # Create experiment folder with timestamp
    timestamp = datetime.now().strftime("%y%m%d_%H%M")
    exp_id = f"{timestamp}_{strategy.value}_run{run_id}"
    
    # Create output path: outputs/[ASR_System]/[strategy]/[exp_id]
    experiment_folder = get_experiment_output_path(exp_id, asr_system, strategy.value)
    os.makedirs(experiment_folder, exist_ok=True)
    
    print(f"Output folder: {experiment_folder}")
    
    # Run evaluation for each sentence count
    for num_sentences in sentence_counts:
        print(f"\nRunning with {num_sentences} example sentences...")
        
        try:
            evaluate_with_example_pairs(
                postprocess_function=postprocess_function,
                experiment_folder_for_run=experiment_folder,
                num_sentences=num_sentences,
                strategy=strategy,
                asr_system=asr_system
            )
            print(f"Completed {num_sentences} sentences")
            
        except Exception as e:
            print(f"Failed {num_sentences} sentences: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\nExperiment completed: {exp_id}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Run ASR post-processing experiments across different systems and strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --asr-system "Azure" --strategy data_driven --model gpt4 --runs 3
  %(prog)s --strategy exhaustive_phoneme --model gpt4 --runs 1 --sentences 2 4 6
  %(prog)s --asr-system Whisper --strategy random_error --model gpt4
        """
    )
    
    # ASR System selection
    available_systems = [s.value for s in ASRSystem]
    parser.add_argument(
        '--asr-system', '--asr',
        choices=available_systems,
        help='ASR system to use. If not specified, uses all available systems'
    )
    
    # Strategy selection
    available_strategies = [s.value for s in ExampleStrategy]
    parser.add_argument(
        '--strategy', '-s',
        choices=available_strategies,
        default='data_driven',
        help='Example selection strategy (default: data_driven)'
    )
    
    # Model selection
    parser.add_argument(
        "--model",
        "-m",
        choices=["gpt4", "azure_gpt", "gpt4_4"],
        default="gpt4",
        help="Model to use for post-processing (default: gpt4=gpt4_4).",
    )
    
    # Number of runs
    parser.add_argument(
        '--runs', '-r',
        type=int,
        default=1,
        help='Number of runs to execute (default: 1)'
    )
    
    # Sentence counts
    parser.add_argument(
        '--sentences',
        type=int,
        nargs='+',
        default=[2, 4, 6, 8, 10],
        help='Number of example sentences to test (default: 2 4 6 8 10)'
    )
    
    # List available systems
    parser.add_argument(
        '--list-systems',
        action='store_true',
        help='List available ASR systems and exit'
    )
    
    args = parser.parse_args()
    
    # Handle list systems
    if args.list_systems:
        print("Available ASR Systems:")
        available = get_available_asr_systems()
        for system in available:
            print(f"  • {system.value}")
        if not available:
            print("  No ASR systems found with data")
        return
    
    # Determine ASR systems to use
    if args.asr_system:
        # Find the specified system
        asr_systems = []
        for system in ASRSystem:
            if system.value == args.asr_system:
                asr_systems = [system]
                break
        if not asr_systems:
            print(f"ASR system '{args.asr_system}' not found")
            return
    else:
        # Use all available systems
        asr_systems = get_available_asr_systems()
        if not asr_systems:
            print("No ASR systems found with data. Please check your data/asr_raw/ directory.")
            return
    
    # Get strategy enum
    strategy = None
    for s in ExampleStrategy:
        if s.value == args.strategy:
            strategy = s
            break
    
    if not strategy:
        print(f"Strategy '{args.strategy}' not found")
        return
    
    # Print configuration
    print("Experiment Configuration:")
    print(f"   ASR Systems: {[s.value for s in asr_systems]}")
    print(f"   Strategy: {strategy.value.replace('_', ' ').title()}")
    print(f"   Model: {args.model}")
    print(f"   Runs: {args.runs}")
    print(f"   Sentence counts: {args.sentences}")
    print()
    
    # Confirm with user
    total_experiments = len(asr_systems) * args.runs
    response = input(f"This will run {total_experiments} experiment(s). Continue? (y/N): ")
    if response.lower() not in ['y', 'yes']:
        print("Cancelled by user")
        return
    
    # Run experiments
    successful_experiments = 0
    total_experiments = 0
    
    for asr_system in asr_systems:
        for run_id in range(1, args.runs + 1):
            total_experiments += 1
            
            success = run_single_experiment(
                asr_system=asr_system,
                strategy=strategy,
                model_name=args.model,
                run_id=run_id,
                sentence_counts=args.sentences
            )
            
            if success:
                successful_experiments += 1
    
    # Final summary
    print(f"\n{'='*60}")
    print("Experiment Suite Completed")
    print(f"   Successful: {successful_experiments}/{total_experiments}")
    print(f"   Failed: {total_experiments - successful_experiments}/{total_experiments}")
    
    if successful_experiments > 0:
        print("\nResults can be found in the outputs directory:")
        for asr_system in asr_systems:
            output_path = os.path.join("outputs", asr_system.value, strategy.value)
            if os.path.exists(output_path):
                print(f"   {output_path}")
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main() 