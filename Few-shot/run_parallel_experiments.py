#!/usr/bin/env python3
"""
Parallel ASR Experiment Runner
Runs multiple ASR systems and techniques in parallel to speed up processing.
"""

import os
import subprocess
from multiprocessing import Pool
from itertools import product

# ASR Systems
ASR_SYSTEMS = [
    'AWS', 'AssemblyAI', 'Azure', 'Deepgram', 'ElevenLabs',
    'GCP', 'Gemini', 'Gladia', 'Speechmatics', 'Whisper'
]

TECHNIQUES = [
    "data_driven",
    "exhaustive_phoneme",
    "random_error",
]

def run_experiment(args):
    """Run a single experiment with given ASR system and technique."""
    asr_system, technique = args
    
    # Create command
    cmd = [
        'python', 'run_experiments.py',
        '--asr-system', asr_system,
        '--strategy', technique,
        '--model', 'gpt4',
        '--runs', '1'  # You can adjust number of runs if needed
    ]
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join('outputsnew', asr_system, technique)
    os.makedirs(output_dir, exist_ok=True)
    
    # Create log file
    log_file = os.path.join(output_dir, f'experiment_log.txt')
    
    print(f"Starting experiment: {asr_system} - {technique}")
    
    try:
        # Run the experiment and capture output
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
            process.wait()
            
        if process.returncode == 0:
            print(f"Completed: {asr_system} - {technique}")
        else:
            print(f"Failed: {asr_system} - {technique}")
            
    except Exception as e:
        print(f"Error running {asr_system} - {technique}: {str(e)}")
        with open(log_file, 'a') as f:
            f.write(f"\nError: {str(e)}")

def main():
    # Generate all combinations of ASR systems and techniques
    experiments = list(product(ASR_SYSTEMS, TECHNIQUES))
    
    print(f"Starting {len(experiments)} experiments in parallel...")
    
    # Create a pool of workers
    # Adjust processes based on your CPU cores, memory, and system capabilities
    with Pool(processes=8) as pool:
        pool.map(run_experiment, experiments)
    
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main()
