import os
import pandas as pd
import numpy as np
from pathlib import Path

# Define the base directory for ASR outputs
OUTPUT_DIR = Path('outputsnew')
RESULTS_DIR = Path('very_last_csvs')

def create_results_directory():
    """Create the results directory if it doesn't exist"""
    if not RESULTS_DIR.exists():
        RESULTS_DIR.mkdir()
        print(f"Created directory: {RESULTS_DIR}")

def get_wer_stats_from_dir(asr_dir):
    """
    Get WER statistics from an ASR directory's data_driven strategy
    Returns a dictionary with min, max, and quartile values
    """
    wer_values = []
    
    # Look specifically for data_driven directory
    data_driven_dir = asr_dir / 'data_driven'
    if not data_driven_dir.exists():
        return None
        
    # Check each run directory
    for run_dir in data_driven_dir.iterdir():
        if run_dir.is_dir() and run_dir.name.endswith('run1'):  # Only use run1 for WER stats
            # Check each audio file directory
            for audio_dir in run_dir.iterdir():
                if audio_dir.is_dir():
                    # Check each n_sentences directory
                    for n_sent_dir in audio_dir.iterdir():
                        if n_sent_dir.is_dir():
                            metrics_file = n_sent_dir / 'evaluation_metrics.csv'
                            if metrics_file.exists():
                                try:
                                    df = pd.read_csv(metrics_file)
                                    # Get the Raw ASR WER
                                    raw_wer = df[df['Type'] == 'Raw ASR']['WER'].values
                                    if len(raw_wer) > 0:
                                        wer_values.append(raw_wer[0])
                                except Exception as e:
                                    print(f"Error reading {metrics_file}: {e}")
    
    if not wer_values:
        return None
        
    wer_values = np.array(wer_values)
    return {
        'min': np.min(wer_values),
        'q1': np.percentile(wer_values, 25),
        'median': np.median(wer_values),
        'q3': np.percentile(wer_values, 75),
        'max': np.max(wer_values),
        'mean': np.mean(wer_values)
    }

def calculate_rwer(raw_wer, improved_wer):
    """Helper function to calculate RWER percentage"""
    return (raw_wer - improved_wer) / raw_wer * 100 if raw_wer > 0 else 0

def process_asr_results(selected_asrs):
    """
    Process results for selected ASRs and average multiple runs if they exist
    Only processes data_driven strategy, includes both Raw ASR and Improved metrics
    """
    results = []
    
    for asr_name, asr_path in selected_asrs:
        asr_results = {'ASR': asr_name}
        data_driven_dir = asr_path / 'data_driven'
        
        if not data_driven_dir.exists():
            continue
            
        # Initialize metrics for both Raw ASR and Improved, per number of sentences
        sentence_counts = ['2', '4', '6', '8', '10']
        metrics = {
            'run1': {'Raw': {n: {'WER': []} for n in sentence_counts},
                    'Improved': {n: {'WER': []} for n in sentence_counts}},
            'run2': {'Raw': {n: {'WER': []} for n in sentence_counts},
                    'Improved': {n: {'WER': []} for n in sentence_counts}}
        }
        
        # Process each run
        for run_dir in data_driven_dir.iterdir():
            if run_dir.is_dir() and ('run1' in run_dir.name or 'run2' in run_dir.name):
                run_name = 'run1' if 'run1' in run_dir.name else 'run2'
                
                # Process each audio file
                for audio_dir in run_dir.iterdir():
                    if audio_dir.is_dir():
                        # Process each n_sentences directory
                        for n_sent_dir in audio_dir.iterdir():
                            if n_sent_dir.is_dir():
                                # Get number of sentences from directory name
                                n_sent = n_sent_dir.name.split('_')[0]
                                if n_sent not in sentence_counts:
                                    continue
                                    
                                metrics_file = n_sent_dir / 'evaluation_metrics.csv'
                                if metrics_file.exists():
                                    try:
                                        df = pd.read_csv(metrics_file)
                                        # Get both Raw ASR and Improved metrics
                                        for metric_type in ['Raw ASR', 'Improved']:
                                            type_key = 'Raw' if metric_type == 'Raw ASR' else 'Improved'
                                            row = df[df['Type'] == metric_type].iloc[0]
                                            metrics[run_name][type_key][n_sent]['WER'].append(row['WER'])
                                    except Exception as e:
                                        print(f"Error processing {metrics_file}: {e}")
        
        # Calculate combined averages for Raw and Improved across both runs, per number of sentences
        for type_key in ['Raw', 'Improved']:
            for n_sent in sentence_counts:
                # Combine metrics from both runs
                combined_wer = (metrics['run1'][type_key][n_sent]['WER'] + 
                              metrics['run2'][type_key][n_sent]['WER'])
                
                if combined_wer:  # If we have any metrics
                    if type_key == 'Raw':
                        raw_wer = np.mean(combined_wer)
                        asr_results[f"{n_sent}sent_Raw_WER"] = raw_wer
                    else:  # Improved
                        improved_wer = np.mean(combined_wer)
                        asr_results[f"{n_sent}sent_Improved_WER"] = improved_wer
                        # Calculate RWER using the raw WER we stored earlier
                        asr_results[f"{n_sent}sent_RWER"] = calculate_rwer(
                            asr_results[f"{n_sent}sent_Raw_WER"], improved_wer)
        
        results.append(asr_results)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    
    # Add header comment to indicate this is for data_driven strategy
    output_file = RESULTS_DIR / 'byasr.csv'
    with open(output_file, 'w') as f:
        f.write("# Results for data_driven strategy only\n")
        df.to_csv(f, index=False)
    print(f"Results saved to {output_file}")

def process_by_strategy(selected_asrs):
    """
    Process results for selected ASRs grouped by example picking strategy
    Only processes 6-sentence examples, includes both Raw ASR and Improved results
    """
    results = []
    strategies = ['data_driven', 'exhaustive_phoneme', 'random_error']
    
    for asr_name, asr_path in selected_asrs:
        asr_results = {'ASR': asr_name}
        
        # Process each strategy
        for strategy in strategies:
            strategy_dir = asr_path / strategy
            if not strategy_dir.exists():
                continue
                
            metrics = {'Raw': {'WER': []}, 'Improved': {'WER': []}}
            
            # Process each run
            for run_dir in strategy_dir.iterdir():
                if run_dir.is_dir() and ('run1' in run_dir.name or 'run2' in run_dir.name):
                    # Process each audio file
                    for audio_dir in run_dir.iterdir():
                        if audio_dir.is_dir():
                            # Look specifically for 6_sentences directory
                            sent_dir = audio_dir / '6_sentences'
                            if sent_dir.exists():
                                metrics_file = sent_dir / 'evaluation_metrics.csv'
                                if metrics_file.exists():
                                    try:
                                        df = pd.read_csv(metrics_file)
                                        # Get both Raw ASR and Improved metrics
                                        raw_wer = df[df['Type'] == 'Raw ASR']['WER'].values
                                        improved_wer = df[df['Type'] == 'Improved']['WER'].values
                                        if len(raw_wer) > 0:
                                            metrics['Raw']['WER'].append(raw_wer[0])
                                        if len(improved_wer) > 0:
                                            metrics['Improved']['WER'].append(improved_wer[0])
                                    except Exception as e:
                                        print(f"Error processing {metrics_file}: {e}")
            
            # Calculate average WER and RWER for this strategy
            if metrics['Raw']['WER']:
                raw_wer = np.mean(metrics['Raw']['WER'])
                improved_wer = np.mean(metrics['Improved']['WER'])
                asr_results[f"{strategy}_Raw_WER"] = raw_wer
                asr_results[f"{strategy}_Improved_WER"] = improved_wer
                asr_results[f"{strategy}_RWER"] = calculate_rwer(raw_wer, improved_wer)
        
        results.append(asr_results)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    output_file = RESULTS_DIR / 'byexamples.csv'
    with open(output_file, 'w') as f:
        f.write("# Results for 6-sentence examples, Raw ASR and Improved, grouped by strategy (data_driven, exhaustive_phoneme, random_error)\n")
        df.to_csv(f, index=False)
    print(f"Strategy results saved to {output_file}")

def process_rwer_all_asrs():
    """
    Process RWER for all ASRs using data_driven strategy and 6-sentence examples
    """
    results = []
    asr_dirs = [d for d in OUTPUT_DIR.iterdir() if d.is_dir()]
    
    for asr_dir in asr_dirs:
        asr_results = {'ASR': asr_dir.name}
        data_driven_dir = asr_dir / 'data_driven'
        
        if not data_driven_dir.exists():
            continue
            
        metrics = {'Raw': [], 'Improved': []}
        
        # Process each run
        for run_dir in data_driven_dir.iterdir():
            if run_dir.is_dir() and ('run1' in run_dir.name or 'run2' in run_dir.name):
                # Process each audio file
                for audio_dir in run_dir.iterdir():
                    if audio_dir.is_dir():
                        # Look specifically for 6_sentences directory
                        sent_dir = audio_dir / '6_sentences'
                        if sent_dir.exists():
                            metrics_file = sent_dir / 'evaluation_metrics.csv'
                            if metrics_file.exists():
                                try:
                                    df = pd.read_csv(metrics_file)
                                    raw_wer = df[df['Type'] == 'Raw ASR']['WER'].values[0]
                                    improved_wer = df[df['Type'] == 'Improved']['WER'].values[0]
                                    metrics['Raw'].append(raw_wer)
                                    metrics['Improved'].append(improved_wer)
                                except Exception as e:
                                    print(f"Error processing {metrics_file}: {e}")
        
        # Calculate averages and RWER
        if metrics['Raw'] and metrics['Improved']:
            raw_wer = np.mean(metrics['Raw'])
            improved_wer = np.mean(metrics['Improved'])
            rwer = (raw_wer - improved_wer) / raw_wer * 100  # RWER as percentage
            
            asr_results['Raw_WER'] = raw_wer
            asr_results['Improved_WER'] = improved_wer
            asr_results['RWER'] = rwer
            
            results.append(asr_results)
    
    # Convert to DataFrame and save
    df = pd.DataFrame(results)
    # Sort by Raw WER for better readability
    df = df.sort_values('Raw_WER')
    output_file = RESULTS_DIR / 'byrwer.csv'
    with open(output_file, 'w') as f:
        f.write("# Results for all ASRs, data_driven strategy, 6-sentence examples\n")
        df.to_csv(f, index=False)
    print(f"RWER results saved to {output_file}")

def main():
    # Create results directory
    create_results_directory()
    
    # Get list of all ASR directories
    asr_dirs = [d for d in OUTPUT_DIR.iterdir() if d.is_dir()]
    
    # Calculate WER stats for each ASR (data_driven strategy only)
    print("Calculating WER statistics for each ASR (data_driven strategy)...")
    wer_stats = {}
    for asr_dir in asr_dirs:
        stats = get_wer_stats_from_dir(asr_dir)
        if stats:
            wer_stats[asr_dir.name] = stats
            print(f"{asr_dir.name}:")
            print(f"  Min: {stats['min']:.2f}")
            print(f"  Q1: {stats['q1']:.2f}")
            print(f"  Median: {stats['median']:.2f}")
            print(f"  Q3: {stats['q3']:.2f}")
            print(f"  Max: {stats['max']:.2f}")
            print(f"  Mean: {stats['mean']:.2f}")
    
    # Get all ASRs sorted by mean WER
    all_asrs = sorted(
        [(name, stats['mean']) for name, stats in wer_stats.items()],
        key=lambda x: x[1]
    )
    
    n_asrs = len(all_asrs)
    if n_asrs < 5:
        print("Warning: Found fewer than 5 ASRs with valid WER statistics")
        selected_indices = range(n_asrs)
    else:
        # Select ASRs at specific percentiles (0%, 25%, 50%, 75%, 100%)
        selected_indices = [
            0,                          # min (0th percentile)
            n_asrs // 4,               # Q1 (25th percentile)
            n_asrs // 2,               # median (50th percentile)
            3 * n_asrs // 4,           # Q3 (75th percentile)
            n_asrs - 1                 # max (100th percentile)
        ]
    
    # Create list of selected ASRs with their paths
    selected_asrs = [(all_asrs[i][0], OUTPUT_DIR / all_asrs[i][0]) for i in selected_indices]
    
    print("\nSelected ASRs (ordered by WER from best to worst):")
    for i, (asr_name, _) in enumerate(selected_asrs):
        position = ["Best (min)", "Q1", "Median", "Q3", "Worst (max)"][i]
        print(f"- {position}: {asr_name} (WER: {wer_stats[asr_name]['mean']:.2%})")
    
    
    # Process results for data_driven strategy
    print("\nProcessing data_driven results for selected ASRs...")
    process_asr_results(selected_asrs)
    
    # Process results by example picking strategy
    print("\nProcessing results by example picking strategy...")
    process_by_strategy(selected_asrs)
    
    # Process RWER for all ASRs
    print("\nProcessing RWER for all ASRs...")
    process_rwer_all_asrs()
    print("Analysis complete!")

if __name__ == "__main__":
    main()
