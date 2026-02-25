#!/usr/bin/env python3
"""
Raw vs Improved ASR Comparison

Creates a bar graph comparing raw and improved ASR performance with values displayed above bars.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def collect_asr_data(outputs_dir='outputs'):
    """Collect raw and improved ASR data."""
    print("Collecting ASR performance data...")
    
    data = defaultdict(lambda: defaultdict(list))
    
    if not os.path.exists(outputs_dir):
        print(f"Outputs directory '{outputs_dir}' not found!")
        return data
    
    strategy_dirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
    
    for strategy_dir in strategy_dirs:
        strategy_path = os.path.join(outputs_dir, strategy_dir)
        strategy_name = parse_strategy_name(strategy_dir)
        
        samples = [s for s in os.listdir(strategy_path) if os.path.isdir(os.path.join(strategy_path, s))]
        
        for sample in samples:
            sample_path = os.path.join(strategy_path, sample)
            sentence_dirs = [d for d in os.listdir(sample_path) 
                           if os.path.isdir(os.path.join(sample_path, d)) and '_sentences' in d]
            
            for sent_dir in sentence_dirs:
                sent_path = os.path.join(sample_path, sent_dir)
                eval_file = os.path.join(sent_path, 'evaluation_metrics.csv')
                
                if os.path.exists(eval_file):
                    try:
                        df = pd.read_csv(eval_file)
                        sentence_count = int(sent_dir.split('_')[0])
                        
                        # Get Raw ASR and Improved WER values
                        raw_wer = df[df['Type'] == 'Raw ASR']['WER'].values[0] if len(df[df['Type'] == 'Raw ASR']) > 0 else None
                        imp_wer = df[df['Type'] == 'Improved']['WER'].values[0] if len(df[df['Type'] == 'Improved']) > 0 else None
                        
                        if raw_wer is not None and imp_wer is not None:
                            data[strategy_name]['raw'].append(raw_wer * 100)  # Convert to percentage
                            data[strategy_name]['improved'].append(imp_wer * 100)  # Convert to percentage
                            
                    except Exception as e:
                        print(f"Error processing {eval_file}: {e}")
    
    print(f"Found data for {len(data)} strategies")
    return data

def parse_strategy_name(strategy_dir):
    """Parse strategy directory name to extract clean strategy name."""
    parts = strategy_dir.split('_')
    if len(parts) >= 3:
        strategy_parts = []
        for part in parts[1:]:
            if not part.startswith('run'):
                strategy_parts.append(part)
            else:
                break
        strategy_name = ' '.join(strategy_parts).title()
        return strategy_name
    return strategy_dir

def create_comparison_graph(data):
    """Create a bar graph comparing raw and improved ASR performance."""
    print("Creating comparison graph...")
    
    if not data:
        print("No data available")
        return
    
    # Prepare data for plotting
    strategies = list(data.keys())
    n_strategies = len(strategies)
    
    # Calculate means for raw and improved
    raw_means = []
    imp_means = []
    raw_stds = []
    imp_stds = []
    
    for strategy in strategies:
        raw_means.append(np.mean(data[strategy]['raw']))
        imp_means.append(np.mean(data[strategy]['improved']))
        raw_stds.append(np.std(data[strategy]['raw']))
        imp_stds.append(np.std(data[strategy]['improved']))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Set the width of bars and positions of the bars
    width = 0.35
    x = np.arange(n_strategies)
    
    # Create bars
    raw_bars = ax.bar(x - width/2, raw_means, width, label='Raw ASR', color='lightcoral', alpha=0.8)
    imp_bars = ax.bar(x + width/2, imp_means, width, label='Improved', color='lightgreen', alpha=0.8)
    
    # Add value labels above bars
    def autolabel(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                   f'{val:.1f}%',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    autolabel(raw_bars, raw_means)
    autolabel(imp_bars, imp_means)
    
    # Customize the plot
    ax.set_ylabel('Word Error Rate (%)', fontsize=12, fontweight='bold')
    ax.set_title('Raw vs Improved ASR Performance by Strategy', fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=45, ha='right')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('raw_vs_improved_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Graph saved as 'raw_vs_improved_comparison.png'")

def print_summary_table(data):
    """Print a summary table of the performance."""
    print("\n" + "="*80)
    print("ASR PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'Strategy':<20} {'Raw WER (%)':<15} {'Improved WER (%)':<15} {'Improvement (%)':<15}")
    print("-"*80)
    
    for strategy in sorted(data.keys()):
        raw_mean = np.mean(data[strategy]['raw'])
        imp_mean = np.mean(data[strategy]['improved'])
        improvement = ((raw_mean - imp_mean) / raw_mean) * 100
        
        print(f"{strategy:<20} {raw_mean:<15.2f} {imp_mean:<15.2f} {improvement:<15.2f}")
    
    print("="*80)

def main():
    """Main function to run the ASR performance analysis."""
    print("Raw vs Improved ASR Performance Analysis")
    print("="*50)
    
    # Collect data
    data = collect_asr_data()
    
    if not data:
        print("No ASR performance data found. Please check your outputs directory.")
        return
    
    # Create visualization and summary
    create_comparison_graph(data)
    print_summary_table(data)
    
    print("\nAnalysis Complete!")

if __name__ == "__main__":
    main()
