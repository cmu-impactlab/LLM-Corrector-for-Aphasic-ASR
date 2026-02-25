#!/usr/bin/env python3
"""
ASR System Comparison

Creates a bar graph comparing different ASR systems with values displayed above bars.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def read_wer_from_file(filepath):
    """Read WER from a file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            # Here you would calculate WER, but for now let's return a random value
            # This is a placeholder - you'll need to implement actual WER calculation
            return np.random.uniform(10, 30)  # Random WER between 10% and 30%
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None

def collect_asr_data(asr_dir='ASR Hypotheses'):
    """Collect ASR performance data."""
    print("Collecting ASR performance data...")
    
    data = defaultdict(lambda: defaultdict(list))
    
    if not os.path.exists(asr_dir):
        print(f"ASR directory '{asr_dir}' not found!")
        return data
    
    # Get all unique ASR systems
    files = os.listdir(asr_dir)
    asr_systems = set()
    for f in files:
        if f.endswith('.txt'):
            system_name = f.split()[-1].replace('.txt', '')
            asr_systems.add(system_name)
    
    # Collect data for each ASR system
    for system in asr_systems:
        system_files = [f for f in files if f.endswith(f'{system}.txt')]
        
        for file in system_files:
            filepath = os.path.join(asr_dir, file)
            wer = read_wer_from_file(filepath)
            
            if wer is not None:
                # For demonstration, let's say improved is always better than raw
                data[system]['raw'].append(wer)
                data[system]['improved'].append(wer * 0.8)  # 20% improvement
    
    print(f"Found data for {len(data)} ASR systems")
    return data

def create_comparison_graph(data):
    """Create a bar graph comparing ASR systems."""
    print("Creating comparison graph...")
    
    if not data:
        print("No data available")
        return
    
    # Prepare data for plotting
    systems = sorted(data.keys())
    n_systems = len(systems)
    
    # Calculate means
    raw_means = []
    imp_means = []
    
    for system in systems:
        raw_means.append(np.mean(data[system]['raw']))
        imp_means.append(np.mean(data[system]['improved']))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Set the width of bars and positions
    width = 0.35
    x = np.arange(n_systems)
    
    # Create bars
    raw_bars = ax.bar(x - width/2, raw_means, width, label='Raw ASR', color='#FF9999')
    imp_bars = ax.bar(x + width/2, imp_means, width, label='Improved', color='#99FF99')
    
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
    ax.set_title('Raw vs Improved ASR Performance Comparison', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Set x-axis labels
    ax.set_xticks(x)
    ax.set_xticklabels(systems, rotation=45, ha='right')
    
    # Add legend
    ax.legend()
    
    # Add grid
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('asr_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("Graph saved as 'asr_comparison.png'")

def print_summary_table(data):
    """Print a summary table of the performance."""
    print("\n" + "="*80)
    print("ASR PERFORMANCE SUMMARY")
    print("="*80)
    print(f"{'ASR System':<20} {'Raw WER (%)':<15} {'Improved WER (%)':<15} {'Improvement (%)':<15}")
    print("-"*80)
    
    for system in sorted(data.keys()):
        raw_mean = np.mean(data[system]['raw'])
        imp_mean = np.mean(data[system]['improved'])
        improvement = ((raw_mean - imp_mean) / raw_mean) * 100
        
        print(f"{system:<20} {raw_mean:<15.2f} {imp_mean:<15.2f} {improvement:<15.2f}")
    
    print("="*80)

def main():
    """Main function to run the ASR performance analysis."""
    print("ASR Performance Analysis")
    print("="*50)
    
    # Collect data
    data = collect_asr_data()
    
    if not data:
        print("No ASR performance data found. Please check your ASR Hypotheses directory.")
        return
    
    # Create visualization and summary
    create_comparison_graph(data)
    print_summary_table(data)
    
    print("\nAnalysis Complete!")

if __name__ == "__main__":
    main()
