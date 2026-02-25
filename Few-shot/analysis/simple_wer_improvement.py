#!/usr/bin/env python3
"""
Simple WER Improvement Analysis

Shows relative WER improvement percentages for different strategies and sentence counts.
Clean, simple bar graph with numbers displayed.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

def collect_wer_improvements(outputs_dir='outputs'):
    """Collect WER improvement data from outputs directory."""
    print("🔍 Collecting WER improvement data...")
    
    data = defaultdict(lambda: defaultdict(list))
    
    if not os.path.exists(outputs_dir):
        print(f"❌ Outputs directory '{outputs_dir}' not found!")
        return data
    
    strategy_dirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
    
    for strategy_dir in strategy_dirs:
        strategy_path = os.path.join(outputs_dir, strategy_dir)
        
        # Parse strategy name (e.g., '414_data_driven_run1' -> 'Data Driven')
        strategy_name = parse_strategy_name(strategy_dir)
        
        # Get all samples
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
                            # Calculate relative WER improvement percentage
                            wer_improvement = ((raw_wer - imp_wer) / raw_wer) * 100
                            data[strategy_name][sentence_count].append(wer_improvement)
                            
                    except Exception as e:
                        print(f"⚠️ Error processing {eval_file}: {e}")
    
    print(f"✅ Found data for {len(data)} strategies")
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

def create_wer_improvement_graph(data):
    """Create a clean bar graph showing WER improvements."""
    print("📊 Creating WER Improvement Graph...")
    
    # Prepare data
    strategies = list(data.keys())
    all_sentence_counts = set()
    for strategy_data in data.values():
        all_sentence_counts.update(strategy_data.keys())
    sentence_counts = sorted(all_sentence_counts)
    
    if not strategies or not sentence_counts:
        print("❌ No data available")
        return
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(sentence_counts))
    width = 0.8 / len(strategies)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    
    for strategy_idx, strategy in enumerate(strategies):
        means = []
        stds = []
        
        for sent_count in sentence_counts:
            improvements = data[strategy][sent_count]
            if improvements:
                mean_imp = np.mean(improvements)
                std_imp = np.std(improvements)
                means.append(mean_imp)
                stds.append(std_imp)
            else:
                means.append(0)
                stds.append(0)
        
        # Create bars (removed error bars)
        bars = ax.bar(x + strategy_idx * width, means, width, 
                     label=strategy, alpha=0.8, color=colors[strategy_idx])
        
        # Add value labels on ALL bars
        for i, (bar, mean) in enumerate(zip(bars, means)):
            # Show numbers on all bars, even if mean is 0
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + 0.5, 
                   f'{mean:.1f}%', 
                   ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Number of Sentences', fontsize=14, fontweight='bold')
    ax.set_ylabel('Relative WER Improvement (%)', fontsize=14, fontweight='bold')
    ax.set_title('Relative WER Improvement by Strategy and Sentence Count', 
                fontsize=16, fontweight='bold', pad=20)
    
    ax.set_xticks(x + width * (len(strategies) - 1) / 2)
    ax.set_xticklabels(sentence_counts, fontsize=12)
    
    ax.legend(fontsize=11, loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    
    # Set y-axis to start from a reasonable minimum
    ax.set_ylim(bottom=-5)
    
    plt.tight_layout()
    plt.savefig('wer_improvement_simple.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("✅ Graph saved as 'wer_improvement_simple.png'")

def print_summary_table(data):
    """Print a simple summary table of the improvements."""
    print("\n" + "="*80)
    print("WER IMPROVEMENT SUMMARY")
    print("="*80)
    print(f"{'Strategy':<20} {'Sentences':<10} {'Samples':<8} {'Mean WER Imp (%)':<15} {'Std Dev':<10}")
    print("-"*80)
    
    for strategy in sorted(data.keys()):
        for sent_count in sorted(data[strategy].keys()):
            improvements = data[strategy][sent_count]
            if improvements:
                mean_imp = np.mean(improvements)
                std_imp = np.std(improvements)
                print(f"{strategy:<20} {sent_count:<10} {len(improvements):<8} {mean_imp:<15.2f} {std_imp:<10.2f}")
    
    print("="*80)

def main():
    """Main function to run the WER improvement analysis."""
    print("🚀 Simple WER Improvement Analysis")
    print("="*50)
    
    # Collect data
    data = collect_wer_improvements()
    
    if not data:
        print("❌ No WER improvement data found. Please check your outputs directory.")
        return
    
    # Create visualization and summary
    create_wer_improvement_graph(data)
    print_summary_table(data)
    
    print("\n🎉 Analysis Complete!")

if __name__ == "__main__":
    main() 