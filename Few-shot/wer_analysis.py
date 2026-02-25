#!/usr/bin/env python3
"""
WER Analysis - Focus on raw WER scores

Shows raw WER values across:
- ASR Systems
- Strategies 
- Sentence counts
- Raw ASR vs Improved
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.file_utils import list_output_experiments

def load_wer_data():
    """Load all WER data from experiments."""
    all_data = []
    
    experiments = list_output_experiments()
    
    for system_name, exp_list in experiments.items():
        for exp_info in exp_list:
            exp_strategy = exp_info['strategy']
            exp_name = exp_info['experiment']
            exp_path = exp_info['path']
            
            if not os.path.exists(exp_path):
                continue
                
            for sample_dir in os.listdir(exp_path):
                sample_path = os.path.join(exp_path, sample_dir)
                if not os.path.isdir(sample_path):
                    continue
                
                for sentence_dir in os.listdir(sample_path):
                    sentence_path = os.path.join(sample_path, sentence_dir)
                    if not os.path.isdir(sentence_path) or not sentence_dir.endswith('_sentences'):
                        continue
                    
                    try:
                        num_sentences = int(sentence_dir.split('_')[0])
                    except ValueError:
                        continue
                    
                    metrics_file = os.path.join(sentence_path, 'evaluation_metrics.csv')
                    if not os.path.exists(metrics_file):
                        continue
                    
                    try:
                        df_metrics = pd.read_csv(metrics_file)
                        
                        for _, row in df_metrics.iterrows():
                            all_data.append({
                                'asr_system': system_name,
                                'strategy': exp_strategy,
                                'experiment': exp_name,
                                'sample': sample_dir,
                                'num_sentences': num_sentences,
                                'type': row['Type'],
                                'wer': float(row['WER']),
                                'cer': float(row['CER']),
                                'sim': float(row['SIM'])
                            })
                        
                    except Exception as e:
                        print(f"⚠️  Error reading {metrics_file}: {e}")
                        continue
    
    return pd.DataFrame(all_data)

def print_wer_summary(df):
    """Print detailed WER summary."""
    print("\n" + "="*80)
    print("📊 RAW WER ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n📈 Dataset Overview:")
    print(f"   • Total data points: {len(df)}")
    print(f"   • ASR systems: {list(df['asr_system'].unique())}")
    print(f"   • Strategies: {list(df['strategy'].unique())}")
    print(f"   • Samples: {list(df['sample'].unique())}")
    print(f"   • Sentence counts: {sorted(df['num_sentences'].unique())}")
    
    # Raw ASR vs Improved comparison
    print(f"\n🎙️ RAW ASR vs IMPROVED WER:")
    for result_type in ['Raw ASR', 'Improved']:
        type_data = df[df['type'] == result_type]
        if not type_data.empty:
            avg_wer = type_data['wer'].mean()
            min_wer = type_data['wer'].min()
            max_wer = type_data['wer'].max()
            std_wer = type_data['wer'].std()
            print(f"   • {result_type}:")
            print(f"     - Average WER: {avg_wer:.3f}")
            print(f"     - Min WER: {min_wer:.3f}")
            print(f"     - Max WER: {max_wer:.3f}")
            print(f"     - Std Dev: {std_wer:.3f}")
    
    # WER by ASR System
    print(f"\n🎤 WER by ASR System:")
    for asr_system in df['asr_system'].unique():
        system_data = df[df['asr_system'] == asr_system]
        print(f"   • {asr_system}:")
        
        for result_type in ['Raw ASR', 'Improved']:
            type_data = system_data[system_data['type'] == result_type]
            if not type_data.empty:
                avg_wer = type_data['wer'].mean()
                print(f"     - {result_type}: {avg_wer:.3f}")
    
    # WER by Strategy (Improved only)
    print(f"\n🎯 WER by Strategy (Improved results only):")
    improved_data = df[df['type'] == 'Improved']
    for strategy in sorted(improved_data['strategy'].unique()):
        strategy_data = improved_data[improved_data['strategy'] == strategy]
        avg_wer = strategy_data['wer'].mean()
        count = len(strategy_data)
        print(f"   • {strategy.replace('_', ' ').title()}: {avg_wer:.3f} (n={count})")
    
    # WER by Number of Sentences (Improved only)
    print(f"\n📝 WER by Number of Example Sentences (Improved results only):")
    for num_sent in sorted(improved_data['num_sentences'].unique()):
        sent_data = improved_data[improved_data['num_sentences'] == num_sent]
        avg_wer = sent_data['wer'].mean()
        count = len(sent_data)
        print(f"   • {num_sent} sentences: {avg_wer:.3f} (n={count})")
    
    # Best and Worst Results
    print(f"\n🏆 BEST RESULTS (Improved only):")
    best_result = improved_data.loc[improved_data['wer'].idxmin()]
    print(f"   • Lowest WER: {best_result['wer']:.3f}")
    print(f"     - ASR System: {best_result['asr_system']}")
    print(f"     - Strategy: {best_result['strategy']}")
    print(f"     - Sample: {best_result['sample']}")
    print(f"     - Sentences: {best_result['num_sentences']}")
    
    print(f"\n❌ WORST RESULTS (Improved only):")
    worst_result = improved_data.loc[improved_data['wer'].idxmax()]
    print(f"   • Highest WER: {worst_result['wer']:.3f}")
    print(f"     - ASR System: {worst_result['asr_system']}")
    print(f"     - Strategy: {worst_result['strategy']}")
    print(f"     - Sample: {worst_result['sample']}")
    print(f"     - Sentences: {worst_result['num_sentences']}")

def create_wer_tables(df):
    """Create detailed WER tables."""
    print(f"\n📋 DETAILED WER TABLES")
    print("="*80)
    
    # Table 1: WER by Strategy and Sentence Count (Improved only)
    improved_data = df[df['type'] == 'Improved']
    
    print(f"\n📊 Table 1: Average WER by Strategy and Sentence Count (Improved Results)")
    print("-" * 70)
    
    pivot_table = improved_data.pivot_table(
        values='wer', 
        index='strategy', 
        columns='num_sentences', 
        aggfunc='mean'
    )
    
    # Format and display the table
    print(f"{'Strategy':<25}", end='')
    for col in sorted(pivot_table.columns):
        print(f"{col:>8}s", end='')
    print()
    print("-" * 70)
    
    for strategy in pivot_table.index:
        print(f"{strategy.replace('_', ' ').title():<25}", end='')
        for col in sorted(pivot_table.columns):
            val = pivot_table.loc[strategy, col]
            if pd.isna(val):
                print(f"{'---':>8}", end='')
            else:
                print(f"{val:>8.3f}", end='')
        print()
    
    # Table 2: WER by ASR System and Type
    print(f"\n📊 Table 2: Average WER by ASR System and Type")
    print("-" * 50)
    
    asr_table = df.pivot_table(
        values='wer',
        index='asr_system',
        columns='type',
        aggfunc='mean'
    )
    
    print(f"{'ASR System':<15}", end='')
    for col in asr_table.columns:
        print(f"{col:>12}", end='')
    print(f"{'Improvement':>12}")
    print("-" * 50)
    
    for asr_system in asr_table.index:
        print(f"{asr_system:<15}", end='')
        raw_wer = asr_table.loc[asr_system, 'Raw ASR'] if 'Raw ASR' in asr_table.columns else None
        imp_wer = asr_table.loc[asr_system, 'Improved'] if 'Improved' in asr_table.columns else None
        
        for col in asr_table.columns:
            val = asr_table.loc[asr_system, col]
            print(f"{val:>12.3f}", end='')
        
        # Calculate improvement percentage
        if raw_wer and imp_wer and raw_wer > 0:
            improvement = ((raw_wer - imp_wer) / raw_wer) * 100
            print(f"{improvement:>11.1f}%")
        else:
            print(f"{'---':>12}")

def plot_wer_comparison(df, save_plots=False):
    """Create WER comparison plots."""
    print(f"\n📊 Generating WER comparison plots...")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('WER Analysis Dashboard', fontsize=16, fontweight='bold')
    
    # Plot 1: WER by ASR System and Type
    ax1 = axes[0, 0]
    
    asr_systems = df['asr_system'].unique()
    types = ['Raw ASR', 'Improved']
    
    x = np.arange(len(asr_systems))
    width = 0.35
    
    for i, result_type in enumerate(types):
        wer_values = []
        for asr_system in asr_systems:
            type_data = df[(df['asr_system'] == asr_system) & (df['type'] == result_type)]
            if not type_data.empty:
                wer_values.append(type_data['wer'].mean())
            else:
                wer_values.append(0)
        
        ax1.bar(x + i * width, wer_values, width, label=result_type, alpha=0.8)
    
    ax1.set_title('Average WER by ASR System')
    ax1.set_xlabel('ASR System')
    ax1.set_ylabel('WER')
    ax1.set_xticks(x + width / 2)
    ax1.set_xticklabels(asr_systems)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: WER by Strategy (Improved only)
    ax2 = axes[0, 1]
    
    improved_data = df[df['type'] == 'Improved']
    strategies = sorted(improved_data['strategy'].unique())
    strategy_wers = [improved_data[improved_data['strategy'] == s]['wer'].mean() for s in strategies]
    
    bars = ax2.bar(range(len(strategies)), strategy_wers, alpha=0.8, color='skyblue')
    ax2.set_title('Average WER by Strategy (Improved)')
    ax2.set_xlabel('Strategy')
    ax2.set_ylabel('WER')
    ax2.set_xticks(range(len(strategies)))
    ax2.set_xticklabels([s.replace('_', '\n').title() for s in strategies], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, value in zip(bars, strategy_wers):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f'{value:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: WER by Number of Sentences (Improved only)
    ax3 = axes[1, 0]
    
    sentence_counts = sorted(improved_data['num_sentences'].unique())
    sentence_wers = [improved_data[improved_data['num_sentences'] == sc]['wer'].mean() for sc in sentence_counts]
    
    ax3.plot(sentence_counts, sentence_wers, marker='o', linewidth=2, markersize=8, color='green')
    ax3.set_title('WER vs Number of Example Sentences (Improved)')
    ax3.set_xlabel('Number of Example Sentences')
    ax3.set_ylabel('WER')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for x, y in zip(sentence_counts, sentence_wers):
        ax3.text(x, y + 0.002, f'{y:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 4: WER Distribution Box Plot (Improved only)
    ax4 = axes[1, 1]
    
    box_data = [improved_data[improved_data['strategy'] == s]['wer'].values for s in strategies]
    
    bp = ax4.boxplot(box_data, patch_artist=True, tick_labels=[s.replace('_', '\n').title() for s in strategies])
    ax4.set_title('WER Distribution by Strategy (Improved)')
    ax4.set_xlabel('Strategy')
    ax4.set_ylabel('WER')
    ax4.tick_params(axis='x', rotation=45)
    
    # Color the boxes
    colors = plt.cm.Set3(np.linspace(0, 1, len(strategies)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.tight_layout()
    
    if save_plots:
        plt.savefig('wer_analysis_dashboard.png', dpi=300, bbox_inches='tight')
        print(f"💾 Saved plot to wer_analysis_dashboard.png")
    
    plt.show()

def main():
    print("🚀 Loading WER data from all experiments...")
    
    df = load_wer_data()
    
    if df.empty:
        print("❌ No data found! Please run some experiments first.")
        return
    
    print(f"✅ Loaded {len(df)} data points")
    
    # Print detailed summary
    print_wer_summary(df)
    
    # Create detailed tables
    create_wer_tables(df)
    
    # Generate plots
    plot_wer_comparison(df, save_plots=True)
    
    print(f"\n🎉 WER analysis complete!")

if __name__ == "__main__":
    main() 