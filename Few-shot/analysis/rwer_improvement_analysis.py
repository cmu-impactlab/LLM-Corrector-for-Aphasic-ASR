#!/usr/bin/env python3
"""
RWER Improvement Analysis - Custom Bar Graphs

Generates specific bar graph visualizations for RWER improvements:
1. ASR System RWER by Number of Sentences
2. Strategy RWER by Number of Sentences  
3. Overall Average RWER by Number of Sentences
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import seaborn as sns

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.file_utils import list_output_experiments

def load_rwer_data():
    """Load all data and calculate RWER improvements."""
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
                        
                        # Get Raw ASR and Improved WER values
                        raw_wer = None
                        improved_wer = None
                        
                        for _, row in df_metrics.iterrows():
                            if row['Type'] == 'Raw ASR':
                                raw_wer = float(row['WER'])
                            elif row['Type'] == 'Improved':
                                improved_wer = float(row['WER'])
                        
                        # Calculate RWER improvement if both values exist
                        if raw_wer is not None and improved_wer is not None and raw_wer > 0:
                            rwer_improvement = ((raw_wer - improved_wer) / raw_wer) * 100
                            
                            all_data.append({
                                'asr_system': system_name,
                                'strategy': exp_strategy,
                                'experiment': exp_name,
                                'sample': sample_dir,
                                'num_sentences': num_sentences,
                                'raw_wer': raw_wer,
                                'improved_wer': improved_wer,
                                'rwer_improvement': rwer_improvement
                            })
                        
                    except Exception as e:
                        print(f"⚠️  Error reading {metrics_file}: {e}")
                        continue
    
    return pd.DataFrame(all_data)

def create_rwer_bar_graphs(df):
    """Create the three requested RWER improvement bar graphs."""
    
    # Set up the figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('RWER Improvement Analysis by Number of Sentences', fontsize=16, fontweight='bold')
    
    # Define colors
    asr_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    strategy_colors = sns.color_palette("husl", 7)
    
    sentence_counts = sorted(df['num_sentences'].unique())
    
    # 1. ASR System RWER by Number of Sentences
    ax1 = axes[0]
    asr_systems = sorted(df['asr_system'].unique())
    
    asr_data = df.groupby(['asr_system', 'num_sentences'])['rwer_improvement'].mean().reset_index()
    
    x = np.arange(len(sentence_counts))
    width = 0.2
    
    for i, asr_system in enumerate(asr_systems):
        system_data = asr_data[asr_data['asr_system'] == asr_system]
        improvements = []
        for sent_count in sentence_counts:
            improvement = system_data[system_data['num_sentences'] == sent_count]['rwer_improvement']
            improvements.append(improvement.iloc[0] if len(improvement) > 0 else 0)
        
        bars = ax1.bar(x + i * width, improvements, width, label=asr_system, color=asr_colors[i], alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, improvements):
            if val > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=8)
    
    ax1.set_title('RWER Improvement by ASR System')
    ax1.set_xlabel('Number of Example Sentences')
    ax1.set_ylabel('RWER Improvement (%)')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(sentence_counts)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Strategy RWER by Number of Sentences
    ax2 = axes[1]
    strategies = sorted(df['strategy'].unique())
    
    strategy_data = df.groupby(['strategy', 'num_sentences'])['rwer_improvement'].mean().reset_index()
    
    x = np.arange(len(sentence_counts))
    width = 0.12
    
    for i, strategy in enumerate(strategies):
        strat_data = strategy_data[strategy_data['strategy'] == strategy]
        improvements = []
        for sent_count in sentence_counts:
            improvement = strat_data[strat_data['num_sentences'] == sent_count]['rwer_improvement']
            improvements.append(improvement.iloc[0] if len(improvement) > 0 else 0)
        
        bars = ax2.bar(x + i * width, improvements, width, 
                      label=strategy.replace('_', ' ').title(), 
                      color=strategy_colors[i], alpha=0.8)
        
        # Add value labels on bars (only for higher values to avoid clutter)
        for bar, val in zip(bars, improvements):
            if val > 15:  # Only show labels for higher values
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=7, rotation=90)
    
    ax2.set_title('RWER Improvement by Strategy')
    ax2.set_xlabel('Number of Example Sentences')
    ax2.set_ylabel('RWER Improvement (%)')
    ax2.set_xticks(x + width * 3)
    ax2.set_xticklabels(sentence_counts)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 3. Overall Average RWER by Number of Sentences (All 4 ASR Systems Combined)
    ax3 = axes[2]
    overall_data = df.groupby('num_sentences')['rwer_improvement'].mean().reset_index()
    
    bars = ax3.bar(overall_data['num_sentences'], overall_data['rwer_improvement'], 
                   color='#9467bd', alpha=0.8, width=0.6)
    
    # Add value labels on bars
    for bar, sent_count in zip(bars, overall_data['num_sentences']):
        improvement = overall_data[overall_data['num_sentences'] == sent_count]['rwer_improvement'].iloc[0]
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{improvement:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax3.set_title('Average RWER Improvement\n(All 4 ASR Systems Combined)')
    ax3.set_xlabel('Number of Example Sentences')
    ax3.set_ylabel('RWER Improvement (%)')
    ax3.set_xticks(overall_data['num_sentences'])
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    output_path = 'rwer_improvement_bar_graphs.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved RWER improvement bar graphs to: {output_path}")
    
    return fig

def print_rwer_summary(df):
    """Print detailed RWER improvement summary."""
    print("\n" + "="*80)
    print("📊 RWER IMPROVEMENT ANALYSIS SUMMARY")
    print("="*80)
    
    print(f"\n📈 Dataset Overview:")
    print(f"   • Total data points: {len(df)}")
    print(f"   • ASR systems: {list(df['asr_system'].unique())}")
    print(f"   • Strategies: {list(df['strategy'].unique())}")
    print(f"   • Sentence counts: {sorted(df['num_sentences'].unique())}")
    
    # Overall RWER improvement by sentence count
    print(f"\n🎯 OVERALL AVERAGE RWER IMPROVEMENT BY SENTENCE COUNT:")
    for sent_count in sorted(df['num_sentences'].unique()):
        avg_improvement = df[df['num_sentences'] == sent_count]['rwer_improvement'].mean()
        count = len(df[df['num_sentences'] == sent_count])
        print(f"   • {sent_count} sentences: {avg_improvement:.1f}% (n={count})")
    
    # RWER improvement by ASR system
    print(f"\n🎤 RWER IMPROVEMENT BY ASR SYSTEM:")
    for asr_system in sorted(df['asr_system'].unique()):
        system_data = df[df['asr_system'] == asr_system]
        avg_improvement = system_data['rwer_improvement'].mean()
        print(f"   • {asr_system}: {avg_improvement:.1f}%")
        
        # By sentence count for this ASR system
        for sent_count in sorted(df['num_sentences'].unique()):
            sent_data = system_data[system_data['num_sentences'] == sent_count]
            if not sent_data.empty:
                avg_sent_improvement = sent_data['rwer_improvement'].mean()
                print(f"     - {sent_count} sentences: {avg_sent_improvement:.1f}%")
    
    # RWER improvement by strategy
    print(f"\n📋 RWER IMPROVEMENT BY STRATEGY:")
    for strategy in sorted(df['strategy'].unique()):
        strategy_data = df[df['strategy'] == strategy]
        avg_improvement = strategy_data['rwer_improvement'].mean()
        print(f"   • {strategy.replace('_', ' ').title()}: {avg_improvement:.1f}%")
        
        # By sentence count for this strategy
        for sent_count in sorted(df['num_sentences'].unique()):
            sent_data = strategy_data[strategy_data['num_sentences'] == sent_count]
            if not sent_data.empty:
                avg_sent_improvement = sent_data['rwer_improvement'].mean()
                print(f"     - {sent_count} sentences: {avg_sent_improvement:.1f}%")
    
    # Best and worst improvements
    print(f"\n🏆 BEST RWER IMPROVEMENTS:")
    best_result = df.loc[df['rwer_improvement'].idxmax()]
    print(f"   • Highest RWER Improvement: {best_result['rwer_improvement']:.1f}%")
    print(f"     - ASR System: {best_result['asr_system']}")
    print(f"     - Strategy: {best_result['strategy']}")
    print(f"     - Sample: {best_result['sample']}")
    print(f"     - Sentences: {best_result['num_sentences']}")
    
    print(f"\n❌ WORST RWER IMPROVEMENTS:")
    worst_result = df.loc[df['rwer_improvement'].idxmin()]
    print(f"   • Lowest RWER Improvement: {worst_result['rwer_improvement']:.1f}%")
    print(f"     - ASR System: {worst_result['asr_system']}")
    print(f"     - Strategy: {worst_result['strategy']}")
    print(f"     - Sample: {worst_result['sample']}")
    print(f"     - Sentences: {worst_result['num_sentences']}")

def main():
    """Main function."""
    print("🚀 Loading RWER improvement data...")
    
    df = load_rwer_data()
    
    if df.empty:
        print("❌ No experiment data found!")
        print("📁 Please run some experiments first using run_experiments.py")
        return
    
    print(f"✅ Loaded {len(df)} RWER improvement data points")
    
    # Print summary
    print_rwer_summary(df)
    
    # Create bar graphs
    print(f"\n📊 Generating RWER improvement bar graphs...")
    create_rwer_bar_graphs(df)
    
    print(f"\n🎉 RWER improvement analysis complete!")
    print(f"📈 Check the generated bar graphs: rwer_improvement_bar_graphs.png")

if __name__ == "__main__":
    main() no