#!/usr/bin/env python3
"""
Per-ASR Analysis and Visualization Tool

This script analyzes experiment results from the per-ASR output directory structure
and generates comprehensive graphs and visualizations.

Usage:
    python analyze_per_asr_results.py
    python analyze_per_asr_results.py --asr-system "Azure"
    python analyze_per_asr_results.py --strategy data_driven
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import numpy as np
from typing import Dict, List, Optional

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.file_utils import ASRSystem, list_output_experiments

# Set style for better looking plots
plt.style.use('default')
sns.set_palette("husl")

def load_experiment_data(asr_system: Optional[str] = None, strategy: Optional[str] = None) -> pd.DataFrame:
    """
    Load experiment data from the new multi-ASR directory structure.
    
    Args:
        asr_system: Filter by specific ASR system (e.g., "Azure")
        strategy: Filter by specific strategy (e.g., "data_driven")
        
    Returns:
        DataFrame with all experiment results
    """
    all_data = []
    
    # Get all experiments
    experiments = list_output_experiments()
    
    if not experiments:
        print("❌ No experiment data found!")
        print("📁 Please run some experiments first using run_experiments.py")
        return pd.DataFrame()
    
    print(f"🔍 Found experiments in {len(experiments)} ASR system(s)")
    
    for system_name, exp_list in experiments.items():
        # Filter by ASR system if specified
        if asr_system and system_name != asr_system:
            continue
            
        print(f"📊 Processing {system_name}: {len(exp_list)} experiments")
        
        for exp_info in exp_list:
            exp_strategy = exp_info['strategy']
            exp_name = exp_info['experiment']
            exp_path = exp_info['path']
            
            # Filter by strategy if specified
            if strategy and exp_strategy != strategy:
                continue
            
            # Look for sample directories
            if not os.path.exists(exp_path):
                continue
                
            for sample_dir in os.listdir(exp_path):
                sample_path = os.path.join(exp_path, sample_dir)
                if not os.path.isdir(sample_path):
                    continue
                
                # Look for sentence count directories
                for sentence_dir in os.listdir(sample_path):
                    sentence_path = os.path.join(sample_path, sentence_dir)
                    if not os.path.isdir(sentence_path) or not sentence_dir.endswith('_sentences'):
                        continue
                    
                    # Extract sentence count
                    try:
                        num_sentences = int(sentence_dir.split('_')[0])
                    except ValueError:
                        continue
                    
                    # Look for evaluation_metrics.csv
                    metrics_file = os.path.join(sentence_path, 'evaluation_metrics.csv')
                    if not os.path.exists(metrics_file):
                        continue
                    
                    try:
                        # Read the metrics
                        df_metrics = pd.read_csv(metrics_file)
                        
                        # Add metadata
                        df_metrics['asr_system'] = system_name
                        df_metrics['strategy'] = exp_strategy
                        df_metrics['experiment'] = exp_name
                        df_metrics['sample'] = sample_dir
                        df_metrics['num_sentences'] = num_sentences
                        df_metrics['file_path'] = metrics_file
                        
                        all_data.append(df_metrics)
                        
                    except Exception as e:
                        print(f"⚠️  Error reading {metrics_file}: {e}")
                        continue
    
    if not all_data:
        print("❌ No valid experiment data found!")
        return pd.DataFrame()
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Clean data
    for col in ['WER', 'CER', 'SIM']:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
    
    combined_df = combined_df.dropna(subset=['WER', 'CER', 'SIM'])
    
    print(f"✅ Loaded {len(combined_df)} experiment results")
    print(f"📈 ASR Systems: {combined_df['asr_system'].unique()}")
    print(f"🎯 Strategies: {combined_df['strategy'].unique()}")
    
    return combined_df

def plot_performance_comparison(df: pd.DataFrame, metric: str = 'WER', save_path: str = None):
    """Plot performance comparison across ASR systems and strategies."""
    
    # Filter for 'Improved' results only
    plot_df = df[df['Type'] == 'Improved'].copy()
    
    if plot_df.empty:
        print(f"❌ No 'Improved' results found for {metric}")
        return
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{metric} Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Performance by ASR System
    ax1 = axes[0, 0]
    asr_data = plot_df.groupby(['asr_system', 'num_sentences'])[metric].mean().reset_index()
    
    for asr_system in asr_data['asr_system'].unique():
        system_data = asr_data[asr_data['asr_system'] == asr_system]
        ax1.plot(system_data['num_sentences'], system_data[metric], 
                marker='o', label=asr_system, linewidth=2)
    
    ax1.set_title('Performance by ASR System')
    ax1.set_xlabel('Number of Example Sentences')
    ax1.set_ylabel(f'{metric} Score')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Performance by Strategy
    ax2 = axes[0, 1]
    strategy_data = plot_df.groupby(['strategy', 'num_sentences'])[metric].mean().reset_index()
    
    for strategy in strategy_data['strategy'].unique():
        strat_data = strategy_data[strategy_data['strategy'] == strategy]
        ax2.plot(strat_data['num_sentences'], strat_data[metric], 
                marker='s', label=strategy.replace('_', ' ').title(), linewidth=2)
    
    ax2.set_title('Performance by Strategy')
    ax2.set_xlabel('Number of Example Sentences')
    ax2.set_ylabel(f'{metric} Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Box plot by ASR System
    ax3 = axes[1, 0]
    asr_systems = plot_df['asr_system'].unique()
    box_data = [plot_df[plot_df['asr_system'] == system][metric].values for system in asr_systems]
    
    bp = ax3.boxplot(box_data, labels=[s.replace(' ', '\n') for s in asr_systems], patch_artist=True)
    ax3.set_title(f'{metric} Distribution by ASR System')
    ax3.set_ylabel(f'{metric} Score')
    
    # Color the boxes
    colors = sns.color_palette("husl", len(asr_systems))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 4. Heatmap: Strategy vs ASR System
    ax4 = axes[1, 1]
    
    # Create pivot table
    heatmap_data = plot_df.groupby(['strategy', 'asr_system'])[metric].mean().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='strategy', columns='asr_system', values=metric)
    
    im = ax4.imshow(heatmap_pivot.values, cmap='RdYlBu_r', aspect='auto')
    
    # Set ticks and labels
    ax4.set_xticks(range(len(heatmap_pivot.columns)))
    ax4.set_yticks(range(len(heatmap_pivot.index)))
    ax4.set_xticklabels([col.replace(' ', '\n') for col in heatmap_pivot.columns], rotation=45)
    ax4.set_yticklabels([idx.replace('_', ' ').title() for idx in heatmap_pivot.index])
    
    # Add colorbar
    plt.colorbar(im, ax=ax4, label=f'Average {metric}')
    ax4.set_title(f'Average {metric}: Strategy vs ASR System')
    
    # Add text annotations
    for i in range(len(heatmap_pivot.index)):
        for j in range(len(heatmap_pivot.columns)):
            value = heatmap_pivot.iloc[i, j]
            if not pd.isna(value):
                ax4.text(j, i, f'{value:.3f}', ha="center", va="center", 
                        color="white" if value > heatmap_pivot.values.mean() else "black")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved plot to {save_path}")
    
    plt.show()

def plot_improvement_analysis(df: pd.DataFrame, save_path: str = None):
    """Plot improvement analysis (Raw vs Improved)."""
    
    # Calculate improvements
    improvement_data = []
    
    for _, row in df.iterrows():
        if row['Type'] == 'Raw ASR':
            # Find corresponding improved result
            improved_row = df[
                (df['asr_system'] == row['asr_system']) &
                (df['strategy'] == row['strategy']) &
                (df['experiment'] == row['experiment']) &
                (df['sample'] == row['sample']) &
                (df['num_sentences'] == row['num_sentences']) &
                (df['Type'] == 'Improved')
            ]
            
            if not improved_row.empty:
                improved_row = improved_row.iloc[0]
                
                # Calculate improvements
                wer_improvement = ((row['WER'] - improved_row['WER']) / row['WER']) * 100 if row['WER'] > 0 else 0
                cer_improvement = ((row['CER'] - improved_row['CER']) / row['CER']) * 100 if row['CER'] > 0 else 0
                sim_improvement = ((improved_row['SIM'] - row['SIM']) / row['SIM']) * 100 if row['SIM'] > 0 else 0
                
                improvement_data.append({
                    'asr_system': row['asr_system'],
                    'strategy': row['strategy'],
                    'experiment': row['experiment'],
                    'sample': row['sample'],
                    'num_sentences': row['num_sentences'],
                    'wer_improvement': wer_improvement,
                    'cer_improvement': cer_improvement,
                    'sim_improvement': sim_improvement,
                    'raw_wer': row['WER'],
                    'improved_wer': improved_row['WER'],
                    'raw_cer': row['CER'],
                    'improved_cer': improved_row['CER'],
                    'raw_sim': row['SIM'],
                    'improved_sim': improved_row['SIM']
                })
    
    if not improvement_data:
        print("❌ No improvement data found (need both Raw ASR and Improved results)")
        return
    
    improvement_df = pd.DataFrame(improvement_data)
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ASR Post-Processing Improvement Analysis', fontsize=16, fontweight='bold')
    
    # 1. WER Improvement by ASR System
    ax1 = axes[0, 0]
    
    asr_systems = improvement_df['asr_system'].unique()
    wer_improvements = [improvement_df[improvement_df['asr_system'] == system]['wer_improvement'].values 
                       for system in asr_systems]
    
    bp1 = ax1.boxplot(wer_improvements, labels=[s.replace(' ', '\n') for s in asr_systems], patch_artist=True)
    ax1.set_title('WER Improvement by ASR System')
    ax1.set_ylabel('WER Improvement (%)')
    ax1.grid(True, alpha=0.3)
    
    # Color boxes
    colors = sns.color_palette("husl", len(asr_systems))
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 2. Improvement by Strategy
    ax2 = axes[0, 1]
    
    strategies = improvement_df['strategy'].unique()
    strategy_improvements = [improvement_df[improvement_df['strategy'] == strategy]['wer_improvement'].values 
                           for strategy in strategies]
    
    bp2 = ax2.boxplot(strategy_improvements, labels=[s.replace('_', '\n').title() for s in strategies], patch_artist=True)
    ax2.set_title('WER Improvement by Strategy')
    ax2.set_ylabel('WER Improvement (%)')
    ax2.grid(True, alpha=0.3)
    
    # Color boxes
    colors = sns.color_palette("husl", len(strategies))
    for patch, color in zip(bp2['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 3. Improvement vs Number of Sentences
    ax3 = axes[1, 0]
    
    sentence_data = improvement_df.groupby(['num_sentences', 'asr_system'])['wer_improvement'].mean().reset_index()
    
    for asr_system in sentence_data['asr_system'].unique():
        system_data = sentence_data[sentence_data['asr_system'] == asr_system]
        ax3.plot(system_data['num_sentences'], system_data['wer_improvement'], 
                marker='o', label=asr_system, linewidth=2)
    
    ax3.set_title('WER Improvement vs Number of Examples')
    ax3.set_xlabel('Number of Example Sentences')
    ax3.set_ylabel('WER Improvement (%)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Correlation matrix
    ax4 = axes[1, 1]
    
    corr_data = improvement_df[['wer_improvement', 'cer_improvement', 'sim_improvement', 'num_sentences']]
    corr_matrix = corr_data.corr()
    
    im = ax4.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax4.set_xticks(range(len(corr_matrix.columns)))
    ax4.set_yticks(range(len(corr_matrix.index)))
    ax4.set_xticklabels([col.replace('_', ' ').title() for col in corr_matrix.columns], rotation=45)
    ax4.set_yticklabels([idx.replace('_', ' ').title() for idx in corr_matrix.index])
    
    # Add correlation values
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', ha="center", va="center",
                    color="white" if abs(corr_matrix.iloc[i, j]) > 0.5 else "black")
    
    plt.colorbar(im, ax=ax4, label='Correlation')
    ax4.set_title('Improvement Metrics Correlation')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved improvement analysis to {save_path}")
    
    plt.show()
    
    return improvement_df

def generate_summary_report(df: pd.DataFrame, improvement_df: pd.DataFrame = None):
    """Generate a summary report of the analysis."""
    
    print("\n" + "="*60)
    print("📊 MULTI-ASR ANALYSIS SUMMARY REPORT")
    print("="*60)
    
    # Basic statistics
    print(f"\n📈 Dataset Overview:")
    print(f"   • Total experiments: {len(df)}")
    print(f"   • ASR systems: {len(df['asr_system'].unique())}")
    print(f"   • Strategies: {len(df['strategy'].unique())}")
    print(f"   • Samples: {len(df['sample'].unique())}")
    print(f"   • Sentence counts: {sorted(df['num_sentences'].unique())}")
    
    # Performance by ASR system
    print(f"\n🎙️ Performance by ASR System (Improved results):")
    improved_df = df[df['Type'] == 'Improved']
    
    for asr_system in improved_df['asr_system'].unique():
        system_data = improved_df[improved_df['asr_system'] == asr_system]
        avg_wer = system_data['WER'].mean()
        avg_cer = system_data['CER'].mean()
        avg_sim = system_data['SIM'].mean()
        print(f"   • {asr_system}:")
        print(f"     - WER: {avg_wer:.3f} (lower is better)")
        print(f"     - CER: {avg_cer:.3f} (lower is better)")
        print(f"     - SIM: {avg_sim:.3f} (higher is better)")
    
    # Performance by strategy
    print(f"\n🎯 Performance by Strategy (Improved results):")
    for strategy in improved_df['strategy'].unique():
        strategy_data = improved_df[improved_df['strategy'] == strategy]
        avg_wer = strategy_data['WER'].mean()
        avg_cer = strategy_data['CER'].mean()
        avg_sim = strategy_data['SIM'].mean()
        print(f"   • {strategy.replace('_', ' ').title()}:")
        print(f"     - WER: {avg_wer:.3f}")
        print(f"     - CER: {avg_cer:.3f}")
        print(f"     - SIM: {avg_sim:.3f}")
    
    # Improvement analysis
    if improvement_df is not None and not improvement_df.empty:
        print(f"\n📈 Improvement Analysis:")
        avg_wer_improvement = improvement_df['wer_improvement'].mean()
        avg_cer_improvement = improvement_df['cer_improvement'].mean()
        avg_sim_improvement = improvement_df['sim_improvement'].mean()
        
        print(f"   • Average WER improvement: {avg_wer_improvement:.1f}%")
        print(f"   • Average CER improvement: {avg_cer_improvement:.1f}%")
        print(f"   • Average SIM improvement: {avg_sim_improvement:.1f}%")
        
        # Best performing combinations
        best_wer = improvement_df.loc[improvement_df['wer_improvement'].idxmax()]
        print(f"\n🏆 Best WER improvement: {best_wer['wer_improvement']:.1f}%")
        print(f"   • ASR System: {best_wer['asr_system']}")
        print(f"   • Strategy: {best_wer['strategy']}")
        print(f"   • Sample: {best_wer['sample']}")
        print(f"   • Examples: {best_wer['num_sentences']}")
    
    print("\n" + "="*60)

def main():
    parser = argparse.ArgumentParser(description="Analyze multi-ASR experiment results")
    parser.add_argument('--asr-system', help='Filter by ASR system (e.g., "Azure")')
    parser.add_argument('--strategy', help='Filter by strategy (e.g., "data_driven")')
    parser.add_argument('--metric', default='WER', choices=['WER', 'CER', 'SIM'], 
                       help='Primary metric for analysis (default: WER)')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files')
    
    args = parser.parse_args()
    
    print("🚀 Starting Multi-ASR Analysis")
    print("="*40)
    
    # Load data
    df = load_experiment_data(asr_system=args.asr_system, strategy=args.strategy)
    
    if df.empty:
        print("❌ No data to analyze. Please run some experiments first!")
        return
    
    # Create output directory for plots
    output_dir = "analysis_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate plots
    print(f"\n📊 Generating {args.metric} performance plots...")
    
    if args.save_plots:
        performance_plot_path = os.path.join(output_dir, f"{args.metric.lower()}_performance_analysis.png")
    else:
        performance_plot_path = None
    
    plot_performance_comparison(df, metric=args.metric, save_path=performance_plot_path)
    
    print(f"\n📈 Generating improvement analysis...")
    
    if args.save_plots:
        improvement_plot_path = os.path.join(output_dir, "improvement_analysis.png")
    else:
        improvement_plot_path = None
    
    improvement_df = plot_improvement_analysis(df, save_path=improvement_plot_path)
    
    # Generate summary report
    generate_summary_report(df, improvement_df)
    
    # Save summary to file
    if args.save_plots and improvement_df is not None:
        summary_path = os.path.join(output_dir, "summary_report.csv")
        improvement_df.to_csv(summary_path, index=False)
        print(f"\n💾 Saved detailed results to {summary_path}")
    
    print(f"\n🎉 Analysis complete!")
    if args.save_plots:
        print(f"📁 Results saved in: {output_dir}/")

if __name__ == "__main__":
    main() 