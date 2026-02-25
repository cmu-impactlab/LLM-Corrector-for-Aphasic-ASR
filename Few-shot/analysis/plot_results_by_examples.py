import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('very_last_csvs/byexamples.csv', skiprows=1)

# Group ASRs by their average raw WER performance
df['avg_raw_wer'] = df[['data_driven_Raw_WER', 'exhaustive_phoneme_Raw_WER', 'random_error_Raw_WER']].mean(axis=1)
df['tier'] = pd.qcut(df['avg_raw_wer'], q=2, labels=['Tier 1 (Better)', 'Tier 2 (Lower)'])

# Create figure for the first plot - Raw vs Improved WER by Strategy
plt.figure(figsize=(15, 10))

# Prepare data for plotting
strategies = ['data_driven', 'exhaustive_phoneme', 'random_error']
x = np.arange(len(df['ASR']))
width = 0.15  # Width of bars

# Plot bars for each strategy
for i, strategy in enumerate(strategies):
    raw_data = df[f'{strategy}_Raw_WER']
    improved_data = df[f'{strategy}_Improved_WER']
    
    plt.bar(x + i*width*2, raw_data, width, label=f'{strategy} (Raw)',
            alpha=0.7, hatch='/')
    plt.bar(x + i*width*2 + width, improved_data, width,
            label=f'{strategy} (Improved)', alpha=0.7)

# Customize the plot
plt.xlabel('ASR Systems', fontsize=12)
plt.ylabel('Word Error Rate (WER)', fontsize=12)
plt.title('WER Comparison Across Different Strategies by ASR System', fontsize=14, pad=20)

# Set x-axis labels
plt.xticks(x + width*2.5, df['ASR'], rotation=45)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.3, axis='y')

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
          fontsize=10, frameon=True)

# Adjust layout
plt.tight_layout()

# Save the first plot
plt.savefig('wer_by_strategy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create second plot - Improvement Percentage by Strategy
plt.figure(figsize=(15, 8))

# Calculate improvement percentages
improvement_data = []
for strategy in strategies:
    improvement = ((df[f'{strategy}_Raw_WER'] - df[f'{strategy}_Improved_WER']) / 
                  df[f'{strategy}_Raw_WER'] * 100)
    improvement_data.append(improvement)

# Create grouped bar plot by tiers
for tier in df['tier'].unique():
    tier_mask = df['tier'] == tier
    tier_asrs = df[tier_mask]['ASR']
    
    for i, strategy in enumerate(strategies):
        improvement = improvement_data[i][tier_mask]
        x_pos = np.arange(len(tier_asrs)) + i*width
        plt.bar(x_pos, improvement, width, label=f'{strategy} ({tier})',
                alpha=0.8)

# Customize the second plot
plt.xlabel('ASR Systems', fontsize=12)
plt.ylabel('WER Improvement (%)', fontsize=12)
plt.title('WER Improvement Percentage by Strategy and ASR Tier', fontsize=14, pad=20)

# Set x-axis labels
plt.xticks(np.arange(len(df['ASR'])) + width, df['ASR'], rotation=45)

# Add grid
plt.grid(True, linestyle='--', alpha=0.3, axis='y')

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
          fontsize=10, frameon=True)

# Adjust layout
plt.tight_layout()

# Save the second plot
plt.savefig('wer_improvement_by_strategy.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a third plot - Heatmap of improvements
plt.figure(figsize=(12, 8))

# Prepare improvement data for heatmap
improvement_matrix = np.zeros((len(df['ASR']), len(strategies)))
for i, strategy in enumerate(strategies):
    improvement = ((df[f'{strategy}_Raw_WER'] - df[f'{strategy}_Improved_WER']) / 
                  df[f'{strategy}_Raw_WER'] * 100)
    improvement_matrix[:, i] = improvement

# Create heatmap
sns.heatmap(improvement_matrix, 
            xticklabels=strategies,
            yticklabels=df['ASR'],
            annot=True,
            fmt='.1f',
            cmap='YlOrRd',
            cbar_kws={'label': 'WER Improvement (%)'})

plt.title('WER Improvement Heatmap by Strategy and ASR', pad=20)
plt.tight_layout()

# Save the heatmap
plt.savefig('wer_improvement_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()


