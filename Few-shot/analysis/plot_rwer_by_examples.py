import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('very_last_csvs/byexamples.csv', skiprows=1)

# Create figure for WER comparison
plt.figure(figsize=(15, 10))

# Prepare data for plotting
strategies = ['data_driven', 'exhaustive_phoneme', 'random_error']
x = np.arange(len(df['ASR']))
width = 0.15  # Width of bars

# Plot bars for each strategy
colors = ['#2ecc71', '#3498db', '#e74c3c']  # Green, Blue, Red
for i, (strategy, color) in enumerate(zip(strategies, colors)):
    raw_data = df[f'{strategy}_Raw_WER']
    improved_data = df[f'{strategy}_Improved_WER']
    
    # Raw WER bars
    plt.bar(x + i*width*2, raw_data, width, 
            label=f'{strategy} (Raw)',
            color=color, alpha=0.7, hatch='/')
    
    # Improved WER bars
    plt.bar(x + i*width*2 + width, improved_data, width,
            label=f'{strategy} (Improved)',
            color=color, alpha=0.4)

# Customize the plot
plt.xlabel('ASR Systems', fontsize=12, fontweight='bold')
plt.ylabel('Word Error Rate (WER)', fontsize=12, fontweight='bold')
plt.title('WER Comparison Across Different Strategies by ASR System', 
          fontsize=14, pad=20, fontweight='bold')

# Set x-axis labels
plt.xticks(x + width*2.5, df['ASR'], rotation=45)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.3, axis='y')

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
          fontsize=10, frameon=True)

# Adjust layout
plt.tight_layout()

# Save the WER plot
plt.savefig('wer_strategy_comparison_enhanced.png', dpi=300, bbox_inches='tight')
plt.close()

# Create figure for WER Improvement Percentage
plt.figure(figsize=(15, 10))

# Calculate improvement percentage for each strategy
for i, (strategy, color) in enumerate(zip(strategies, colors)):
    raw_wer = df[f'{strategy}_Raw_WER']
    improved_wer = df[f'{strategy}_Improved_WER']
    
    # Calculate improvement percentage
    improvement_percentage = ((raw_wer - improved_wer) / raw_wer) * 100
    
    # Plot improvement percentage bars
    plt.bar(x + i*width, improvement_percentage, width,
            label=f'{strategy}',
            color=color, alpha=0.7)

    # Add percentage labels on top of bars
    for j, pct in enumerate(improvement_percentage):
        plt.text(x[j] + i*width, pct, f'{pct:.1f}%', 
                ha='center', va='bottom')

# Customize the improvement plot
plt.xlabel('ASR Systems', fontsize=12, fontweight='bold')
plt.ylabel('WER Improvement (%)', fontsize=12, fontweight='bold')
plt.title('WER Improvement Percentage by Strategy\n(Higher is Better)', 
          fontsize=14, pad=20, fontweight='bold')

# Set x-axis labels
plt.xticks(x + width*1.5, df['ASR'], rotation=45)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.3, axis='y')

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
          fontsize=10, frameon=True)

# Set y-axis limits to start from 0
plt.ylim(0, None)  # Start from 0, auto-adjust upper limit

# Adjust layout
plt.tight_layout()

# Save the improvement percentage plot
plt.savefig('wer_improvement_percentage_by_strategy.png', dpi=300, bbox_inches='tight')
plt.close()