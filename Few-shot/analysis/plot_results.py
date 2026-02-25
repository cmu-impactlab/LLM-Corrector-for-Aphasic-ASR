import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('very_last_csvs/byasr.csv', skiprows=1)  # Skip the comment row

# Prepare the data for plotting
sentence_counts = [2, 4, 6, 8, 10]

# Create figure and axes with a larger size
plt.figure(figsize=(15, 8))

# Plot Raw WER
for idx, asr in enumerate(df['ASR']):
    raw_wer_values = [df[f'{n}sent_Raw_WER'].iloc[idx] for n in sentence_counts]
    improved_wer_values = [df[f'{n}sent_Improved_WER'].iloc[idx] for n in sentence_counts]
    
    # Plot with different line styles for Raw and Improved
    plt.plot(sentence_counts, raw_wer_values, marker='o', linestyle='-', 
             label=f'{asr} (Raw)', linewidth=2, markersize=8)
    plt.plot(sentence_counts, improved_wer_values, marker='s', linestyle='--',
             label=f'{asr} (Improved)', linewidth=2, markersize=8)

# Customize the plot
plt.xlabel('Number of Sentences', fontsize=12)
plt.ylabel('Word Error Rate (WER)', fontsize=12)
plt.title('WER Comparison Across Different Sentence Counts by ASR System', fontsize=14, pad=20)

# Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# Customize x-axis
plt.xticks(sentence_counts)

# Add legend with better positioning and formatting
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., 
          fontsize=10, frameon=True)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('wer_by_sentence_count.png', dpi=300, bbox_inches='tight')
plt.close()

# Create a second plot focusing on the improvement percentage
plt.figure(figsize=(15, 8))

for idx, asr in enumerate(df['ASR']):
    raw_wer_values = np.array([df[f'{n}sent_Raw_WER'].iloc[idx] for n in sentence_counts])
    improved_wer_values = np.array([df[f'{n}sent_Improved_WER'].iloc[idx] for n in sentence_counts])
    
    # Calculate improvement percentage
    improvement_percentage = ((raw_wer_values - improved_wer_values) / raw_wer_values) * 100
    
    plt.plot(sentence_counts, improvement_percentage, marker='o', linestyle='-',
             label=f'{asr}', linewidth=2, markersize=8)

# Customize the second plot
plt.xlabel('Number of Sentences', fontsize=12)
plt.ylabel('WER Improvement (%)', fontsize=12)
plt.title('WER Improvement Percentage by Sentence Count', fontsize=14, pad=20)

# Add grid
plt.grid(True, linestyle='--', alpha=0.7)

# Customize x-axis
plt.xticks(sentence_counts)

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.,
          fontsize=10, frameon=True)

# Adjust layout
plt.tight_layout()

# Save the second plot
plt.savefig('wer_improvement_percentage.png', dpi=300, bbox_inches='tight')
plt.close()
