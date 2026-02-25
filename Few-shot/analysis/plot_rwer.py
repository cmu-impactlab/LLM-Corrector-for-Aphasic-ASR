import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('very_last_csvs/byrwer.csv', skiprows=1)

# Sort ASRs by Raw WER for better visualization
df = df.sort_values('Raw_WER')

# 1. Combined Bar and Line Plot
plt.figure(figsize=(15, 8))

# Create bar plot for WER values
x = np.arange(len(df['ASR']))
width = 0.35

# Plot bars
plt.bar(x - width/2, df['Raw_WER'], width, label='Raw WER', color='#3498db', alpha=0.7)
plt.bar(x + width/2, df['Improved_WER'], width, label='Improved WER', color='#2ecc71', alpha=0.7)

# Plot RWER as a line
ax2 = plt.twinx()
line = ax2.plot(x, df['RWER'], 'r-', label='RWER (%)', linewidth=2, marker='o')
ax2.set_ylabel('Relative WER Improvement (%)', color='red', fontsize=12)

# Customize the plot
plt.xlabel('ASR Systems', fontsize=12)
plt.ylabel('Word Error Rate (WER)', fontsize=12)
plt.title('WER Values and Relative Improvement by ASR System', fontsize=14, pad=20)

# Set x-axis labels
plt.xticks(x, df['ASR'], rotation=45, ha='right')

# Add grid
plt.grid(True, linestyle='--', alpha=0.3)

# Combine legends
lines1, labels1 = plt.gca().get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', bbox_to_anchor=(1.05, 1))

# Add value labels on the line
for i, rwer in enumerate(df['RWER']):
    ax2.text(i, rwer, f'{rwer:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('wer_and_rwer_combined.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Sorted RWER Bar Plot with Performance Categories
plt.figure(figsize=(15, 8))

# Create color mapping based on RWER ranges
def get_color(rwer):
    if rwer >= 40:
        return '#2ecc71'  # Green for high improvement
    elif rwer >= 20:
        return '#f1c40f'  # Yellow for medium improvement
    else:
        return '#e74c3c'  # Red for lower improvement

colors = [get_color(rwer) for rwer in df['RWER']]

# Create bar plot
bars = plt.bar(x, df['RWER'], color=colors)

# Add value labels on top of bars
for i, v in enumerate(df['RWER']):
    plt.text(i, v, f'{v:.1f}%', ha='center', va='bottom')

# Customize the plot
plt.xlabel('ASR Systems', fontsize=12)
plt.ylabel('Relative WER Improvement (%)', fontsize=12)
plt.title('Relative WER Improvement by ASR System\n(Color indicates improvement level)', 
          fontsize=14, pad=20)

# Set x-axis labels
plt.xticks(x, df['ASR'], rotation=45, ha='right')

# Add grid
plt.grid(True, linestyle='--', alpha=0.3, axis='y')

# Add legend for color categories
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#2ecc71', label='High Improvement (≥40%)'),
    Patch(facecolor='#f1c40f', label='Medium Improvement (20-40%)'),
    Patch(facecolor='#e74c3c', label='Lower Improvement (<20%)')
]
plt.legend(handles=legend_elements, loc='upper right')

plt.tight_layout()
plt.savefig('rwer_categorized.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Scatter plot showing relationship between Raw WER and Improvement
plt.figure(figsize=(12, 8))

# Create scatter plot
plt.scatter(df['Raw_WER'], df['RWER'], s=100, alpha=0.6)

# Add labels for each point
for i, txt in enumerate(df['ASR']):
    plt.annotate(txt, (df['Raw_WER'].iloc[i], df['RWER'].iloc[i]),
                xytext=(5, 5), textcoords='offset points')

# Add trend line
z = np.polyfit(df['Raw_WER'], df['RWER'], 1)
p = np.poly1d(z)
plt.plot(df['Raw_WER'], p(df['Raw_WER']), "r--", alpha=0.8, 
         label=f'Trend line (R² = {np.corrcoef(df["Raw_WER"], df["RWER"])[0,1]**2:.2f})')

plt.xlabel('Raw WER', fontsize=12)
plt.ylabel('Relative WER Improvement (%)', fontsize=12)
plt.title('Relationship between Initial WER and Improvement', fontsize=14, pad=20)

plt.grid(True, linestyle='--', alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig('wer_improvement_correlation.png', dpi=300, bbox_inches='tight')
plt.close()


