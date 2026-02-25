import streamlit as st
import pandas as pd
import os
import glob
import re
import matplotlib.pyplot as plt

# --- Page Configuration ---
st.set_page_config(
    page_title="ASR Post-Processing Analysis",
    page_icon="📊",
    layout="wide"
)

# --- Data Loading and Caching ---

@st.cache_data
def load_experiment_data():
    """
    Scans the 'outputs' directory to find all experiment results,
    parses the metadata from the file paths, and returns a tidy DataFrame.
    """
    all_data = []
    file_paths = glob.glob('outputs/**/evaluation_metrics_full_debug.csv', recursive=True)

    if not file_paths:
        st.error("Fatal: No 'evaluation_metrics_full_debug.csv' files found in the 'outputs' directory. Please ensure the directory structure is correct and contains data.")
        return pd.DataFrame()

    path_pattern = re.compile(r"outputs[/\\](\d+_\w+)[/\\]([\w\d]+)[/\\](\d+)_sentences")

    for path in file_paths:
        match = path_pattern.search(path)
        if match:
            exp_run, sample_id, num_sentences = match.groups()
            
            try:
                exp_id, run_id = exp_run.split('_')
                
                # Try to read CSV with error handling for malformed files
                try:
                    # Use on_bad_lines='skip' to skip malformed lines (pandas >= 1.3)
                    # For older pandas versions, this will fall back to the except block
                    df_metrics = pd.read_csv(path, on_bad_lines='skip')
                except (pd.errors.ParserError, TypeError) as e:
                    # Fallback for older pandas versions or if on_bad_lines parameter isn't supported
                    try:
                        df_metrics = pd.read_csv(path, error_bad_lines=False, warn_bad_lines=False)
                    except (pd.errors.ParserError, TypeError):
                        st.warning(f"Skipping malformed CSV file {path}: {e}")
                        continue
                except UnicodeDecodeError as e:
                    st.warning(f"Skipping file with encoding issues {path}: {e}")
                    continue
                except Exception as e:
                    st.warning(f"Error reading CSV file {path}: {e}")
                    continue

                if not all(col in df_metrics.columns for col in ['Type', 'WER', 'CER', 'SIM']):
                    st.warning(f"Skipping file with incorrect format: {path}")
                    continue
                
                # Filter out debug rows and ensure numeric columns
                df_metrics = df_metrics[~df_metrics['Type'].str.contains('---', na=False)]
                
                # Convert numeric columns to float, handling any non-numeric values
                for col in ['WER', 'CER', 'SIM']:
                    df_metrics[col] = pd.to_numeric(df_metrics[col], errors='coerce')
                
                # Drop rows where any of the numeric columns is NaN
                df_metrics = df_metrics.dropna(subset=['WER', 'CER', 'SIM'])
                
                # Skip files with no valid data after cleaning
                if df_metrics.empty:
                    st.warning(f"No valid data found in {path} after cleaning")
                    continue
                
                df_metrics['exp_id'] = exp_id
                df_metrics['run_id'] = run_id
                df_metrics['sample_id'] = sample_id
                df_metrics['num_sentences'] = int(num_sentences)
                df_metrics['full_path'] = path
                all_data.append(df_metrics)
            except (ValueError, FileNotFoundError) as e:
                st.warning(f"Could not read or process {path}: {e}")
            except Exception as e:
                st.warning(f"An unexpected error occurred while processing {path}: {e}")

    if not all_data:
        st.error("Fatal: No valid data could be loaded. Please check the contents of your 'evaluation_metrics.csv' files.")
        return pd.DataFrame()

    master_df = pd.concat(all_data, ignore_index=True)

    def get_experiment_name(exp_id):
        model_map = {"41": "GPT-4.1"}
        model = model_map.get(exp_id[:2], f"Model_{exp_id[:2]}")
        technique = exp_id[2]
        return f"{model} - Tech {technique}"

    master_df['exp_name'] = master_df['exp_id'].apply(get_experiment_name)
    return master_df

# --- Helper Functions ---

def read_text_file(path):
    """Safely read a text file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"File not found: {path}"
    except Exception as e:
        return f"Error reading file: {e}"

# --- Plotting Functions ---

def plot_performance_vs_sentences(df, selected_experiments, selected_metric, selected_sample):
    """
    Plots a line chart of a selected metric vs. the number of sentences in the prompt.
    This version correctly plots both 'Improved' and 'Raw ASR' as separate lines.
    """
    st.header(f"Performance Analysis: {selected_metric}")
    st.info("Solid lines represent the model-improved text. Dashed lines represent the corresponding Raw ASR baseline for that specific run.")

    # Define yaxis_title at the beginning to use throughout the function
    yaxis_title = f"{selected_metric} Score"
    if selected_metric in ['WER', 'CER']:
        yaxis_title += " (Lower is Better)"
    else: # SIM
        yaxis_title += " (Higher is Better)"

    # 1. Filter data based on user's selection of experiments.
    #    Only show "Examples Removed" versions, not "Full" versions
    plot_df = df[
        (df['exp_name'].isin(selected_experiments)) & 
        (df['Type'].str.contains('Examples Removed', na=False))
    ].copy()

    # 2. Filter by sample if a specific one is chosen
    if selected_sample != 'Overall':
        plot_df = plot_df[plot_df['sample_id'] == selected_sample]
        title = f"{selected_metric} vs. # Sentences for Sample: <b>{selected_sample}</b>"
    else:
        title = f"Average {selected_metric} vs. # Sentences (Aggregated Across All Samples)"

    if plot_df.empty:
        st.warning("No data available for the current selection.")
        return

    # 3. Aggregate data: Group by experiment, sentence count, AND Type.
    #    This calculates the mean for 'Improved' and 'Raw ASR' separately at each point.
    agg_df = plot_df.groupby(['exp_name', 'num_sentences', 'Type'])[selected_metric].mean().reset_index()

    # 4. Create the line plot using matplotlib.
    #    - Different colors for each experiment
    #    - Different line styles for 'Improved' (solid) and 'Raw ASR' (dashed)
    
    plt.figure(figsize=(12, 8))
    
    # Get unique experiments and types
    experiments = agg_df['exp_name'].unique()
    colors = plt.cm.Set1(range(len(experiments)))
    
    for i, exp in enumerate(experiments):
        exp_data = agg_df[agg_df['exp_name'] == exp]
        
        # Plot Improved (solid line)
        improved_data = exp_data[exp_data['Type'].str.contains('Improved')]
        if not improved_data.empty:
            plt.plot(improved_data['num_sentences'], improved_data[selected_metric], 
                    color=colors[i], linestyle='-', marker='o', 
                    label=f"{exp} - Improved", linewidth=2)
        
        # Plot Raw ASR (dashed line)
        raw_data = exp_data[exp_data['Type'].str.contains('Raw ASR')]
        if not raw_data.empty:
            plt.plot(raw_data['num_sentences'], raw_data[selected_metric], 
                    color=colors[i], linestyle='--', marker='s', 
                    label=f"{exp} - Raw ASR", linewidth=2)
    
    plt.xlabel("Number of Example Sentences", fontsize=12)
    plt.ylabel(yaxis_title, fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    st.pyplot(plt)
    
    # Add box plot below the line chart
    st.subheader(f"📊 Distribution Analysis: {selected_metric}")
    st.info("Box plots show the distribution of scores across different runs and samples for each number of example sentences.")
    
    # Create box plot grouped by number of sentences with Raw ASR vs Improved comparison
    plt.figure(figsize=(12, 8))
    
    # Prepare data for matplotlib boxplot
    sentence_counts = sorted(plot_df['num_sentences'].unique())
    
    # Separate data by type
    improved_data = plot_df[plot_df['Type'].str.contains('Improved')]
    raw_data = plot_df[plot_df['Type'].str.contains('Raw ASR')]
    
    # Create positions for box plots
    positions_improved = [x - 0.2 for x in range(len(sentence_counts))]
    positions_raw = [x + 0.2 for x in range(len(sentence_counts))]
    
    # Create box plots for each sentence count
    improved_boxes = []
    raw_boxes = []
    
    for i, count in enumerate(sentence_counts):
        improved_values = improved_data[improved_data['num_sentences'] == count][selected_metric].values
        raw_values = raw_data[raw_data['num_sentences'] == count][selected_metric].values
        
        if len(improved_values) > 0:
            improved_boxes.append(improved_values)
        else:
            improved_boxes.append([])
            
        if len(raw_values) > 0:
            raw_boxes.append(raw_values)
        else:
            raw_boxes.append([])
    
    # Plot box plots
    bp1 = plt.boxplot(improved_boxes, positions=positions_improved, widths=0.3, 
                     patch_artist=True, boxprops=dict(facecolor='lightblue'))
    bp2 = plt.boxplot(raw_boxes, positions=positions_raw, widths=0.3, 
                     patch_artist=True, boxprops=dict(facecolor='lightcoral'))
    
    plt.xlabel("Number of Example Sentences", fontsize=12)
    plt.ylabel(yaxis_title, fontsize=12)
    plt.title(f"{selected_metric} Score Distribution by Number of Example Sentences", 
              fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(range(len(sentence_counts)), sentence_counts)
    
    # Add legend
    plt.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Improved', 'Raw ASR'], loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    st.pyplot(plt)
    
    # Add bar chart below the box plot
    st.subheader(f"📊 Bar Chart Analysis: {selected_metric}")
    st.info("Bar charts show the average performance for each number of example sentences.")
    
    # Create bar chart grouped by number of sentences
    plt.figure(figsize=(12, 8))
    
    # Get aggregated data for bar chart
    bar_data = plot_df.groupby(['num_sentences', 'Type'])[selected_metric].mean().reset_index()
    
    # Separate data by type
    improved_bar_data = bar_data[bar_data['Type'].str.contains('Improved')]
    raw_bar_data = bar_data[bar_data['Type'].str.contains('Raw ASR')]
    
    # Create positions for bars
    x_pos = range(len(sentence_counts))
    width = 0.35
    
    # Create bars
    improved_values = [improved_bar_data[improved_bar_data['num_sentences'] == count][selected_metric].iloc[0] 
                      if not improved_bar_data[improved_bar_data['num_sentences'] == count].empty else 0 
                      for count in sentence_counts]
    raw_values = [raw_bar_data[raw_bar_data['num_sentences'] == count][selected_metric].iloc[0] 
                 if not raw_bar_data[raw_bar_data['num_sentences'] == count].empty else 0 
                 for count in sentence_counts]
    
    plt.bar([x - width/2 for x in x_pos], improved_values, width, 
           label='Improved', color='lightblue', alpha=0.8)
    plt.bar([x + width/2 for x in x_pos], raw_values, width, 
           label='Raw ASR', color='lightcoral', alpha=0.8)
    
    plt.xlabel("Number of Example Sentences", fontsize=12)
    plt.ylabel(yaxis_title, fontsize=12)
    plt.title(f"Average {selected_metric} Score by Number of Example Sentences", 
              fontsize=14, fontweight='bold')
    
    # Set x-axis labels
    plt.xticks(x_pos, sentence_counts)
    
    # Add value labels on bars
    for i, (imp_val, raw_val) in enumerate(zip(improved_values, raw_values)):
        if imp_val > 0:
            plt.text(i - width/2, imp_val + max(improved_values + raw_values) * 0.01, 
                    f'{imp_val:.3f}', ha='center', va='bottom', fontsize=10)
        if raw_val > 0:
            plt.text(i + width/2, raw_val + max(improved_values + raw_values) * 0.01, 
                    f'{raw_val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    st.pyplot(plt)
    
    # Add WER improvement chart by sentence count and technique
    if selected_metric == 'WER':
        st.subheader("📈 WER Improvement by Sentence Count and Technique")
        st.info("Shows the relative WER improvement (%) for each technique at different sentence counts.")
        
        # Calculate WER improvements for each experiment and sentence count
        improvement_data = []
        
        for exp_name in selected_experiments:
            exp_data = plot_df[plot_df['exp_name'] == exp_name]
            
            for sentence_count in sentence_counts:
                sent_data = exp_data[exp_data['num_sentences'] == sentence_count]
                
                raw_wer = sent_data[sent_data['Type'].str.contains('Raw ASR')]['WER'].mean()
                improved_wer = sent_data[sent_data['Type'].str.contains('Improved')]['WER'].mean()
                
                if pd.notna(raw_wer) and pd.notna(improved_wer) and raw_wer > 0:
                    wer_improvement = ((raw_wer - improved_wer) / raw_wer) * 100
                    improvement_data.append({
                        'Experiment': exp_name,
                        'Sentence Count': sentence_count,
                        'WER Improvement (%)': wer_improvement,
                        'Raw WER': raw_wer,
                        'Improved WER': improved_wer
                    })
        
        if improvement_data:
            improvement_df = pd.DataFrame(improvement_data)
            
            # Create grouped bar chart
            plt.figure(figsize=(14, 8))
            
            # Get unique experiments and sentence counts
            experiments = improvement_df['Experiment'].unique()
            sentence_counts_avail = sorted(improvement_df['Sentence Count'].unique())
            
            # Set up positions for grouped bars
            x = range(len(sentence_counts_avail))
            width = 0.8 / len(experiments)
            colors = plt.cm.Set1(range(len(experiments)))
            
            # Create bars for each experiment
            for i, exp in enumerate(experiments):
                exp_data = improvement_df[improvement_df['Experiment'] == exp]
                improvements = []
                
                for count in sentence_counts_avail:
                    exp_count_data = exp_data[exp_data['Sentence Count'] == count]
                    if not exp_count_data.empty:
                        improvements.append(exp_count_data['WER Improvement (%)'].iloc[0])
                    else:
                        improvements.append(0)
                
                positions = [pos + i * width for pos in x]
                bars = plt.bar(positions, improvements, width, 
                             label=exp, color=colors[i], alpha=0.8)
                
                # Add value labels on bars
                for j, (pos, val) in enumerate(zip(positions, improvements)):
                    if val > 0:
                        plt.text(pos, val + max(improvement_df['WER Improvement (%)']) * 0.01, 
                               f'{val:.1f}%', ha='center', va='bottom', fontsize=9, rotation=0)
            
            plt.xlabel("Number of Example Sentences", fontsize=12)
            plt.ylabel("WER Improvement (%)", fontsize=12)
            plt.title("WER Improvement (%) by Sentence Count and Technique", fontsize=14, fontweight='bold')
            
            # Set x-axis labels
            plt.xticks([pos + width * (len(experiments) - 1) / 2 for pos in x], sentence_counts_avail)
            
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3, axis='y')
            plt.tight_layout()
            
            st.pyplot(plt)
            
            # Show detailed table
            st.subheader("📊 Detailed WER Improvement Table")
            
            # Pivot table for better display
            pivot_df = improvement_df.pivot(index='Sentence Count', 
                                          columns='Experiment', 
                                          values='WER Improvement (%)')
            
            # Format the dataframe
            pivot_df = pivot_df.round(2)
            pivot_df = pivot_df.fillna('-')
            
            st.dataframe(pivot_df, use_container_width=True)
            
        else:
            st.warning("No WER improvement data available for the selected experiments and sentence counts.")


def display_text_comparison(df, data_root="."):
    """
    Displays a side-by-side comparison of GT, Raw ASR, and Improved text.
    """
    st.header("Detailed Text Comparison")
    
    available_exps = sorted(df[df['Type'].str.contains('Improved', na=False)]['exp_name'].unique())
    available_samples = sorted(df['sample_id'].unique())

    # Set default experiment to GPT-4.1 - Tech 4
    default_experiment = "GPT-4.1 - Tech 4"
    default_exp_index = available_exps.index(default_experiment) if default_experiment in available_exps else 0

    st.info("Select a specific experiment, sample, run, and sentence count to compare the text outputs.")
    col1, col2 = st.columns(2)
    with col1:
        sel_exp = st.selectbox("Select Experiment", available_exps, index=default_exp_index, key="text_exp")
    with col2:
        sel_sample = st.selectbox("Select Sample", available_samples, key="text_sample")

    filtered_df = df[(df['exp_name'] == sel_exp) & (df['sample_id'] == sel_sample) & (df['Type'].str.contains('Improved.*Examples Removed', na=False))]

    if filtered_df.empty:
        st.warning(f"No data found for {sel_exp} on sample {sel_sample}.")
        return

    col3, col4 = st.columns(2)
    with col3:
        sel_run = st.selectbox("Select Run", sorted(filtered_df['run_id'].unique()), key="text_run")
    with col4:
        available_sentences = sorted(filtered_df['num_sentences'].unique())
        default_sentence_count = 6 if 6 in available_sentences else available_sentences[0]
        default_sentence_index = available_sentences.index(default_sentence_count)
        sel_sentences = st.selectbox("Select # Sentences", available_sentences, index=default_sentence_index, key="text_sentences")
    
    exp_id = filtered_df['exp_id'].iloc[0]
    base_path = os.path.join(data_root, 'outputs', f"{exp_id}_{sel_run}", sel_sample, f"{sel_sentences}_sentences")

    # The ground truth and original ASR paths are constant for a given sample
    gt_path = os.path.join(data_root, 'data', 'ground_truth', f"{sel_sample}.txt")
    raw_asr_path = os.path.join(data_root, 'data', 'asr_raw', f"{sel_sample} ASR.txt")

    # The improved text and prompt are specific to the experiment run
    improved_path = os.path.join(base_path, 'IMPROVED_FULL.txt')
    prompt_path = os.path.join(base_path, 'PROMPT_EXAMPLES.txt')

    gt_text = read_text_file(gt_path)
    raw_asr_text = read_text_file(raw_asr_path)
    improved_text = read_text_file(improved_path)
    prompt_text = read_text_file(prompt_path)

    st.markdown("---")
    res_col1, res_col2, res_col3 = st.columns(3)
    with res_col1:
        st.markdown("##### 📜 Ground Truth")
        st.text_area("GT", gt_text, height=400, key='gt_text_area')
    with res_col2:
        st.markdown("##### 🤖 Raw ASR")
        st.text_area("Raw ASR", raw_asr_text, height=400, key='asr_text_area')
    with res_col3:
        st.markdown("##### ✨ Model Improved")
        st.text_area("Improved Text", improved_text, height=400, key='imp_text_area')

    with st.expander("Show Prompt Examples Used for This Run"):
        st.code(prompt_text, language='text')

def display_average_improvements(df):
    """
    Displays average WER improvement and semantic similarity increase across experiments.
    """
    st.header("📊 Average Improvements Summary")
    
    # Add filters for experiment and sentence count
    exp_names = sorted(df['exp_name'].unique())
    sentence_counts = sorted(df['num_sentences'].unique())
    
    # Set defaults to GPT-4.1 - Tech 4 and 6 sentences
    default_experiment = "GPT-4.1 - Tech 4"
    default_exp_selection = [default_experiment] if default_experiment in exp_names else exp_names[:1]
    default_sentence_count = 6 if 6 in sentence_counts else sentence_counts[0]
    
    col1, col2 = st.columns(2)
    with col1:
        selected_experiments = st.multiselect(
            "Select Experiments to Analyze",
            exp_names,
            default=default_exp_selection,
            key="improvements_exp_select"
        )
    with col2:
        selected_sentence_count = st.selectbox(
            "Select Number of Examples",
            sentence_counts,
            index=sentence_counts.index(default_sentence_count) if default_sentence_count in sentence_counts else 0,
            key="improvements_sentence_select"
        )
    
    # Filter dataframe based on selections
    filtered_df = df[
        (df['exp_name'].isin(selected_experiments)) &
        (df['num_sentences'] == selected_sentence_count)
    ]
    
    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
        return
    
    # Calculate improvements for selected experiments and sentence count
    improvements_data = []
    
    for exp_name in df['exp_name'].unique():
        exp_df = df[df['exp_name'] == exp_name]
        
        # Get raw and improved metrics - only use "Examples Removed" versions
        raw_df = exp_df[exp_df['Type'].str.contains('Raw ASR.*Examples Removed', na=False)]
        improved_df = exp_df[exp_df['Type'].str.contains('Improved.*Examples Removed', na=False)]
        
        if raw_df.empty or improved_df.empty:
            continue
            
        # Calculate averages across all samples and sentence counts
        avg_raw_wer = raw_df['WER'].mean()
        avg_improved_wer = improved_df['WER'].mean()
        avg_raw_sim = raw_df['SIM'].mean()
        avg_improved_sim = improved_df['SIM'].mean()
        
        # Calculate relative improvements
        wer_improvement = ((avg_raw_wer - avg_improved_wer) / avg_raw_wer) * 100 if avg_raw_wer > 0 else 0
        sim_increase = avg_improved_sim - avg_raw_sim
        sim_relative_increase = ((avg_improved_sim - avg_raw_sim) / avg_raw_sim) * 100 if avg_raw_sim > 0 else 0
        
        improvements_data.append({
            'Experiment': exp_name,
            'Avg Raw WER': f"{avg_raw_wer:.3f}",
            'Avg Improved WER': f"{avg_improved_wer:.3f}",
            'WER Improvement (%)': f"{wer_improvement:.1f}",
            'Avg Raw SIM': f"{avg_raw_sim:.1f}",
            'Avg Improved SIM': f"{avg_improved_sim:.1f}",
            'SIM Increase (abs)': f"{sim_increase:.1f}",
            'SIM Increase (%)': f"{sim_relative_increase:.1f}"
        })
    
    if not improvements_data:
        st.warning("No data available for improvements calculation.")
        return
    
    improvements_df = pd.DataFrame(improvements_data)
    
    # Display summary table
    st.subheader("🎯 Overall Performance Summary")
    st.dataframe(improvements_df, use_container_width=True)
    
    # Display key metrics in columns
    st.subheader("🏆 Key Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_wer_exp = improvements_df.loc[improvements_df['WER Improvement (%)'].str.rstrip('%').astype(float).idxmax()]
        st.metric(
            "Best WER Improvement",
            f"{best_wer_exp['WER Improvement (%)']}%",
            f"{best_wer_exp['Experiment']}"
        )
    
    with col2:
        best_sim_exp = improvements_df.loc[improvements_df['SIM Increase (abs)'].str.rstrip('').astype(float).idxmax()]
        st.metric(
            "Best SIM Increase",
            f"{best_sim_exp['SIM Increase (abs)']} points",
            f"{best_sim_exp['Experiment']}"
        )
    
    with col3:
        avg_wer_improvement = improvements_df['WER Improvement (%)'].str.rstrip('%').astype(float).mean()
        st.metric(
            "Average WER Improvement",
            f"{avg_wer_improvement:.1f}%",
            "Across all experiments"
        )
    
    # Detailed breakdown by sentence count
    st.subheader("📈 Performance by Number of Examples")
    
    sentence_breakdown = []
    for num_sentences in sorted(df['num_sentences'].unique()):
        sent_df = df[df['num_sentences'] == num_sentences]
        
        raw_df = sent_df[sent_df['Type'].str.contains('Raw ASR.*Examples Removed', na=False)]
        improved_df = sent_df[sent_df['Type'].str.contains('Improved.*Examples Removed', na=False)]
        
        if raw_df.empty or improved_df.empty:
            continue
            
        avg_raw_wer = raw_df['WER'].mean()
        avg_improved_wer = improved_df['WER'].mean()
        avg_raw_sim = raw_df['SIM'].mean()
        avg_improved_sim = improved_df['SIM'].mean()
        
        wer_improvement = ((avg_raw_wer - avg_improved_wer) / avg_raw_wer) * 100 if avg_raw_wer > 0 else 0
        sim_increase = avg_improved_sim - avg_raw_sim
        
        sentence_breakdown.append({
            'Examples': f"{num_sentences} sentences",
            'WER Improvement': f"{wer_improvement:.1f}%",
            'SIM Increase': f"{sim_increase:.1f} points",
            'Raw WER': f"{avg_raw_wer:.3f}",
            'Improved WER': f"{avg_improved_wer:.3f}",
            'Raw SIM': f"{avg_raw_sim:.1f}",
            'Improved SIM': f"{avg_improved_sim:.1f}"
        })
    
    if sentence_breakdown:
        sentence_df = pd.DataFrame(sentence_breakdown)
        st.dataframe(sentence_df, use_container_width=True)
    
    # Visual charts
    st.subheader("📊 Improvement Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # WER Improvement Chart
        wer_data = improvements_df[['Experiment', 'WER Improvement (%)']].copy()
        wer_data['WER Improvement (%)'] = wer_data['WER Improvement (%)'].str.rstrip('%').astype(float)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(wer_data['Experiment'], wer_data['WER Improvement (%)'], 
                      color='green', alpha=0.7)
        plt.xlabel("Experiment", fontsize=12)
        plt.ylabel("WER Improvement (%)", fontsize=12)
        plt.title("WER Improvement by Experiment", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(plt)
    
    with col2:
        # SIM Increase Chart
        sim_data = improvements_df[['Experiment', 'SIM Increase (abs)']].copy()
        sim_data['SIM Increase (abs)'] = sim_data['SIM Increase (abs)'].str.rstrip('').astype(float)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(sim_data['Experiment'], sim_data['SIM Increase (abs)'], 
                      color='blue', alpha=0.7)
        plt.xlabel("Experiment", fontsize=12)
        plt.ylabel("Semantic Similarity Increase", fontsize=12)
        plt.title("Semantic Similarity Increase by Experiment", fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(plt)

# --- Main Application ---
def main():
    st.title("🔬 ASR Post-Processing Experiment Visualizer")

    master_df = load_experiment_data()

    if master_df.empty:
        st.stop()

    st.sidebar.title("⚙️ Controls")
    
    view_selection = st.sidebar.radio(
        "Select View",
        ["Performance vs. # Sentences", "Average Improvements Summary", "Detailed Text Comparison"]
    )
    
    st.sidebar.markdown("---")
    
    if view_selection == "Performance vs. # Sentences":
        exp_names = sorted(master_df['exp_name'].unique())
        
        # Set default to GPT-4.1 technique 4 only
        default_experiment = "GPT-4.1 - Tech 4"
        default_selection = [default_experiment] if default_experiment in exp_names else []
        
        selected_experiments = st.sidebar.multiselect(
            "Select Experiments to Compare",
            exp_names,
            default=default_selection
        )
        
        selected_metric = st.sidebar.radio(
            "Select Metric",
            ['WER', 'CER', 'SIM'],
            horizontal=True
        )
        
        sample_ids = ['Overall'] + sorted(master_df['sample_id'].unique())
        selected_sample = st.sidebar.selectbox(
            "Select a Specific Sample (or Overall)",
            sample_ids
        )

        if selected_experiments:
            plot_performance_vs_sentences(master_df, selected_experiments, selected_metric, selected_sample)
        else:
            st.info("Please select at least one experiment from the sidebar to see the results.")

    elif view_selection == "Average Improvements Summary":
        display_average_improvements(master_df)

    elif view_selection == "Detailed Text Comparison":
        display_text_comparison(master_df)

if __name__ == "__main__":
    main()