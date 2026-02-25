import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import glob
import numpy as np
from pathlib import Path
import re

# Page configuration
st.set_page_config(
    page_title="ASR Improvement Dashboard",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .metric-title {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        opacity: 0.8;
    }
    
    .improvement-positive {
        color: #00ff88;
        font-weight: bold;
    }
    
    .improvement-negative {
        color: #ff6b6b;
        font-weight: bold;
    }
    
    .data-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    .stSelectbox > div > div {
        background-color: white;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=60)  # Cache for 60 seconds to allow auto-detection of new experiments
def load_all_data():
    """Load and combine all evaluation data from the new nested outputs directory structure."""
    processed_data_list = []
    # New pattern for the nested structure:
    # outputs/{experiment_id}/{file_base}/{num_sentences}_sentences/evaluation_metrics.csv
    outputs_pattern = "outputs/*/*/*/evaluation_metrics.csv"
    csv_files = glob.glob(outputs_pattern)
    
    found_experiments = set()
    
    for csv_file_path in csv_files:
        try:
            path = Path(csv_file_path)
            parts = path.parts # e.g., ('outputs', 'exp1', 'fileA', '10_sentences', 'evaluation_metrics.csv')
            
            if len(parts) < 5:
                st.warning(f"Skipping unexpected file path structure: {csv_file_path}")
                continue

            experiment_id = parts[1]
            file_base = parts[2]
            num_sents_dir = parts[3]
            
            found_experiments.add(experiment_id)

            try:
                num_example_sentences = int(num_sents_dir.split('_')[0])
            except ValueError:
                st.warning(f"Could not parse num_example_sentences from {num_sents_dir} in {csv_file_path}")
                continue

            df_eval_metrics = pd.read_csv(csv_file_path)
            
            if 'Type' not in df_eval_metrics.columns or len(df_eval_metrics) != 2:
                st.warning(f"Skipping malformed evaluation_metrics.csv: {csv_file_path}. Expected 'Type' column and 2 rows.")
                continue

            raw_metrics_row = df_eval_metrics[df_eval_metrics['Type'] == 'Raw ASR']
            improved_metrics_row = df_eval_metrics[df_eval_metrics['Type'] == 'Improved']

            if raw_metrics_row.empty or improved_metrics_row.empty:
                st.warning(f"Could not find 'Raw ASR' or 'Improved' types in {csv_file_path}")
                continue
            
            raw_metrics_data = raw_metrics_row.iloc[0]
            improved_metrics_data = improved_metrics_row.iloc[0]

            record = {
                'experiment': experiment_id,
                'filename': file_base,  # Using 'filename' to map file_base for compatibility
                'num_example_sentences': num_example_sentences,
                'WER_ASR': raw_metrics_data.get('WER', np.nan),
                'CER_ASR': raw_metrics_data.get('CER', np.nan),
                'S_ASR': raw_metrics_data.get('SIM', np.nan), # CSV uses 'SIM' for similarity
                'WER_IMP': improved_metrics_data.get('WER', np.nan),
                'CER_IMP': improved_metrics_data.get('CER', np.nan),
                'S_IMP': improved_metrics_data.get('SIM', np.nan)  # CSV uses 'SIM' for similarity
            }
            processed_data_list.append(record)
            
        except Exception as e:
            st.error(f"Error processing file {csv_file_path}: {e}")
    
    if found_experiments:
        st.session_state['found_experiments'] = sorted(list(found_experiments))

    if not processed_data_list:
        return pd.DataFrame()
        
    combined_df = pd.DataFrame(processed_data_list)
    return combined_df

def calculate_improvement_stats(df):
    """Calculate improvement percentages and statistics using integrated ASR metrics."""
    if df.empty:
        return df

    df_copy = df.copy() # Work on a copy to avoid SettingWithCopyWarning

    # Ensure required columns for calculation exist, fill with NaN if not to prevent errors
    metric_cols_to_check = ['WER_ASR', 'CER_ASR', 'S_ASR', 'WER_IMP', 'CER_IMP', 'S_IMP']
    for col in metric_cols_to_check:
        if col not in df_copy.columns:
            df_copy[col] = np.nan
            
            # Calculate percentage improvements
    # For WER and CER, lower is better: (ASR - IMP) / ASR * 100
    df_copy['WER_improvement_pct'] = np.where(
        (df_copy['WER_ASR'].notna()) & (df_copy['WER_ASR'] > 0) & df_copy['WER_IMP'].notna(),
        ((df_copy['WER_ASR'] - df_copy['WER_IMP']) / df_copy['WER_ASR'] * 100),
        np.nan 
    )
    df_copy['CER_improvement_pct'] = np.where(
        (df_copy['CER_ASR'].notna()) & (df_copy['CER_ASR'] > 0) & df_copy['CER_IMP'].notna(),
        ((df_copy['CER_ASR'] - df_copy['CER_IMP']) / df_copy['CER_ASR'] * 100),
        np.nan
    )
    
    # For Semantic Similarity (S_IMP), higher is better: (IMP - ASR) / ASR * 100
    # Handle S_ASR == 0 cases carefully
    conditions_s_imp = [
        (df_copy['S_ASR'].notna()) & (df_copy['S_IMP'].notna()) & (df_copy['S_ASR'] > 0),                # Standard case: S_ASR > 0
        (df_copy['S_ASR'].notna()) & (df_copy['S_IMP'].notna()) & (df_copy['S_ASR'] == 0) & (df_copy['S_IMP'] > 0), # S_ASR is 0, S_IMP is positive
        (df_copy['S_ASR'].notna()) & (df_copy['S_IMP'].notna()) & (df_copy['S_ASR'] == 0) & (df_copy['S_IMP'] == 0) # Both S_ASR and S_IMP are 0
    ]
    choices_s_imp = [
        ((df_copy['S_IMP'] - df_copy['S_ASR']) / df_copy['S_ASR'] * 100),
        100.0,  # Max improvement if starting from 0 to positive
        0.0     # No improvement if both are 0
    ]
    df_copy['S_improvement_pct'] = np.select(conditions_s_imp, choices_s_imp, default=np.nan)

    # Store original ASR values for reference/compatibility with charts expecting 'WER_original', etc.
    df_copy['WER_original'] = df_copy['WER_ASR']
    df_copy['CER_original'] = df_copy['CER_ASR']
    df_copy['S_original'] = df_copy['S_ASR']
    
    return df_copy

def create_metric_card(title, value, improvement=None, format_str=".3f"):
    """Create a modern metric card"""
    if improvement is not None:
        improvement_class = "improvement-positive" if improvement > 0 else "improvement-negative"
        improvement_text = f"<div class='metric-delta'><span class='{improvement_class}'>{'↗' if improvement > 0 else '↘'} {improvement:+.1f}%</span></div>"
    else:
        improvement_text = ""
    
    return f"""
    <div class="metric-container">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value:{format_str}}</div>
        {improvement_text}
    </div>
    """

def create_comparison_charts(df):
    """Create comprehensive comparison charts"""
    charts = {}
    
    # Prepare data for plotting - side by side experiments
    plot_data = []
    processed_originals = set()  # Track which files we've added originals for
    
    for _, row in df.iterrows():
        base_data = {
            'filename': row['filename'],
            'experiment': row['experiment'],
            'sentences': row['num_example_sentences']
        }
        
        # Add improved metrics for each experiment
        plot_data.append({**base_data, 'metric': 'WER', 'value': row['WER_IMP'], 'type': 'Improved'})
        plot_data.append({**base_data, 'metric': 'CER', 'value': row['CER_IMP'], 'type': 'Improved'})
        plot_data.append({**base_data, 'metric': 'Semantic Similarity', 'value': row['S_IMP'], 'type': 'Improved'})
        
        # Add original metrics only ONCE per filename (not per experiment)
        filename = row['filename']
        if filename not in processed_originals and 'WER_original' in row and not pd.isna(row['WER_original']):
            # Use "Original ASR" as experiment name for raw metrics
            original_base_data = {
                'filename': filename,
                'experiment': 'Original ASR',
                'sentences': row['num_example_sentences']
            }
            plot_data.append({**original_base_data, 'metric': 'WER', 'value': row['WER_original'], 'type': 'Original'})
            plot_data.append({**original_base_data, 'metric': 'CER', 'value': row['CER_original'], 'type': 'Original'})
            plot_data.append({**original_base_data, 'metric': 'Semantic Similarity', 'value': row['S_original'], 'type': 'Original'})
            processed_originals.add(filename)
    
    plot_df = pd.DataFrame(plot_data)
    
    # WER Comparison - Side by Side
    wer_data = plot_df[plot_df['metric'] == 'WER']
    charts['wer'] = px.bar(
        wer_data, 
        x='filename', 
        y='value', 
        color='experiment',
        barmode='group',
        title="Word Error Rate Comparison - All Experiments",
        labels={'value': 'WER', 'filename': 'File Base', 'experiment': 'Experiment'},
        height=500
    )
    charts['wer'].update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80)
    )
    
    # CER Comparison - Side by Side
    cer_data = plot_df[plot_df['metric'] == 'CER']
    charts['cer'] = px.bar(
        cer_data, 
        x='filename', 
        y='value', 
        color='experiment',
        barmode='group',
        title="Character Error Rate Comparison - All Experiments",
        labels={'value': 'CER', 'filename': 'File Base', 'experiment': 'Experiment'},
        height=500
    )
    charts['cer'].update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80)
    )
    
    # Semantic Similarity Comparison - Side by Side
    sem_data = plot_df[plot_df['metric'] == 'Semantic Similarity']
    charts['semantic'] = px.bar(
        sem_data, 
        x='filename', 
        y='value', 
        color='experiment',
        barmode='group',
        title="Semantic Similarity Comparison - All Experiments",
        labels={'value': 'Semantic Similarity (%)', 'filename': 'File Base', 'experiment': 'Experiment'},
        height=500
    )
    charts['semantic'].update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=80)
    )
    
    return charts

def create_improvement_analysis(df):
    """Create improvement analysis charts"""
    if 'WER_improvement_pct' not in df.columns:
        return None
    
    # Average improvements by experiment
    exp_improvements = df.groupby('experiment').agg({
        'WER_improvement_pct': 'mean',
        'CER_improvement_pct': 'mean',
        'S_improvement_pct': 'mean'
    }).reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='WER Improvement',
        x=exp_improvements['experiment'],
        y=exp_improvements['WER_improvement_pct'],
        marker_color='#ff6b6b'
    ))
    
    fig.add_trace(go.Bar(
        name='CER Improvement',
        x=exp_improvements['experiment'],
        y=exp_improvements['CER_improvement_pct'],
        marker_color='#4ecdc4'
    ))
    
    fig.add_trace(go.Bar(
        name='Semantic Improvement',
        x=exp_improvements['experiment'],
        y=exp_improvements['S_improvement_pct'],
        marker_color='#45b7d1'
    ))
    
    fig.update_layout(
        title='Average Improvement by Experiment',
        xaxis_title='Experiment',
        yaxis_title='Improvement (%)',
        barmode='group',
        height=400
    )
    
    return fig

def main():
    # Header
    st.markdown('<h1 class="main-header">🎙️ ASR Improvement Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    with st.spinner("🔄 Loading evaluation data..."):
        df = load_all_data()
    
    if df.empty:
        st.error("❌ No evaluation data found. Please ensure you have CSV files in the outputs directory.")
        st.info("Expected structure: `outputs/experiment_id/file_base/N_sentences/evaluation_metrics.csv`")
        return
    
    # Determine if raw ASR data is available for improvement calculations
    has_raw_data = all(col in df.columns for col in ['WER_ASR', 'CER_ASR', 'S_ASR']) and \
                   not df[['WER_ASR', 'CER_ASR', 'S_ASR']].isnull().all().all()

    # Calculate improvements - this function now also adds 'original' columns
    df = calculate_improvement_stats(df)
    
    if not has_raw_data:
        st.warning("⚠️ Original ASR metrics (WER_ASR, CER_ASR, S_ASR) are missing or incomplete in the loaded data. Improvement calculations will be limited or unavailable.")
        # Ensure improvement percentage columns exist with NaN if not calculable
        for col_pct in ['WER_improvement_pct', 'CER_improvement_pct', 'S_improvement_pct']:
            if col_pct not in df.columns:
                df[col_pct] = np.nan
    
    # Sidebar
    st.sidebar.title("🎛️ Controls")
    
    # Refresh button to clear cache and detect new experiments
    if st.sidebar.button("🔄 Refresh Data", help="Refresh to detect new experiments"):
        st.cache_data.clear()
        st.rerun()
    
    # Filters
    st.sidebar.subheader("📊 Filters")
    
    # File filter
    files = sorted(df['filename'].unique())
    selected_files = st.sidebar.multiselect(
        "Select File Bases (Samples)",
        files,
        default=files,
        help="Choose which audio files to analyze"
    )
    
    # Sentence count filter (single selection only)
    sentence_counts = sorted(df['num_example_sentences'].unique())
    selected_sentences = st.sidebar.selectbox(
        "Example Sentences",
        sentence_counts,
        index=0,
        help="Number of example sentences used in improvement (select one at a time)"
    )
    
    # Add a new column for experiment type (strip _runN, preserve underscores)
    def get_base_experiment(exp):
        return re.sub(r'_run\d+$', '', exp)
    df['experiment_type'] = df['experiment'].apply(get_base_experiment)

    # Use experiment_type for all selectors and plots
    experiment_types = sorted(df['experiment_type'].unique())
    selected_types = st.sidebar.multiselect("Experiment Types", experiment_types, default=experiment_types, key="exp_types_selector")

    # Filter by experiment_type and file
    filtered_df_by_experiment_and_file = df[
        (df['experiment_type'].isin(selected_types)) &
        (df['filename'].isin(selected_files))
    ]
    # For Comparisons tab only - apply sentence count filter
    filtered_df_for_comparisons = filtered_df_by_experiment_and_file[
        (filtered_df_by_experiment_and_file['num_example_sentences'] == selected_sentences)
    ]
    if filtered_df_by_experiment_and_file.empty:
        st.warning("⚠️ No data matches the selected experiments and files.")
        return
    if filtered_df_for_comparisons.empty:
        st.warning("⚠️ No data matches the selected sentence count filter for comparisons.")
        filtered_df_for_comparisons = filtered_df_by_experiment_and_file
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Data Summary")
    st.sidebar.metric("Total Records (All Sentences)", len(filtered_df_by_experiment_and_file))
    st.sidebar.metric("Records (Selected Sentences)", len(filtered_df_for_comparisons))
    st.sidebar.metric("Audio Files", len(filtered_df_by_experiment_and_file['filename'].unique()))
    st.sidebar.metric("Experiments", len(filtered_df_by_experiment_and_file['experiment'].unique()))
    st.sidebar.metric("Sentence Configurations", len(filtered_df_by_experiment_and_file['num_example_sentences'].unique()))
    
    # Show detected experiments
    if 'found_experiments' in st.session_state:
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔍 Detected Experiments")
        for exp in st.session_state['found_experiments']:
            st.sidebar.markdown(f"• {exp}")
    
    # Data validation warnings
    if not filtered_df_by_experiment_and_file.empty:
        nan_issues = []
        if filtered_df_by_experiment_and_file['WER_IMP'].isna().any():
            nan_issues.append("WER_IMP")
        if filtered_df_by_experiment_and_file['CER_IMP'].isna().any():
            nan_issues.append("CER_IMP") 
        if filtered_df_by_experiment_and_file['S_IMP'].isna().any():
            nan_issues.append("S_IMP")
        
        if nan_issues:
            st.sidebar.markdown("---")
            st.sidebar.subheader("⚠️ Data Issues")
            st.sidebar.warning(f"Found NaN values in: {', '.join(nan_issues)}")
            st.sidebar.info("Check your CSV files for missing data")
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    # Summary metrics (handle NaN values) - using ALL sentence counts
    with col1:
        avg_wer = filtered_df_by_experiment_and_file['WER_IMP'].mean() if not filtered_df_by_experiment_and_file['WER_IMP'].isna().all() else 0
        wer_improvement = filtered_df_by_experiment_and_file['WER_improvement_pct'].mean() if has_raw_data and not filtered_df_by_experiment_and_file['WER_improvement_pct'].isna().all() else None
        st.markdown(create_metric_card("Average WER", avg_wer, wer_improvement), unsafe_allow_html=True)
    
    with col2:
        avg_cer = filtered_df_by_experiment_and_file['CER_IMP'].mean() if not filtered_df_by_experiment_and_file['CER_IMP'].isna().all() else 0
        cer_improvement = filtered_df_by_experiment_and_file['CER_improvement_pct'].mean() if has_raw_data and not filtered_df_by_experiment_and_file['CER_improvement_pct'].isna().all() else None
        st.markdown(create_metric_card("Average CER", avg_cer, cer_improvement), unsafe_allow_html=True)
    
    with col3:
        avg_semantic = filtered_df_by_experiment_and_file['S_IMP'].mean() if not filtered_df_by_experiment_and_file['S_IMP'].isna().all() else 0
        sem_improvement = filtered_df_by_experiment_and_file['S_improvement_pct'].mean() if has_raw_data and not filtered_df_by_experiment_and_file['S_improvement_pct'].isna().all() else None
        st.markdown(create_metric_card("Semantic Similarity", avg_semantic, sem_improvement, ".1f"), unsafe_allow_html=True)
    
    with col4:
        # Handle NaN values when finding best experiment - using ALL sentence counts
        if not filtered_df_by_experiment_and_file.empty and not filtered_df_by_experiment_and_file['S_IMP'].isna().all():
            # Only consider rows with valid S_IMP values
            valid_data = filtered_df_by_experiment_and_file.dropna(subset=['S_IMP'])
            if not valid_data.empty:
                best_experiment = valid_data.loc[valid_data['S_IMP'].idxmax(), 'experiment']
                best_score = valid_data['S_IMP'].max()
            else:
                best_experiment = "No valid data"
                best_score = 0
        else:
            best_experiment = "No data available"
            best_score = 0
            
        st.markdown(f"""
        <div class="metric-container">
            <div class="metric-title">Best Experiment</div>
            <div class="metric-value" style="font-size: 1.5rem;">{best_experiment}</div>
            <div class="metric-delta">Highest semantic similarity: {best_score:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Experiment Type Trends (All Runs Mixed)", "📈 Detailed Comparisons", "🔍 Analysis", "📋 Data"])
    
    with tab1:
        st.subheader("📊 Experiment Type Trends (All Runs Mixed)")
        show_dots = st.checkbox("Show Dots", value=True, key="show_dots_boxplot")
        boxpoints_mode = "all" if show_dots else False
        metric = st.selectbox("Metric", ["WER_IMP", "CER_IMP", "S_IMP"], index=0, key="metric_boxplot")
        experiment_types = sorted(filtered_df_by_experiment_and_file['experiment_type'].unique())
        selected_types = st.multiselect("Experiment Types", experiment_types, default=experiment_types, key="exp_types_boxplot")
        for exp_type in selected_types:
            exp_df = filtered_df_by_experiment_and_file[filtered_df_by_experiment_and_file['experiment_type'] == exp_type]
            fig = go.Figure()
            colors = {"Improved": "#1f77b4", "Raw ASR": "#ff7f0e"}
            num_sentences_sorted = sorted(exp_df['num_example_sentences'].unique())
            for metric_label, metric_col in [("Improved", metric), ("Raw ASR", metric.replace("_IMP", "_original"))]:
                if metric_col not in exp_df.columns:
                    continue
                show_legend = True
                for n_sent in num_sentences_sorted:
                    y_vals = exp_df.loc[exp_df['num_example_sentences'] == n_sent, metric_col]
                    if y_vals.empty:
                        continue
                    fig.add_trace(go.Box(
                        y=y_vals,
                        x=[n_sent] * len(y_vals),
                        name=f"{metric_label}",
                        marker_color=colors[metric_label],
                        boxpoints=boxpoints_mode,
                        jitter=0.3,
                        pointpos=0,
                        legendgroup=metric_label,
                        showlegend=show_legend,
                        width=0.5
                    ))
                    show_legend = False
            fig.update_layout(
                title=f"{exp_type} · {metric.replace('_IMP','')}",
                xaxis_title="num of ASR chosen",
                yaxis_title=metric.replace('_IMP',''),
                boxmode="group",
                xaxis=dict(categoryorder='category ascending')
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("📈 Detailed Comparisons (Bar Chart per Sample)")
        st.info(f"Showing comparisons for {selected_sentences} example sentences only. Use the sidebar filter to change.")
        metric = st.selectbox("Metric (Comparisons)", ["WER_IMP", "CER_IMP", "S_IMP"], index=0, key="metric_comparisons")
        experiment_types = sorted(filtered_df_for_comparisons['experiment_type'].unique())
        selected_types = st.multiselect("Experiment Types (Comparisons)", experiment_types, default=experiment_types, key="exp_types_comparisons")
        colors = {"Improved": "#1f77b4", "Raw ASR": "#ff7f0e"}
        for exp_type in selected_types:
            exp_df = filtered_df_for_comparisons[filtered_df_for_comparisons['experiment_type'] == exp_type]
            # Group by filename and average across runs
            agg_df = exp_df.groupby('filename').agg({
                metric: 'mean',
                metric.replace('_IMP', '_original'): 'mean'
            }).reset_index()
            # Prepare data for bar chart
            bar_data = []
            for idx, row in agg_df.iterrows():
                bar_data.append({
                    'filename': row['filename'],
                    'Metric': 'Improved',
                    'Value': row[metric]
                })
                bar_data.append({
                    'filename': row['filename'],
                    'Metric': 'Raw ASR',
                    'Value': row[metric.replace('_IMP', '_original')]
                })
            bar_df = pd.DataFrame(bar_data)
            fig = px.bar(
                bar_df,
                x='filename',
                y='Value',
                color='Metric',
                barmode='group',
                color_discrete_map=colors,
                title=f"{exp_type} · {metric.replace('_IMP','')} (Averaged per Sample)"
            )
            fig.update_layout(
                xaxis_title="Sample (File)",
                yaxis_title=metric.replace('_IMP',''),
                legend_title=""
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🔍 Detailed Analysis")
        st.info("Showing best performers across ALL sentence counts.")
        
        # Best performing files - using ALL sentence counts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🏆 Best Final WER Scores**")
            # Always sort by the best (lowest) final WER_IMP
            best_wer_df = filtered_df_by_experiment_and_file.nsmallest(5, 'WER_IMP')
            
            cols_to_show_wer = ['filename', 'experiment', 'num_example_sentences', 'WER_IMP']
            # Add original and improvement columns if data is available
            if has_raw_data and 'WER_original' in best_wer_df.columns:
                cols_to_show_wer.append('WER_original')
            if has_raw_data and 'WER_improvement_pct' in best_wer_df.columns:
                cols_to_show_wer.append('WER_improvement_pct')
            
            # Filter to ensure only existing columns are selected before displaying
            best_wer_display = best_wer_df[[col for col in cols_to_show_wer if col in best_wer_df.columns]]
            if not best_wer_display.empty:
                st.dataframe(best_wer_display, hide_index=True)
            else:
                st.info("No data to display for best WER scores with current filters.")
        
        with col2:
            st.markdown("**🎯 Best Final Semantic Scores**")
            # Always sort by the best (highest) final S_IMP
            best_semantic_df = filtered_df_by_experiment_and_file.nlargest(5, 'S_IMP')
            
            cols_to_show_sem = ['filename', 'experiment', 'num_example_sentences', 'S_IMP']
            # Add original and improvement columns if data is available
            if has_raw_data and 'S_original' in best_semantic_df.columns:
                cols_to_show_sem.append('S_original')
            if has_raw_data and 'S_improvement_pct' in best_semantic_df.columns:
                cols_to_show_sem.append('S_improvement_pct')
            
            # Filter to ensure only existing columns are selected before displaying
            best_semantic_display = best_semantic_df[[col for col in cols_to_show_sem if col in best_semantic_df.columns]]
            if not best_semantic_display.empty:
                st.dataframe(best_semantic_display, hide_index=True)
            else:
                st.info("No data to display for best semantic scores with current filters.")
        
        # Correlation analysis - using ALL sentence counts
        if len(filtered_df_by_experiment_and_file) > 1:
            st.subheader("🔗 Metric Correlations")
            
            corr_data = filtered_df_by_experiment_and_file[['WER_IMP', 'CER_IMP', 'S_IMP', 'num_example_sentences']].corr()
            
            fig_corr = px.imshow(
                corr_data,
                title="Correlation Matrix of Metrics (All Sentence Counts)",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with tab4:
        st.subheader("📋 Raw Data")
        
        # Display options
        show_raw = st.checkbox("Show Original ASR Metrics", value=False) if has_raw_data else False
        
        # Prepare display columns
        display_cols = ['experiment', 'filename', 'num_example_sentences', 'WER_IMP', 'CER_IMP', 'S_IMP']
        
        if has_raw_data and show_raw:
            display_cols.extend(['WER_original', 'CER_original', 'S_original'])
        
        if has_raw_data:
            display_cols.extend(['WER_improvement_pct', 'CER_improvement_pct', 'S_improvement_pct'])
        
        # Filter columns that exist
        available_cols = [col for col in display_cols if col in filtered_df_by_experiment_and_file.columns]
        
        st.dataframe(
            filtered_df_by_experiment_and_file[available_cols],
            use_container_width=True,
            hide_index=True
        )
        
        # Download button
        csv_data = filtered_df_by_experiment_and_file.to_csv(index=False)
        st.download_button(
            label="💾 Download Results as CSV",
            data=csv_data,
            file_name=f"asr_results_{len(filtered_df_by_experiment_and_file)}_records.csv",
            mime="text/csv",
            help="Download the filtered results as a CSV file"
        )

if __name__ == "__main__":
    main() 