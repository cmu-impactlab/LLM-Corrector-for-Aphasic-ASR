import streamlit as st
import os
import jiwer
import re

# --- App Configuration ---
st.set_page_config(layout="wide", page_title="ASR Alignment Visualizer")

# --- Core Functions (Copied and adapted from previous script) ---

@st.cache_data
def find_substring_indices(text_words, substring_words):
    """Finds all start and end indices of a word-list substring in a larger word-list."""
    indices = set()
    len_sub = len(substring_words)
    if len_sub == 0:
        return indices
    for i in range(len(text_words) - len_sub + 1):
        if text_words[i:i + len_sub] == substring_words:
            indices.update(range(i, i + len_sub))
    return indices

def format_alignment_for_display(jiwer_output, ref_name="Reference", hyp_name="Hypothesis", ref_example_indices=None):
    """
    Formats the jiwer alignment into a readable table, with optional highlighting
    for words that were part of the examples.
    """
    if ref_example_indices is None:
        ref_example_indices = set()

    ref_words = jiwer_output.references[0]
    hyp_words = jiwer_output.hypotheses[0]

    max_ref_len = max(len(w) for w in ref_words) if ref_words else len(ref_name)
    max_hyp_len = max(len(w) for w in hyp_words) if hyp_words else len(hyp_name)
    
    col_1_width = max(max_ref_len, len(ref_name)) + 4
    col_2_width = max(max_hyp_len, len(hyp_name)) + 2

    output_lines = []
    header = (
        ref_name.upper().ljust(col_1_width) +
        hyp_name.upper().ljust(col_2_width) +
        "OPERATION"
    )
    output_lines.append(header)
    output_lines.append("-" * (len(header) + 5))

    alignment_chunks = jiwer_output.alignments[0]

    for chunk in alignment_chunks:
        ref_slice = ref_words[chunk.ref_start_idx:chunk.ref_end_idx]
        hyp_slice = hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx]

        chunk_example_indices = {
            i for i in range(chunk.ref_start_idx, chunk.ref_end_idx) 
            if i in ref_example_indices
        }

        def mark_word(word, is_example):
            return f"*{word}*" if is_example else word

        if chunk.type == "equal":
            for i, (ref_word, hyp_word) in enumerate(zip(ref_slice, hyp_slice)):
                is_ex = (chunk.ref_start_idx + i) in chunk_example_indices
                line = (
                    mark_word(ref_word, is_ex).ljust(col_1_width) +
                    hyp_word.ljust(col_2_width) +
                    ("example" if is_ex else "correct")
                )
                output_lines.append(line)
        
        elif chunk.type == "substitute":
            for i, (ref_word, hyp_word) in enumerate(zip(ref_slice, hyp_slice)):
                is_ex = (chunk.ref_start_idx + i) in chunk_example_indices
                line = (
                    mark_word(ref_word, is_ex).ljust(col_1_width) +
                    hyp_word.ljust(col_2_width) +
                    "SUBSTITUTION" + (" (example)" if is_ex else "")
                )
                output_lines.append(line)

        elif chunk.type == "delete":
            for i, ref_word in enumerate(ref_slice):
                is_ex = (chunk.ref_start_idx + i) in chunk_example_indices
                line = (
                    mark_word(ref_word, is_ex).ljust(col_1_width) +
                    "---".ljust(col_2_width) +
                    "DELETION" + (" (example)" if is_ex else "")
                )
                output_lines.append(line)

        elif chunk.type == "insert":
            for hyp_word in hyp_slice:
                line = (
                    "---".ljust(col_1_width) +
                    hyp_word.ljust(col_2_width) +
                    "INSERTION"
                )
                output_lines.append(line)
    
    return "\n".join(output_lines)

@st.cache_data
def load_data_for_run(run, sample, num_sentences_folder):
    """Loads all necessary text files for a given selection."""
    data = {
        "gt_full": None, "imp_full": None,
        "gt_removed": None, "imp_removed": None,
        "gt_example_sentences": [], "asr_example_sentences": []
    }
    
    sample_path = os.path.join("outputs", run, sample, num_sentences_folder)
    
    paths = {
        "gt_removed": os.path.join(sample_path, "GT_examples_removed.txt"),
        "imp_removed": os.path.join(sample_path, "IMPROVED_examples_removed.txt"),
        "gt_full": os.path.join("data", "ground_truth", f"{sample}.txt"),
        "imp_full": os.path.join(sample_path, "IMPROVED_FULL.txt"),
        "prompt_examples": os.path.join(sample_path, "PROMPT_EXAMPLES.txt")
    }

    for key, path in paths.items():
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                if key == "prompt_examples":
                    lines = content.strip().split('\n')
                    for line in lines:
                        if '➜' in line:
                            parts = line.split('➜', 1)
                            if len(parts) == 2:
                                asr_text, gt_text = parts
                                data["asr_example_sentences"].append(asr_text.strip())
                                data["gt_example_sentences"].append(gt_text.strip())
                else:
                    text = content.strip()
                    if key == "gt_full":
                        text = " ".join(jiwer.RemovePunctuation()(text).split())
                    data[key] = text
    return data

# --- STREAMLIT UI ---

st.title("ASR Post-Processing Alignment Visualizer")

with st.sidebar:
    st.header("Controls")
    
    runs = sorted([d for d in os.listdir("outputs") if os.path.isdir(os.path.join("outputs", d))])
    if not runs:
        st.error("No run folders found in the 'outputs' directory.")
        st.stop()
    
    default_run = "414_run1"
    default_run_index = runs.index(default_run) if default_run in runs else 0
    selected_run = st.selectbox("Select Experiment Run", runs, index=default_run_index)

    run_path = os.path.join("outputs", selected_run)
    samples = sorted([d for d in os.listdir(run_path) if os.path.isdir(os.path.join(run_path, d))])
    if not samples:
        st.error(f"No sample folders found in '{run_path}'.")
        st.stop()
    selected_sample = st.selectbox("Select Sample", samples)

    sample_path = os.path.join(run_path, selected_sample)
    num_sent_folders = sorted([d for d in os.listdir(sample_path) if d.endswith("_sentences")], key=lambda x: int(x.split('_')[0]))
    if not num_sent_folders:
        st.error(f"No sentence folders found in '{sample_path}'.")
        st.stop()
    selected_num_folder = st.selectbox("Select Number of Examples", num_sent_folders)


st.header(f"Analysis for: `{selected_sample}`")
st.subheader(f"Run: `{selected_run}` ({selected_num_folder})")

data = load_data_for_run(selected_run, selected_sample, selected_num_folder)

with st.expander("Show Example Pairs Used in Prompt", expanded=True):
    if data["gt_example_sentences"]:
        for i, (gt_sent, asr_sent) in enumerate(zip(data["gt_example_sentences"], data["asr_example_sentences"])):
            st.markdown(f"**Pair {i+1}**")
            st.text(f"ASR: {asr_sent}")
            # --- THIS LINE IS CORRECTED ---
            st.text(f"GT:  {gt_sent}") 
    else:
        st.info("No example sentences were found or parsed for this run.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Full vs. Full", 
    "Removed vs. Removed", 
    "Full GT vs. Removed Improved", 
    "Removed GT vs. Full Improved"
])

with tab1:
    st.markdown("Compares the **complete original GT** against the **complete improved text** from the LLM.")
    st.markdown("Words from the GT that were part of an example sentence are marked with asterisks (`*word*`). This shows how the LLM handled the examples in context.")
    
    if data["gt_full"] and data["imp_full"]:
        gt_full_words = data["gt_full"].split()
        example_indices = set()
        for sent in data["gt_example_sentences"]:
            example_indices.update(find_substring_indices(gt_full_words, sent.split()))
            
        alignment = jiwer.process_words(data["gt_full"], data["imp_full"])
        formatted_text = format_alignment_for_display(
            alignment, 
            ref_name="Full GT", 
            hyp_name="Full Improved",
            ref_example_indices=example_indices
        )
        st.code(formatted_text, language=None)
    else:
        st.warning("Could not perform comparison: 'gt_full' or 'imp_full' data is missing.")

with tab2:
    st.markdown("Compares the GT and Improved texts **after** the example sentences have been removed via alignment.")
    st.markdown("This is the 'official' comparison used for calculating the final WER/CER metrics.")

    if data["gt_removed"] and data["imp_removed"]:
        alignment = jiwer.process_words(data["gt_removed"], data["imp_removed"])
        formatted_text = format_alignment_for_display(
            alignment, 
            ref_name="Removed GT", 
            hyp_name="Removed Improved"
        )
        st.code(formatted_text, language=None)
    else:
        st.warning("Could not perform comparison: 'gt_removed' or 'imp_removed' data is missing.")

with tab3:
    st.markdown("A diagnostic view comparing the **complete original GT** against the **example-removed improved text**.")
    st.markdown("This can help visualize the effect of the removal process on the improved text.")

    if data["gt_full"] and data["imp_removed"]:
        alignment = jiwer.process_words(data["gt_full"], data["imp_removed"])
        formatted_text = format_alignment_for_display(
            alignment, 
            ref_name="Full GT", 
            hyp_name="Removed Improved"
        )
        st.code(formatted_text, language=None)
    else:
        st.warning("Could not perform comparison: 'gt_full' or 'imp_removed' data is missing.")

with tab4:
    st.markdown("A diagnostic view comparing the **example-removed GT** against the **complete improved text**.")
    st.markdown("This highlights how much of the full improved text consists of the (now missing) examples.")

    if data["gt_removed"] and data["imp_full"]:
        alignment = jiwer.process_words(data["gt_removed"], data["imp_full"])
        formatted_text = format_alignment_for_display(
            alignment, 
            ref_name="Removed GT", 
            hyp_name="Full Improved"
        )
        st.code(formatted_text, language=None)
    else:
        st.warning("Could not perform comparison: 'gt_removed' or 'imp_full' data is missing.")