# Multi-ASR Fusion

This module implements the multi-hypothesis fusion approach described in Section 3.1 of the paper. It collects transcripts from ten ASR services, fuses them into a single corrected transcript using an LLM, and evaluates WER and semantic similarity against ground truth.

## Overview

`fuse.py` loads raw transcripts from all ten ASR engines, segments them into sentence-level units using `wtpsplit`, and calls an LLM (GPT-4.1, DeepSeek R1, or Gemini 2.5 Pro) to fuse them into one coherent transcript. Evaluation plots of WER and semantic similarity are produced for every engine and fusion mode.

## Directory Structure

```
Multi-ASR/
├── fuse.py                            # Main script: load, fuse, evaluate, plot
├── requirements.txt                   # Python dependencies
├── raw_transcripts/                   # Raw outputs from 10 ASR services (60 files)
│   ├── aprocsa1554a AssemblyAI.txt
│   ├── aprocsa1554a Azure.txt
│   └── ...                            # [sample_id] [Service].txt
├── ground_truth/                      # Reference transcripts (6 samples)
│   ├── aprocsa1554a.txt
│   └── ...
├── evaluation_outputs (ALL 10)/       # Fusion outputs using all 10 ASRs
│   ├── aprocsa1554a GPT-4.1.txt
│   ├── aprocsa1554a DeepSeek.txt
│   ├── aprocsa1554a Gemini POST.txt
│   └── ...
├── input_audio/                       # WAV files (not tracked in git — place here manually)
│   ├── aprocsa1554a.wav
│   └── ...
├── evaluation_outputs (4 WORST)/      # Fusion outputs using the 4 weakest ASRs only
└── experiment_outputs (RAND 100)/     # Variable-ASR experiment outputs (100 random combos per m)
```

> **Audio files**: `input_audio/` is not tracked in git due to file size. To populate it, download the video files from the [APROCSA corpus](https://doi.org/10.3390/data7110148), run `Data Preparation/extract_corpus_data.py` to extract the audio, then place the resulting WAV files here named as `aprocsa[ID]a.wav` (e.g. `aprocsa1554a.wav`).

## Setup

### Prerequisites

- Python 3.8+
- API keys for the LLM services you want to use:
  - **GPT-4.1**: Azure OpenAI deployment
  - **DeepSeek R1**: DeepSeek API key
  - **Gemini 2.5 Pro**: Google AI API key

### Installation

```bash
cd Multi-ASR
pip install -r requirements.txt
```

### API Key Configuration

Open `fuse.py` and fill in your API keys in the three fusion functions (search for `<YOUR ... API KEY>`):

```python
# GPT-4.1 (Azure OpenAI)
client = AzureOpenAI(
    api_key="<YOUR AZURE OPENAI API KEY>",
    ...
    azure_endpoint="<YOUR AZURE OPENAI ENDPOINT>"
)

# DeepSeek R1
client = OpenAI(
    api_key="<YOUR DEEPSEEK API KEY>",
    ...
)

# Gemini 2.5 Pro
client = genai.Client(api_key="<YOUR GEMINI API KEY>")
```

## Usage

Run from the `Multi-ASR/` directory:

```bash
python fuse.py
```

This will:
1. Load raw transcripts from `raw_transcripts/` and ground truth from `ground_truth/`
2. Fuse all 10 ASR hypotheses with GPT-4.1, DeepSeek R1, and Gemini 2.5 Pro; save results to `evaluation_outputs (ALL 10)/`
3. Compute WER and semantic similarity for every ASR engine and fusion mode; save to `metrics.csv`
4. Produce WER and semantic similarity bar charts (per-sample and concatenated)
5. Run the Variable-ASR experiment (prompts whether to reuse cached results); save boxplots and scatter plot to `experiment_outputs (RAND 100)/`

## ASR Services

`_SUFFIX_MAP` at the top of `fuse.py` defines which ASR services are recognized. Each entry maps a lowercase filename suffix to a canonical key and display label:

```python
_SUFFIX_MAP = {
    "assemblyai":   ("AssemblyAI",  "AssemblyAI Slam-1"),
    "aws":          ("AWS",         "AWS Transcribe"),
    "azure":        ("Azure",       "Azure Speech"),
    "deepgram":     ("Deepgram",    "Deepgram Nova-3"),
    "elevenlabs":   ("ElevenLabs",  "ElevenLabs Scribe v1"),
    "gcp":          ("GCP",         "GCP Chirp 2"),
    "gemini":       ("Gemini",      "Gemini 2.5 Pro Audio"),
    "gladia":       ("Gladia",      "Gladia Solaria"),
    "speechmatics": ("Speechmatics","Speechmatics Ursa 2"),
    "whisper":      ("Whisper",     "Whisper v3"),
}
```

Transcript files in `raw_transcripts/` must be named `[sample_id] [Service].txt` where the service name (lowercased) matches a key in `_SUFFIX_MAP`. To add a new ASR system, add an entry here and place the corresponding transcript files in `raw_transcripts/`.

## LLM Fusion Modes

| Mode | Model | Output suffix |
|---|---|---|
| GPT-4.1 Fusion | GPT-4.1 (Azure OpenAI) | `GPT-4.1` |
| DeepSeek R1 Fusion | DeepSeek R1 | `DeepSeek` |
| Gemini 2.5 Pro Fusion | Gemini 2.5 Pro | `Gemini POST` |

## Variable-ASR Experiment

`run_experiment()` (called automatically at the end of `main()`) tests how WER improvement scales with the number of fused ASR systems. For each value of m in `{1, 2, 3, 5, 7}`, it:

1. Samples up to 100 random m-combinations of the 10 available ASR systems (seeded for reproducibility).
2. Fuses each combination with GPT-4.1 via `fuse_combo()`, caching per-sample outputs in `experiment_outputs (RAND 100)/` for reuse.
3. Computes the **relative WER improvement** for each combination:

   ```
   rel_impr = (avg_baseline_WER − fused_WER) / avg_baseline_WER × 100
   ```

   where `avg_baseline_WER` is the mean WER of the m selected ASRs before fusion.

4. Saves incremental boxplots (`wer_upto_{m}.svg`) after each m value is processed, and a final scatter plot (`scatter_with_fit.svg`) of baseline WER vs. relative improvement across all combinations.

On first run, `fuse.py` will prompt:

```
Reuse existing random experiment data? (y/N)
```

Enter `y` to skip API calls for combinations already cached in `experiment_outputs (RAND 100)/`, or `N` to re-run everything.

## Data

All experiments use the [APROCSA corpus](https://doi.org/10.3390/data7110148) — six 20-minute conversational speech sessions from individuals with aphasia (IDs: 1554, 1713, 1731, 1738, 1833, 1944).

## Citation

See the root [CITATION.cff](../CITATION.cff) or cite as:

```bibtex
@inproceedings{wen2026llm,
  title     = {{LLM}-Based Post-{ASR} Error Correction for Disordered Speech},
  author    = {Wen, Hangyi and Assefa, Mikiyas and Semsayan, Anas and Feo-Flushing, Eduardo},
  booktitle = {ICASSP},
  year      = {2026},
}
```
