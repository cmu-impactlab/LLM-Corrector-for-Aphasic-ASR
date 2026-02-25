# Few-Shot Post-ASR Error Correction

This module implements the few-shot single-hypothesis correction approach described in Section 3.2 of the paper. It uses GPT-4.1 with in-context example pairs to correct ASR transcripts of aphasic speech, evaluated across ten ASR systems on the APROCSA corpus.

## Overview

Each experiment selects a handful of (ASR error, ground truth) example pairs from a held-out pool, constructs a few-shot prompt, and sends each utterance to GPT-4.1 for correction. Results are evaluated with WER and SBERT semantic similarity.

## Directory Structure

```
Few-shot/
├── main.py                        # Main experiment runner (interactive)
├── run_experiments.py             # CLI launcher for single experiments
├── run_per_asr_experiments.py     # Batch runner across all ASR systems
├── run_parallel_experiments.py    # Parallel batch runner (multi-process)
├── prompt.py                      # Example selection algorithms
├── generate_examples.py           # Precomputes example pools
├── app.py                         # Streamlit results dashboard
├── models/
│   └── azure_gpt.py               # GPT-4.1 few-shot correction
├── evaluators/
│   └── evaluator.py               # WER + SBERT evaluation
├── utils/                         # File utilities, metrics helpers
├── analysis/                      # Standalone result analysis scripts
└── data/
    ├── asr_raw/                   # Raw ASR outputs (10 systems × 6 samples)
    │   ├── AssemblyAI/
    │   ├── AWS/
    │   ├── Azure/
    │   ├── Deepgram/
    │   ├── ElevenLabs/
    │   ├── GCP/
    │   ├── Gemini/
    │   ├── Gladia/
    │   ├── Speechmatics/
    │   └── Whisper/
    ├── ground_truth/              # Reference transcripts (6 samples)
    └── examples_max/              # Precomputed example pools (6 samples)
```

## Setup

### Prerequisites

- Python 3.8+
- Azure OpenAI API access (GPT-4.1 deployment)

### Installation

```bash
cd Few-shot
pip install -r requirements.txt
```

### API Key Configuration

Create a `.env` file in the `Few-shot/` directory:

```env
AZURE_OPENAI_API_KEY=your_azure_api_key
AZURE_OPENAI_ENDPOINT=https://your-endpoint.cognitiveservices.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
```

The `.env` file is excluded from git. Never commit API keys.

## Usage

### Interactive Experiment Runner

```bash
python main.py
```

Prompts for example selection strategy and experiment parameters, then runs correction and saves results to `outputs/`.

### CLI Launcher

```bash
python run_experiments.py --asr-system "Azure" --strategy exhaustive_phoneme --model gpt4
python run_experiments.py --asr-system Whisper --strategy random_error --sentences 2 4 6 8 10
```

### Run Across All ASR Systems

```bash
python run_per_asr_experiments.py
```

Runs the correction experiment for every available ASR system and saves results organized by system.

### Streamlit Dashboard

```bash
streamlit run app.py
```

Interactive visualization of WER and semantic similarity across systems, strategies, and sentence counts.

## Example Selection Strategies

| Strategy | Description |
|---|---|
| `exhaustive_phoneme` | Greedy selection maximizing phoneme coverage across examples |
| `random_error` | Random selection from error sentence pairs |
| `data_driven` | Speaker-specific error patterns combined with phonetic diversity |

## Evaluation Metrics

- **WER** (Word Error Rate): word-level accuracy relative to ground truth
- **Semantic Similarity**: sentence-transformer cosine similarity (SBERT)

## Data

All experiments use the [APROCSA corpus](https://doi.org/10.3390/data7110148) — six 20-minute conversational speech sessions from individuals with aphasia (IDs: 1554, 1713, 1731, 1738, 1833, 1944). Ground truth transcripts are in `data/ground_truth/`. Raw ASR outputs for all ten services are in `data/asr_raw/`.

## Output Format

```
outputs/
└── [ASR_System]/
    └── [strategy]/
        └── [timestamp]_[strategy]_run[N]/
            └── [sample_id]/
                └── [K]_sentences/
                    ├── IMPROVED.txt           # LLM-corrected transcript
                    ├── PROMPT_EXAMPLES.txt    # Examples used in the prompt
                    └── evaluation_metrics.csv # WER + similarity scores
```

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
