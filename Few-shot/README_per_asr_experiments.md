# Per-ASR Experiment Documentation

This document describes running few-shot correction experiments across each ASR system individually using `run_per_asr_experiments.py`.

## Overview

`run_per_asr_experiments.py` runs the few-shot GPT-4.1 correction pipeline on each of the ten ASR systems in turn, applying a specified example selection strategy and saving results organized by system.

This is distinct from Multi-ASR Fusion (which combines hypotheses from all systems into one transcript). Here, each ASR system's transcript is corrected independently.

## Directory Structure

### Data Organization

```
data/
├── asr_raw/
│   ├── Azure/
│   │   ├── aprocsa1554a Azure.txt
│   │   └── ...
│   ├── AssemblyAI/
│   ├── AWS/
│   ├── Deepgram/
│   ├── ElevenLabs/
│   ├── GCP/
│   ├── Gemini/
│   ├── Gladia/
│   ├── Speechmatics/
│   └── Whisper/
├── ground_truth/
│   ├── aprocsa1554a.txt
│   └── ...
└── examples_max/
    ├── aprocsa1554a_max_examples.json
    └── ...
```

### Output Organization

```
outputs/
├── Azure/
│   ├── exhaustive_phoneme/
│   │   └── [timestamp]_exhaustive_phoneme_run1/
│   └── random_error/
├── AssemblyAI/
│   └── ...
└── ...
```

## Running Experiments

### Run All ASR Systems

```bash
python run_per_asr_experiments.py
```

Runs all ten ASR systems with the default strategy and saves results to `outputs/`.

### Single ASR System

```bash
python run_per_asr_experiments.py --single-asr "Azure" --single-strategy "exhaustive_phoneme" --runs 1
python run_per_asr_experiments.py --single-asr "Whisper" --single-strategy "random_error" --runs 1
```

### Parallel Execution (PowerShell)

```powershell
./run_experiments_parallel.ps1
```

Launches one process per ASR system in parallel using PowerShell jobs.

### Check Available Systems

```bash
python run_experiments.py --list-systems
```

## Available Strategies

| Strategy | Description |
|---|---|
| `exhaustive_phoneme` | Greedy selection maximizing phoneme coverage |
| `random_error` | Random selection from error sentence pairs |
| `data_driven` | Speaker-specific error patterns with phonetic diversity |

## Supported ASR Systems

| System | Full Name |
|---|---|
| `Azure` | Azure Speech |
| `AssemblyAI` | AssemblyAI Slam-1 |
| `AWS` | AWS Transcribe |
| `Deepgram` | Deepgram Nova-3 |
| `ElevenLabs` | ElevenLabs Scribe v1 |
| `GCP` | GCP Chirp 2 |
| `Gemini` | Gemini 2.5 Pro Audio |
| `Gladia` | Gladia Solaria |
| `Speechmatics` | Speechmatics Ursa 2 |
| `Whisper` | Whisper v3 |

## Output Files

Each experiment run creates:

- `IMPROVED.txt` — LLM-corrected transcript
- `PROMPT_EXAMPLES.txt` — Example pairs used in the prompt
- `ASR_examples_removed.txt` — ASR text with example sentences removed
- `GT_examples_removed.txt` — Ground truth with example sentences removed
- `evaluation_metrics.csv` — WER, CER, and semantic similarity scores

## Analysis

```bash
# Analyze results across all ASR systems
python analyze_per_asr_results.py

# Filter by ASR system
python analyze_per_asr_results.py --asr-system "Azure"

# Filter by strategy
python analyze_per_asr_results.py --strategy exhaustive_phoneme

# Interactive dashboard
streamlit run app.py
```

## Troubleshooting

**No ASR systems found**: Check that `data/asr_raw/` contains subdirectories named after each system.

**File not found**: Files must follow the naming convention `[sample_id] [System].txt`, e.g., `aprocsa1554a Azure.txt`.

**Import errors**: Run from the `Few-shot/` directory, not a subdirectory.

**API errors**: Ensure `.env` contains valid `AZURE_OPENAI_API_KEY` and `AZURE_OPENAI_ENDPOINT`.
