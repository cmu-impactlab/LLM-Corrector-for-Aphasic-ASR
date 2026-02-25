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
└── experiment_outputs (RAND 100)/     # Random-100-utterance subset experiment (Fig. 3)
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
1. Load all raw transcripts from `raw_transcripts/`
2. Segment transcripts into sentence-level units
3. Fuse hypotheses with each enabled LLM post-processing mode
4. Evaluate WER and semantic similarity against `ground_truth/`
5. Save fused transcripts to `evaluation_outputs (ALL 10)/`
6. Produce evaluation plots

## ASR Services

Ten services are supported (raw transcripts must be named `[sample_id] [Service].txt`):

| Service key | Full name |
|---|---|
| `AssemblyAI` | AssemblyAI Slam-1 |
| `AWS` | AWS Transcribe |
| `Azure` | Azure Speech |
| `Deepgram` | Deepgram Nova-3 |
| `ElevenLabs` | ElevenLabs Scribe v1 |
| `GCP` | GCP Chirp 2 |
| `Gemini` | Gemini 2.5 Pro Audio |
| `Gladia` | Gladia Solaria |
| `Speechmatics` | Speechmatics Ursa 2 |
| `Whisper` | Whisper v3 |

## LLM Fusion Modes

| Mode | Model | Output suffix |
|---|---|---|
| GPT-4.1 Fusion | GPT-4.1 (Azure OpenAI) | `GPT-4.1` |
| DeepSeek R1 Fusion | DeepSeek R1 | `DeepSeek` |
| Gemini 2.5 Pro Fusion | Gemini 2.5 Pro | `Gemini POST` |

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
