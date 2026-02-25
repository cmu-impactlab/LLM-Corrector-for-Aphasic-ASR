# LLM-Based Post-ASR Error Correction for Disordered Speech

Source code, LLM prompts, sampling functions, and transcript data used in the paper:

> **"LLM-Based Post-ASR Error Correction for Disordered Speech"**
> Hangyi Wen\*, Mikiyas Assefa\*, Anas Semsayan\*, Eduardo Feo-Flushing (\*equal contribution)
> Carnegie Mellon University, School of Computer Science

The paper evaluates three LLM-based strategies for post-ASR error correction on aphasic speech (APROCSA corpus):
1. **Multi-ASR Fusion** — hypotheses from ten ASR services fused by an LLM
2. **Few-Shot Correction** — single-hypothesis correction via few-shot prompt engineering
3. **LoRA SFT** — supervised fine-tuning of Qwen2.5-14B with LoRA adapters

## Repository Structure

```
LLM-Corrector-for-Aphasic-ASR/
│
├── Data Preparation/              # Corpus processing (+1 component)
│   ├── extract_corpus_data.py        # Extracts PAR utterances from .cha + .mp4 files
│   ├── select_examples.py            # Example selection (exhaustive phoneme, random)
│   ├── aprocsa*.cha                  # CHAT transcript files (6 samples)
│   └── *.mp4                         # Video files (6 samples, not tracked in git)
│
├── Multi-ASR/                     # Section 3.1: Multi-ASR Fusion
│   ├── fuse.py                       # Main script: fuse with GPT-4.1/DeepSeek R1/Gemini
│   ├── raw_transcripts/              # Raw outputs from 10 ASR services (60 files)
│   ├── ground_truth/                 # Reference transcripts (6 samples)
│   ├── evaluation_outputs (ALL 10)/  # Fusion results for all 10 ASRs + metrics
│   ├── evaluation_outputs (4 WORST)/ # Fusion results for 4 weakest ASRs
│   └── experiment_outputs (RAND 100)/# Random subset experiment outputs (Fig. 3)
│
├── Few-shot/                      # Section 3.2: Few-Shot Single-Hypothesis Correction
│   ├── main.py                       # Main experiment runner
│   ├── run_experiments.py            # CLI experiment launcher
│   ├── streamlit_app.py              # Interactive results dashboard
│   ├── models/azure_gpt.py           # GPT-4.1 few-shot correction
│   ├── evaluators/evaluator.py       # WER + SBERT semantic similarity evaluation
│   ├── utils/                        # Example selection, metrics, file utilities
│   ├── data/                         # Per-system ASR outputs and ground truth
│   └── analysis/                     # Standalone result analysis and visualization scripts
│
└── LoRA SFT/                      # Section 3.3: Supervised Fine-Tuning
    ├── train_qwen.py                 # LoRA fine-tuning (Qwen2.5-14B, TRL SFTTrainer)
    ├── dataset/                      # Alpaca-style JSONL training sets
    │   ├── aprocsa1944a-random.jsonl      # Random selection (26 utterances)
    │   └── aprocsa1944a-exhaustive.jsonl  # Exhaustive-phoneme selection (26 utterances)
    ├── run_two.sh                    # Launch both training runs on 2 GPUs
    ├── wer-score.py                  # WER evaluation against ground truth
    ├── scripts/                      # Generation, comparison, and timed run helpers
    ├── outputs/                      # WER scores and model comparison results
    └── docs/USER_GUIDE.md            # Detailed setup and usage guide
```

## Quick Start

### Data Preparation
```bash
cd "Data Preparation"
python extract_corpus_data.py   # requires ffmpeg + pylangacq
```

### Multi-ASR Fusion (Section 3.1)
```bash
cd Multi-ASR
python fuse.py
```

### Few-Shot Correction (Section 3.2)
```bash
cd Few-shot
pip install -r requirements.txt
python main.py
```

### LoRA SFT (Section 3.3)
```bash
cd "LoRA SFT"
pip install unsloth trl transformers peft bitsandbytes datasets accelerate
./run_two.sh    # trains random + exhaustive models on 2 GPUs
```

## Data

All experiments use the [APROCSA corpus](https://doi.org/10.3390/data7110148) — six 20-minute conversational speech samples from individuals with aphasia (IDs: 1554, 1713, 1731, 1738, 1833, 1944).

ASR services: AssemblyAI Slam-1, AWS Transcribe, Azure Speech, Deepgram Nova-3, ElevenLabs Scribe v1, GCP Chirp 2, Gemini 2.5 Pro Audio, Gladia Solaria, Speechmatics Ursa 2, Whisper v3.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{wen2026llm,
  title     = {{LLM}-Based Post-{ASR} Error Correction for Disordered Speech},
  author    = {Wen, Hangyi and Assefa, Mikiyas and Semsayan, Anas and Feo-Flushing, Eduardo},
  booktitle = {ICASSP},
  year      = {2026},
}
```
