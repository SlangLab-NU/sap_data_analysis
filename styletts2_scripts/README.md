# StyleTTS2 Voice Cloning Pipeline for SAP Dataset

This pipeline generates synthetic speech from dysarthric SAP speakers using StyleTTS2 voice cloning, then evaluates the quality through speaker similarity and WER metrics.

## Overview

The pipeline creates "typical-sounding" speech that preserves the speaker's voice identity but removes dysarthric characteristics. This allows comparison of:
- **Reference (dysarthric)**: Original SAP speaker audio
- **Synthetic (typical)**: Same speaker's voice, but with normal speech patterns

## Prerequisites

### Containers Required

1. **StyleTTS2 Container** (`sap_analysis_styletts2.sif`)
   - Used for: Speaker selection, TTS synthesis, speaker similarity
   - Dockerfile location: `../sap_data_analysis/sap_analysis_styletts2.sif`
   - Run container (update .sh with your scratch dir) `bash run_styletts2.sh`

2. **NeMo Container** (`sap_analysis_nemo.sif`)
   - Used for: WER calculation
   - Location: `../sap_data_analysis/sap_analysis_nemo.sif`
   - Run container (update .sh with your scratch dir) `bash run_sap_nemo.sh`

### Input Data

- **Speaker ratings CSV**: `VallE/egs/sap/analysis/ratings/speaker_ratings_TRAIN.csv`
- **Extracted SAP audio**: `VallE/egs/sap/extracted/TRAIN/`

## Pipeline Steps

### Step 1: Select Speakers for Synthesis

**Script:** `select_speakers_for_synthesis.py`  
**Container:** StyleTTS2

Selects diverse speakers from specified etiologies (default: 25 Parkinson's + 25 Cerebral Palsy).
```bash
# Run from styletts2_scripts/ directory
python select_speakers_for_synthesis.py
```

**Outputs:**
- `styletts2_synthesis/synthesis_manifest.csv` - Selected speakers with metadata
- `styletts2_synthesis/reference_audio/` - Copied reference audio files

**Arguments:**
- `--valle-dir`: Path to VallE directory (default: `../../../VallE`)
- `--speakers-per-etiology`: Number per etiology (default: 25)
- `--output-dir`: Output directory (default: `styletts2_synthesis`)

---

### Step 2: Generate Synthetic Speech

**Script:** `generate_tts_samples.py`  
**Container:** StyleTTS2

Uses StyleTTS2 to clone each speaker's voice and synthesize their prompt text.
```bash
python generate_tts_samples.py
```

**Outputs:**
- `styletts2_synthesis/generated_audio/` - Synthesized audio files
- `styletts2_synthesis/generated_audio/synthesis_results.csv` - Synthesis status

**Arguments:**
- `--manifest`: Synthesis manifest CSV (default: auto-detected)
- `--diffusion-steps`: Quality parameter, 5-20 (default: 10)
- `--max-speakers`: Limit for testing (default: None = all)

**Note:** First run downloads ~400MB of StyleTTS2 models

---

### Step 3: Calculate Speaker Similarity

**Script:** `calculate_secs.py` (Speaker Embedding Cosine Similarity)  
**Container:** StyleTTS2

Measures how well StyleTTS2 preserved the speaker's voice identity using WavLM embeddings.
```bash
python calculate_secs.py
```

**Outputs:**
- `styletts2_synthesis/speaker_similarity.csv` - Cosine similarity scores

**Metrics:**
- **Cosine Similarity**: 0.0-1.0 (higher = better voice cloning)
  - > 0.90: Very likely same speaker
  - > 0.75: Possibly same speaker
  - < 0.75: Likely different speakers

**Arguments:**
- `--synthesis-results`: Input CSV (default: auto-detected)
- `--output`: Output CSV path

---

### Step 4: Calculate WER Comparison

**Script:** `calculate_wer.py`  
**Container:** NeMo (GPU required)

Transcribes both reference and synthetic audio using NeMo ASR, calculates WER for comparison.
```bash
# Requires GPU node
python calculate_wer.py
```

**Outputs:**
- `styletts2_synthesis/synthesis_wer_comparison.csv` - WER scores and comparison

**Metrics:**
- **Reference WER**: ASR performance on dysarthric speech
- **Synthetic WER**: ASR performance on typical speech (cloned voice)
- **WER Improvement**: Difference (positive = synthetic is better)
- **Percent Improvement**: Relative improvement percentage

**Arguments:**
- `--synthesis-results`: Input CSV (default: auto-detected)
- `--model-name`: NeMo model (default: `stt_en_conformer_ctc_large`)
- `--batch-size`: Batch size (default: 8)

---

## Output Files Structure
```
styletts2_synthesis/
├── synthesis_manifest.csv              # Selected speakers + metadata
├── reference_audio/                    # Original SAP audio (1 per speaker)
│   ├── <speaker_id>.wav
│   └── ...
├── generated_audio/                    # StyleTTS2 synthesized audio
│   ├── <speaker_id>_synthetic.wav
│   ├── ...
│   └── synthesis_results.csv          # Synthesis status
├── speaker_similarity.csv              # WavLM cosine similarity
└── synthesis_wer_comparison.csv        # WER comparison results
```

## Expected Results

### Typical Findings:
- **Speaker Similarity**: Mean ~0.80-0.90 (good voice cloning)
- **Reference WER**: High (20-90% for dysarthric speech)
- **Synthetic WER**: Low (0-20% for typical speech)
- **WER Improvement**: 50-100% reduction in most cases

### Interpretation:
Large WER improvements demonstrate that **dysarthric characteristics** (not the speaker's voice itself) are the primary cause of ASR errors.

---

## Troubleshooting

### StyleTTS2 synthesis fails
- Check NLTK data downloaded: `python -c "import nltk; nltk.download('punkt_tab')"`
- Verify PyTorch version: Should be 2.1.x (not 2.6+)

### NeMo WER calculation errors
- Ensure GPU available: `nvidia-smi`
- Reduce batch size if OOM: `--batch-size 4`
- Check audio format: Script auto-converts stereo to mono

### Container build issues
- **Disk space**: Use scratch directory for builds
- **Network timeouts**: Retry or use older container versions
- **Dependency conflicts**: Follow exact versions in Dockerfile

---

## Notes

- **First run**: Models auto-download (~1GB total across both containers)
- **Processing time**: ~1-2 min per speaker for synthesis, ~30 sec for WER
- **Storage**: ~50MB per 50 speakers (audio + CSVs)
- **Voice Cloning Ethics**: Synthetic samples are research-only, not for distribution

---

## Citation

If using this pipeline, cite:
- **SAP Dataset**: Speech Accessibility Project
- **StyleTTS2**: Li et al., "StyleTTS 2: Towards Human-Level Text-to-Speech..."
- **NeMo**: NVIDIA NeMo Toolkit
- **WavLM**: Microsoft WavLM for speaker verification

---
