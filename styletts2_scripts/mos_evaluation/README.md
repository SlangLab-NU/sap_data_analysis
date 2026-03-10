# MOS Evaluation App

Speech quality evaluation interface for StyleTTS2 synthesized samples.

## Setup

### 1. Generate the sample manifest
From inside `mos_evaluation/`:
```bash
python generate_manifest.py
```
This reads `../speaker_similarity.csv` and generates `samples.json`.

### 2. Build the Docker image
```bash
docker build -t mos-eval .
```

### 3. Run the container
```bash
docker run -p 8080:8080 \
  -v $(pwd)/../reference_audio:/app/static/audio/reference \
  -v $(pwd)/../generated_audio:/app/static/audio/synthetic \
  -v $(pwd)/results:/app/results \
  mos-eval
```

### 4. Open the interface
Visit [http://localhost:8080](http://localhost:8080) in a browser.

Raters enter their assigned ID and evaluate all 50 samples sequentially.
Scores are saved to `results/ratings.csv` on the host machine.

## Results

`results/ratings.csv` contains one row per submission:

| Field | Description |
|---|---|
| `timestamp` | UTC time of submission |
| `rater_id` | Assigned rater code |
| `speaker_id` | Speaker from manifest |
| `etiology` | Speaker diagnosis |
| `cosine_similarity` | SECS score from analysis |
| `original_score` | MOS rating (1–5) for original audio |
| `synthetic_score` | MOS rating (1–5) for synthesized audio |
| `a_was_original` | Whether Sample A was the original (blind check) |

## Notes

- Sample order is randomized per session
- A/B labels are randomized per sample (blind evaluation)
- Audio files are mounted at runtime — not baked into the image