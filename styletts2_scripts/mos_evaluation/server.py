import csv, json, os, random
from datetime import datetime
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

app = FastAPI()

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RESULTS_FILE = os.path.join(RESULTS_DIR, "ratings.csv")

# Write CSV header if file doesn't exist
if not os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "rater_id", "speaker_id", "etiology",
            "cosine_similarity", "original_score", "synthetic_score",
            "a_was_original"
        ])

with open("samples.json") as f:
    ALL_SAMPLES = json.load(f)


class Rating(BaseModel):
    rater_id: str
    speaker_id: str
    original_score: int
    synthetic_score: int
    a_was_original: bool


@app.get("/api/samples")
def get_samples():
    samples = ALL_SAMPLES.copy()
    random.shuffle(samples)
    # Randomize which is labeled A/B per sample
    for s in samples:
        s["a_is_original"] = random.choice([True, False])
    return JSONResponse(samples)


@app.post("/api/submit")
def submit_rating(rating: Rating):
    sample = next((s for s in ALL_SAMPLES if s["id"] == rating.speaker_id), {})
    with open(RESULTS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            datetime.utcnow().isoformat(),
            rating.rater_id,
            rating.speaker_id,
            sample.get("etiology", ""),
            sample.get("cosine_similarity", ""),
            rating.original_score,
            rating.synthetic_score,
            rating.a_was_original,
        ])
    return {"status": "ok"}


app.mount("/audio", StaticFiles(directory="static/audio"), name="audio")
app.mount("/", StaticFiles(directory="static", html=True), name="static")