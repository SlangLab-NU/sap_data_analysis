import csv, json, os

csv_path = "../styletts2_synthesis/speaker_similarity.csv"
manifest = []

with open(csv_path) as f:
    reader = csv.DictReader(f)
    for row in reader:
        manifest.append({
            "id": row["speaker_id"],
            "etiology": row["etiology"],
            "cosine_similarity": float(row["cosine_similarity"]),
            "original": f"/audio/reference/{os.path.basename(row['original_audio'])}",
            "synthetic": f"/audio/synthetic/{os.path.basename(row['synthetic_audio'])}",
        })

with open("samples.json", "w") as f:
    json.dump(manifest, f, indent=2)

print(f"Generated samples.json with {len(manifest)} entries")