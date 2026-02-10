import gzip
import json
from pathlib import Path

def extract_manifest(input_path, output_path=None):
    """
    Extract a .jsonl.gz file to a readable .jsonl file.
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        print(f"File not found: {input_path}")
        return
    
    # Default output: same name without .gz
    if output_path is None:
        output_path = input_path.with_suffix('')  # Remove .gz
    
    output_path = Path(output_path)
    
    print(f"Extracting: {input_path.name}")
    print(f"Output to: {output_path}")
    
    line_count = 0
    with gzip.open(input_path, 'rt', encoding='utf-8') as f_in:
        with open(output_path, 'w', encoding='utf-8') as f_out:
            for line in f_in:
                f_out.write(line)
                line_count += 1
    
    print(f"✓ Extracted {line_count} lines")
    print(f"✓ Saved to: {output_path}")
    
    # Show first entry as preview
    print("\nFirst entry preview:")
    with open(output_path, 'r') as f:
        first_line = f.readline()
        print(json.dumps(json.loads(first_line), indent=2))

# Main
manifest_dir = Path("/scratch/lewis.jor/VallE/egs/libritts/data/manifests")

# List all manifests
print("Available manifest files:")
print("="*60)

recordings = []
supervisions = []
other = []

for f in sorted(manifest_dir.glob("*.jsonl.gz")):
    if 'recordings' in f.name.lower():
        recordings.append(f)
    elif 'supervisions' in f.name.lower():
        supervisions.append(f)
    else:
        other.append(f)

print("\nRECORDINGS:")
for i, f in enumerate(recordings):
    print(f"  {i+1}. {f.name}")

print("\nSUPERVISIONS:")
for i, f in enumerate(supervisions):
    print(f"  {i+1}. {f.name}")

if other:
    print("\nOTHER:")
    for i, f in enumerate(other):
        print(f"  {i+1}. {f.name}")

# Extract a supervision file
print("\n" + "="*60)
if supervisions:
    print(f"\nExtracting first supervision file...")
    extract_manifest(supervisions[0])
else:
    print("\nNo supervision files found!")