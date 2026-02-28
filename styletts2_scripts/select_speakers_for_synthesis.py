import pandas as pd
import json
import random
from pathlib import Path
import argparse
import shutil
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser(description="Select SAP speakers and prepare for StyleTTS2 synthesis")
    
    parser.add_argument(
        "--valle-dir",
        type=Path,
        default=Path("../../../VallE"),
        help="Path to VallE directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("styletts2_synthesis"),
        help="Output directory for synthesis preparation"
    )
    parser.add_argument(
        "--speakers-per-etiology",
        type=int,
        default=25,
        help="Number of speakers to select per etiology"
    )
    
    return parser.parse_args()


def select_speakers(ratings_csv, etiologies, n_per_etiology):
    """
    Select speakers from specified etiologies.
    Returns DataFrame of selected speakers.
    """
    df = pd.read_csv(ratings_csv)
    
    selected = []
    
    for etiology in etiologies:
        # Filter by etiology
        etiology_speakers = df[df['Etiology'] == etiology]
        
        print(f"\n{etiology}:")
        print(f"  Available speakers: {len(etiology_speakers)}")
        
        # Select n speakers (or all if fewer than n)
        if len(etiology_speakers) >= n_per_etiology:
            # Random sample
            sampled = etiology_speakers.sample(n=n_per_etiology, random_state=42)
        else:
            sampled = etiology_speakers
            print(f"  Warning: Only {len(sampled)} speakers available (requested {n_per_etiology})")
        
        selected.append(sampled)
        print(f"  Selected: {len(sampled)} speakers")
    
    selected_df = pd.concat(selected, ignore_index=True)
    
    return selected_df


def get_random_utterance(speaker_dir):
    """
    Get a random audio file and its transcript from speaker's JSON.
    Returns: (audio_path, transcript, prompt_text) or None
    """
    speaker_dir = Path(speaker_dir)
    
    # Find JSON file
    json_files = list(speaker_dir.glob("*.json"))
    if not json_files:
        return None
    
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    
    # Get all files with transcripts
    files_with_transcripts = []
    for file_entry in data.get('Files', []):
        filename = file_entry.get('Filename', '')
        prompt = file_entry.get('Prompt', {})
        transcript = prompt.get('Transcript', '')
        prompt_text = prompt.get('Prompt Text', '')
        
        if transcript and filename:
            audio_path = speaker_dir / filename
            if audio_path.exists():
                files_with_transcripts.append({
                    'audio_path': audio_path,
                    'transcript': transcript,
                    'prompt_text': prompt_text
                })
    
    if not files_with_transcripts:
        return None
    
    # Select random utterance
    selected = random.choice(files_with_transcripts)
    
    return selected


def prepare_synthesis_data(selected_df, sap_extracted_dir, output_dir):
    """
    Prepare data for StyleTTS2 synthesis.
    For each speaker: select random utterance, copy reference audio, create manifest.
    """
    sap_extracted_dir = Path(sap_extracted_dir)
    output_dir = Path(output_dir)
    
    # Create subdirectories
    reference_dir = output_dir / "reference_audio"
    reference_dir.mkdir(parents=True, exist_ok=True)
    
    synthesis_manifest = []
    
    print("\nPreparing synthesis data...")
    
    for _, speaker_row in tqdm(selected_df.iterrows(), total=len(selected_df), desc="Processing speakers"):
        speaker_id = speaker_row['Speaker_ID']
        etiology = speaker_row['Etiology']
        
        # Find speaker directory
        speaker_dir = sap_extracted_dir / speaker_id
        
        if not speaker_dir.exists():
            print(f"  Warning: Speaker directory not found: {speaker_id}")
            continue
        
        # Get random utterance
        utterance = get_random_utterance(speaker_dir)
        
        if not utterance:
            print(f"  Warning: No valid utterances for {speaker_id}")
            continue
        
        # Copy reference audio
        ref_audio_dest = reference_dir / f"{speaker_id}.wav"
        shutil.copy(utterance['audio_path'], ref_audio_dest)
        
        # Add to manifest
        synthesis_manifest.append({
            'speaker_id': speaker_id,
            'etiology': etiology,
            'original_audio': str(utterance['audio_path']),
            'reference_audio': str(ref_audio_dest),
            'transcript': utterance['transcript'],
            'prompt_text': utterance['prompt_text'],
            'num_utterances': speaker_row['Total_Utterances'],
            'avg_rating': speaker_row.get('Average_Rating', None)
        })
    
    # Save manifest
    manifest_df = pd.DataFrame(synthesis_manifest)
    manifest_df.to_csv(output_dir / 'synthesis_manifest.csv', index=False)
    
    print(f"\n✓ Prepared {len(manifest_df)} speakers for synthesis")
    print(f"  Reference audio saved to: {reference_dir}")
    print(f"  Manifest saved to: {output_dir / 'synthesis_manifest.csv'}")
    
    return manifest_df


def main():
    args = get_args()

    ratings_csv = args.valle_dir / "egs/sap/analysis/ratings/speaker_ratings_TRAIN.csv"
    sap_extracted = args.valle_dir / "egs/sap/extracted/TRAIN"

    print("="*60)
    print("SAP Speaker Selection for StyleTTS2 Synthesis")
    print("="*60)
    
    # Select speakers
    etiologies = ["Parkinson's Disease", "Cerebral Palsy"]
    selected_df = select_speakers(
        ratings_csv, 
        etiologies, 
        args.speakers_per_etiology
    )
    
    print(f"\nTotal selected speakers: {len(selected_df)}")
    
    # Prepare synthesis data
    manifest_df = prepare_synthesis_data(
        selected_df,
        sap_extracted,
        args.output_dir
    )
    
    print("\n" + "="*60)
    print("Preparation Complete!")
    print(f"Output directory: {args.output_dir}")
    print("\nNext steps:")
    print("  1. Review synthesis_manifest.csv")
    print("  2. Run StyleTTS2 synthesis script")
    print("  3. Compare original vs synthetic WER")
    print("="*60)


if __name__ == "__main__":
    main()