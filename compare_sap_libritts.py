import pandas as pd
from pathlib import Path
import argparse
import re
import gzip
import json
from tqdm import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sap-csv",
        type=Path,
        required=True,
        help="Path to SAP master_prompt_list.csv"
    )
    parser.add_argument(
        "--libritts-manifests",
        type=Path,
        default=Path("VallE/egs/libritts/data/manifests"),
        help="Path to LibriTTS manifest directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sap_novel_sentences_libritts.csv"),
        help="Output CSV path"
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("/scratch/lewis.jor"),
        help="Base working directory"
    )
    return parser.parse_args()


def load_libritts_transcripts(manifest_dir):
    """
    Load transcripts from LibriTTS supervision jsonl.gz files.
    """
    manifest_dir = Path(manifest_dir)
    
    if not manifest_dir.exists():
        print(f"❌ Manifest directory not found: {manifest_dir}")
        return []
    
    # Find supervision files (they contain the text)
    supervision_files = list(manifest_dir.glob("*supervisions*.jsonl.gz"))
    
    if not supervision_files:
        print(f"❌ No supervision files found in {manifest_dir}")
        return []
    
    print(f"\nFound {len(supervision_files)} LibriTTS supervision files:")
    for f in supervision_files:
        print(f"  - {f.name}")
    
    all_transcripts = []
    
    for supervision_file in supervision_files:
        print(f"\nReading {supervision_file.name}...")
        
        count = 0
        with gzip.open(supervision_file, 'rt', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # Extract text from the structure
                    text = data.get('text', '')
                    
                    if text:
                        all_transcripts.append({
                            'text': text,
                            'manifest_file': supervision_file.name,
                            'speaker_id': data.get('speaker', 'unknown'),
                            'recording_id': data.get('recording_id', 'unknown'),
                            'id': data.get('id', 'unknown'),
                        })
                        count += 1
                
                except json.JSONDecodeError as e:
                    continue
        
        print(f"  Loaded {count} transcripts")
    
    print(f"\n✓ Total LibriTTS utterances loaded: {len(all_transcripts)}")
    return all_transcripts


def load_novel_sentences(csv_path):
    """Load only Novel Sentences from SAP CSV."""
    df = pd.read_csv(csv_path)
    
    # Filter for Novel Sentences
    novel_sentences = df[df['Categories'].str.contains('Novel Sentences', na=False, case=False)]
    
    # Get unique prompts
    prompts = novel_sentences['Prompt_Text'].dropna().unique()
    
    # Create lookup
    prompt_lookup = {p.lower().strip(): p for p in prompts}
    
    print(f"\nLoaded {len(prompt_lookup)} unique Novel Sentences from SAP")
    print(f"\nExample Novel Sentences:")
    for i, prompt in enumerate(list(prompt_lookup.values())[:5]):
        print(f"  {i+1}. {prompt}")
    
    return prompt_lookup, df


def check_exact_phrase_match(sap_sentence, libritts_text):
    """Check if SAP sentence appears in LibriTTS text."""
    escaped = re.escape(sap_sentence.lower().strip())
    pattern = r'\b' + escaped + r'\b'
    return bool(re.search(pattern, libritts_text.lower()))


def search_for_matches(novel_sentences_lookup, libritts_transcripts):
    """Search for matches - OPTIMIZED."""
    matches = []
    matches_set = set()
    
    sentences_list = list(novel_sentences_lookup.items())
    
    print(f"\nSearching {len(libritts_transcripts)} LibriTTS utterances...")
    print(f"Checking against {len(sentences_list)} Novel Sentences...")
    
    # Build index
    libritts_lookup = {}
    for item in libritts_transcripts:
        text_lower = item['text'].lower()
        if text_lower not in libritts_lookup:
            libritts_lookup[text_lower] = []
        libritts_lookup[text_lower].append(item)
    
    print(f"Built index of {len(libritts_lookup)} unique LibriTTS texts")
    
    # Search
    for sentence_lower, sentence_original in tqdm(sentences_list, desc="Checking SAP sentences"):
        
        for text_lower, items in libritts_lookup.items():
            # Quick substring check first
            if sentence_lower in text_lower:
                # Only do regex if substring found
                if check_exact_phrase_match(sentence_lower, text_lower):
                    # Found a match!
                    for item in items:
                        match_key = (sentence_original, item['text'])
                        if match_key not in matches_set:
                            matches_set.add(match_key)
                            matches.append({
                                'SAP_Novel_Sentence': sentence_original,
                                'LibriTTS_Text': item['text'],
                                'LibriTTS_Manifest': item['manifest_file'],
                                'Speaker_ID': item['speaker_id'],
                                'Recording_ID': item['recording_id'],
                            })
                            print(f"\n  ✓ MATCH: {sentence_original}")
    
    return matches


def save_results(matches, output_path):
    """Save results."""
    if not matches:
        print("\n" + "="*60)
        print("RESULT: No matches found")
        print("="*60)
        print("\nThis confirms that SAP's Novel Sentences are unique prompts")
        print("that don't appear in LibriTTS audiobook transcriptions.")
        return
    
    matches_df = pd.DataFrame(matches)
    matches_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Results saved to {output_path}")
    print(f"\nTotal matches: {len(matches_df)}")
    
    print("\nALL MATCHES:")
    for i, row in matches_df.iterrows():
        print(f"\n{i+1}. SAP: '{row['SAP_Novel_Sentence']}'")
        print(f"   LibriTTS: '{row['LibriTTS_Text']}'")
        print(f"   Speaker: {row['Speaker_ID']}")


def main():
    args = get_args()
    
    print("="*60)
    print("SAP Novel Sentences vs LibriTTS Comparison")
    print("="*60)
    
    # Construct full path to manifests
    manifest_dir = args.working_dir / args.libritts_manifests
    
    # Load LibriTTS transcripts from supervision files
    libritts_transcripts = load_libritts_transcripts(manifest_dir)
    
    if not libritts_transcripts:
        print("\n❌ No LibriTTS transcripts loaded. Exiting.")
        return
    
    # Load SAP novel sentences
    novel_sentences, _ = load_novel_sentences(args.sap_csv)
    
    if not novel_sentences:
        print("\n❌ No Novel Sentences found in SAP CSV")
        return
    
    # Search
    matches = search_for_matches(novel_sentences, libritts_transcripts)
    
    # Save
    save_results(matches, args.output)
    
    print("\n" + "="*60)
    print("Complete!")
    print("="*60)


if __name__ == "__main__":
    main()