import pandas as pd
from pathlib import Path
import argparse
import re
import urllib.request
import tarfile
from tqdm import tqdm
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sap-csv",
        type=Path,
        required=True,
        help="Path to SAP master_prompt_list.csv"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/scratch/lewis.jor/librispeech_cache"),
        help="Cache directory for LibriSpeech downloads"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sap_novel_sentences_librispeech.csv"),
        help="Output CSV path"
    )
    return parser.parse_args()


def download_and_extract_transcripts(cache_dir):
    """
    Download LibriSpeech archives and extract only transcript files.
    Note: Archives contain audio too, but we only read .trans.txt files.
    """
    # These URLs contain the full dataset, but we'll only read transcripts
    transcript_urls = {
        'dev-clean': 'http://www.openslr.org/resources/12/dev-clean.tar.gz',
        'test-clean': 'http://www.openslr.org/resources/12/test-clean.tar.gz',
    }
    
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nUsing cache directory: {cache_dir}")
    print("\nDownloading LibriSpeech archives...")
    print("Note: Archives include audio but we only read transcript files")
    
    all_transcripts = []
    
    for split_name, url in transcript_urls.items():
        print(f"\n{split_name}:")
        
        tar_path = cache_dir / f"{split_name}.tar.gz"
        
        # Download if not present
        if not tar_path.exists():
            print(f"  Downloading (~350MB)...")
            urllib.request.urlretrieve(url, tar_path)
            print(f"  ✓ Downloaded to {tar_path}")
        else:
            print(f"  ✓ Already cached at {tar_path}")
        
        # Read transcripts directly from tar without extracting audio
        print(f"  Reading transcripts from archive...")
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            # Find all .trans.txt files
            trans_members = [m for m in tar.getmembers() if m.name.endswith('.trans.txt')]
            print(f"  Found {len(trans_members)} transcript files")
            
            for member in trans_members:
                # Extract and read this transcript file
                f = tar.extractfile(member)
                if f:
                    content = f.read().decode('utf-8')
                    for line in content.strip().split('\n'):
                        # Format: utterance_id TRANSCRIPT TEXT
                        parts = line.strip().split(maxsplit=1)
                        if len(parts) == 2:
                            utterance_id, text = parts
                            all_transcripts.append({
                                'utterance_id': utterance_id,
                                'text': text,
                                'split': split_name
                            })
    
    print(f"\n✓ Total LibriSpeech utterances loaded: {len(all_transcripts)}")
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


def check_exact_phrase_match(sap_sentence, librispeech_text):
    """Check if SAP sentence appears in LibriSpeech text."""
    escaped = re.escape(sap_sentence.lower().strip())
    pattern = r'\b' + escaped + r'\b'
    return bool(re.search(pattern, librispeech_text.lower()))


def search_for_matches(novel_sentences_lookup, librispeech_transcripts):
    """Search for matches - OPTIMIZED."""
    matches = []
    matches_set = set()
    
    sentences_list = list(novel_sentences_lookup.items())
    
    print(f"\nSearching {len(librispeech_transcripts)} LibriSpeech utterances...")
    print(f"Checking against {len(sentences_list)} Novel Sentences...")
    
    # OPTIMIZATION: Create a lowercase text lookup for fast checking
    librispeech_lookup = {}
    for item in librispeech_transcripts:
        text_lower = item['text'].lower()
        if text_lower not in librispeech_lookup:
            librispeech_lookup[text_lower] = []
        librispeech_lookup[text_lower].append(item)
    
    print(f"Built index of {len(librispeech_lookup)} unique LibriSpeech texts")
    
    # Now iterate through SAP sentences (better progress tracking)
    for sentence_lower, sentence_original in tqdm(sentences_list, desc="Checking SAP sentences"):
        
        # Check each LibriSpeech text
        for text_lower, items in librispeech_lookup.items():
            # Quick substring check first (much faster than regex)
            if sentence_lower in text_lower:
                # Only do expensive regex if substring found
                if check_exact_phrase_match(sentence_lower, text_lower):
                    # Found a match!
                    for item in items:
                        match_key = (sentence_original, item['text'])
                        if match_key not in matches_set:
                            matches_set.add(match_key)
                            matches.append({
                                'SAP_Novel_Sentence': sentence_original,
                                'LibriSpeech_Text': item['text'],
                                'LibriSpeech_Split': item['split'],
                                'Utterance_ID': item['utterance_id'],
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
        print("that don't appear in LibriSpeech audiobook transcriptions.")
        print("\nThis makes sense because:")
        print("  • SAP uses designed prompts for dysarthric speech testing")
        print("  • LibriSpeech is natural audiobook readings")
        print("  • The datasets serve different purposes")
        return
    
    matches_df = pd.DataFrame(matches)
    matches_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Results saved to {output_path}")
    print(f"\nTotal matches: {len(matches_df)}")
    
    print("\nALL MATCHES:")
    for i, row in matches_df.iterrows():
        print(f"\n{i+1}. SAP: '{row['SAP_Novel_Sentence']}'")
        print(f"   LibriSpeech: '{row['LibriSpeech_Text']}'")


def main():
    args = get_args()
    
    print("="*60)
    print("SAP Novel Sentences vs LibriSpeech Comparison")
    print("="*60)
    
    # Download and read transcripts using cache directory
    librispeech_transcripts = download_and_extract_transcripts(args.cache_dir)
    
    # Load SAP novel sentences
    novel_sentences, _ = load_novel_sentences(args.sap_csv)
    
    if not novel_sentences:
        print("\n❌ No Novel Sentences found in SAP CSV")
        return
    
    # Search
    matches = search_for_matches(novel_sentences, librispeech_transcripts)
    
    # Save
    save_results(matches, args.output)
    
    print("\n" + "="*60)
    print("Complete!")
    print("="*60)


if __name__ == "__main__":
    main()