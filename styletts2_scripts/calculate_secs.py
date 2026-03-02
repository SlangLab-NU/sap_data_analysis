import torch
import soundfile as sf
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import resample_poly
from math import gcd
from transformers import AutoFeatureExtractor, AutoModel
from torch.nn.functional import cosine_similarity
from tqdm import tqdm
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_ID = "microsoft/wavlm-base-plus-sv"


def get_args():
    parser = argparse.ArgumentParser(description="Calculate speaker embedding similarity using WavLM")
    
    parser.add_argument(
        "--synthesis-results",
        type=Path,
        default=Path("styletts2_synthesis/generated_audio/synthesis_results.csv"),
        help="Path to synthesis results CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("styletts2_synthesis/speaker_similarity.csv"),
        help="Output CSV path"
    )
    
    return parser.parse_args()


def load_audio(filepath: str, target_sr: int = 16000) -> np.ndarray:
    """Load and resample audio to 16kHz mono."""
    wav, sr = sf.read(filepath, dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)  # mono
    
    if sr != target_sr:
        g = gcd(sr, target_sr)
        wav = resample_poly(wav, target_sr // g, sr // g).astype(np.float32)
    
    return wav


def get_embedding(filepath: str, extractor, model) -> torch.Tensor:
    """Extract speaker embedding from audio file."""
    try:
        wav = load_audio(filepath)
        inputs = extractor(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Mean-pool over time dimension
        return outputs.last_hidden_state.mean(dim=1).squeeze()
    
    except Exception as e:
        logger.warning(f"Failed to extract embedding from {Path(filepath).name}: {e}")
        return None


def calculate_similarities(synthesis_df, extractor, model):
    """
    Calculate cosine similarity between original and synthetic audio.
    """
    results = []
    
    logger.info(f"Calculating speaker similarity for {len(synthesis_df)} speakers...")
    
    for _, row in tqdm(synthesis_df.iterrows(), total=len(synthesis_df), desc="Processing"):
        speaker_id = row['speaker_id']
        etiology = row['etiology']
        original_audio = row['reference_audio']
        synthetic_audio = row['synthetic_audio']
        status = row['status']
        
        # Skip failed syntheses
        if status != 'success' or pd.isna(synthetic_audio):
            results.append({
                'speaker_id': speaker_id,
                'etiology': etiology,
                'original_audio': original_audio,
                'synthetic_audio': synthetic_audio,
                'cosine_similarity': None,
                'interpretation': status,
                'status': status
            })
            continue
        
        # Extract embeddings
        emb1 = get_embedding(original_audio, extractor, model)
        emb2 = get_embedding(synthetic_audio, extractor, model)
        
        if emb1 is None or emb2 is None:
            results.append({
                'speaker_id': speaker_id,
                'etiology': etiology,
                'original_audio': original_audio,
                'synthetic_audio': synthetic_audio,
                'cosine_similarity': None,
                'interpretation': 'embedding extraction failed',
                'status': 'failed'
            })
            continue
        
        # Calculate cosine similarity
        sim = cosine_similarity(emb1.unsqueeze(0), emb2.unsqueeze(0)).item()
        
        # Interpretation
        if sim > 0.90:
            interpretation = "Very likely same speaker"
        elif sim > 0.75:
            interpretation = "Possibly same speaker"
        else:
            interpretation = "Likely different speakers"
        
        results.append({
            'speaker_id': speaker_id,
            'etiology': etiology,
            'original_audio': original_audio,
            'synthetic_audio': synthetic_audio,
            'cosine_similarity': round(sim, 4),
            'interpretation': interpretation,
            'status': 'success'
        })
    
    return pd.DataFrame(results)


def print_statistics(similarity_df):
    """Print summary statistics."""
    valid = similarity_df[similarity_df['status'] == 'success']
    
    if len(valid) == 0:
        logger.warning("No successful comparisons")
        return
    
    logger.info("\n" + "="*60)
    logger.info("SPEAKER SIMILARITY STATISTICS (WavLM Embeddings)")
    logger.info("="*60)
    logger.info(f"Total speakers: {len(similarity_df)}")
    logger.info(f"Successful: {len(valid)}")
    logger.info(f"Failed: {len(similarity_df) - len(valid)}")
    
    logger.info(f"\nCosine Similarity (1.0 = identical):")
    logger.info(f"  Mean: {valid['cosine_similarity'].mean():.4f}")
    logger.info(f"  Median: {valid['cosine_similarity'].median():.4f}")
    logger.info(f"  Std: {valid['cosine_similarity'].std():.4f}")
    logger.info(f"  Min: {valid['cosine_similarity'].min():.4f}")
    logger.info(f"  Max: {valid['cosine_similarity'].max():.4f}")
    
    # Interpretation breakdown
    logger.info(f"\nInterpretation Breakdown:")
    for interp in ['Very likely same speaker', 'Possibly same speaker', 'Likely different speakers']:
        count = len(valid[valid['interpretation'] == interp])
        pct = (count / len(valid)) * 100
        logger.info(f"  {interp}: {count} ({pct:.1f}%)")
    
    # By etiology
    logger.info(f"\nBy Etiology:")
    for etiology in valid['etiology'].unique():
        etiology_data = valid[valid['etiology'] == etiology]
        logger.info(f"  {etiology}:")
        logger.info(f"    Count: {len(etiology_data)}")
        logger.info(f"    Mean similarity: {etiology_data['cosine_similarity'].mean():.4f}")
        logger.info(f"    Std: {etiology_data['cosine_similarity'].std():.4f}")
    
    # Best matches
    logger.info(f"\nTop 5 Best Matches:")
    top5 = valid.nlargest(5, 'cosine_similarity')
    for _, row in top5.iterrows():
        logger.info(f"  {row['speaker_id'][:12]}... ({row['etiology']}): {row['cosine_similarity']:.4f}")
    
    # Worst matches
    logger.info(f"\nTop 5 Worst Matches:")
    bottom5 = valid.nsmallest(5, 'cosine_similarity')
    for _, row in bottom5.iterrows():
        logger.info(f"  {row['speaker_id'][:12]}... ({row['etiology']}): {row['cosine_similarity']:.4f}")


def main():
    args = get_args()
    
    logger.info("="*60)
    logger.info("Speaker Embedding Similarity (WavLM)")
    logger.info("="*60)
    
    # Load WavLM model
    logger.info(f"Loading model: {MODEL_ID}")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(MODEL_ID)
    model.eval()
    logger.info("✓ Model loaded")
    
    # Load synthesis results
    synthesis_df = pd.read_csv(args.synthesis_results)
    logger.info(f"Loaded {len(synthesis_df)} synthesis results")
    
    # Calculate similarities
    similarity_df = calculate_similarities(synthesis_df, extractor, model)
    
    # Save results
    similarity_df.to_csv(args.output, index=False)
    logger.info(f"\n✓ Results saved to: {args.output}")
    
    # Print statistics
    print_statistics(similarity_df)
    
    logger.info("\n" + "="*60)
    logger.info("Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()