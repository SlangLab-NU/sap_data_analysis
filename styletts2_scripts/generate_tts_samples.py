import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import nltk
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    logger.info("Downloading NLTK punkt_tab...")
    nltk.download('punkt_tab', quiet=True)
    nltk.download('punkt', quiet=True)

def get_args():
    parser = argparse.ArgumentParser(description="Generate StyleTTS2 voice clones from SAP speakers")
    
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("styletts2_synthesis/synthesis_manifest.csv"),
        help="Path to synthesis manifest CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("styletts2_synthesis/generated_audio"),
        help="Output directory for synthesized audio"
    )
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=10,
        help="Number of diffusion steps (higher = better quality, slower)"
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum speakers to process (for testing)"
    )
    
    return parser.parse_args()

def load_styletts2_model():
    """Load StyleTTS2 model."""
    try:
        from styletts2 import tts
    except ImportError:
        logger.error("StyleTTS2 not installed. Install with: pip install styletts2")
        return None
    
    logger.info("Loading StyleTTS2 model...")
    logger.info("  (Model will auto-download on first run, ~400MB)")
    
    # Initialize model (auto-downloads LibriTTS checkpoint)
    tts_model = tts.StyleTTS2()
    
    logger.info("✓ StyleTTS2 model loaded successfully")
    
    return tts_model


def synthesize_for_speaker(speaker_data, tts_model, output_dir, diffusion_steps):
    """
    Synthesize speech for one speaker using their reference audio.
    
    Args:
        speaker_data: Row from manifest with speaker info
        tts_model: StyleTTS2 model instance
        output_dir: Directory to save synthesized audio
        diffusion_steps: Quality parameter
    
    Returns:
        dict with synthesis results
    """
    speaker_id = speaker_data['speaker_id']
    ref_audio = speaker_data['reference_audio']
    prompt_text = speaker_data['prompt_text']
    transcript = speaker_data['transcript']
    
    # Create output path
    output_path = output_dir / f"{speaker_id}_synthetic.wav"
    
    try:
        # Synthesize using prompt text as the text to speak
        tts_model.inference(
            prompt_text,
            target_voice_path=ref_audio,
            output_wav_file=str(output_path),
            diffusion_steps=diffusion_steps,
            alpha=0.3,  # Style control
            beta=0.7,   # Speaker similarity control
            embedding_scale=1
        )
        
        return {
            'speaker_id': speaker_id,
            'etiology': speaker_data['etiology'],
            'reference_audio': ref_audio,
            'synthetic_audio': str(output_path),
            'prompt_text': prompt_text,
            'transcript': transcript,
            'status': 'success'
        }
        
    except Exception as e:
        logger.error(f"Failed to synthesize for speaker {speaker_id}: {e}")
        
        return {
            'speaker_id': speaker_id,
            'etiology': speaker_data['etiology'],
            'reference_audio': ref_audio,
            'synthetic_audio': None,
            'prompt_text': prompt_text,
            'transcript': transcript,
            'status': f'failed: {str(e)}'
        }


def main():
    args = get_args()
    
    logger.info("="*60)
    logger.info("StyleTTS2 Synthesis for SAP Speakers")
    logger.info("="*60)
    
    # Load manifest
    manifest_df = pd.read_csv(args.manifest)
    
    if args.max_speakers:
        manifest_df = manifest_df.head(args.max_speakers)
    
    logger.info(f"Loaded {len(manifest_df)} speakers from manifest")
    logger.info(f"Diffusion steps: {args.diffusion_steps}")
    
    # Load StyleTTS2 model
    tts_model = load_styletts2_model()
    if not tts_model:
        return
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate synthetic speech
    results = []
    
    for _, speaker_data in tqdm(manifest_df.iterrows(), total=len(manifest_df), desc="Synthesizing"):
        result = synthesize_for_speaker(speaker_data, tts_model, output_dir, args.diffusion_steps)
        results.append(result)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output_dir / 'synthesis_results.csv', index=False)
    
    # Print summary
    successful = len(results_df[results_df['status'] == 'success'])
    failed = len(results_df) - successful
    
    logger.info("\n" + "="*60)
    logger.info("Synthesis Complete!")
    logger.info("="*60)
    logger.info(f"Total speakers: {len(results_df)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"\nResults saved to:")
    logger.info(f"  Audio: {args.output_dir}")
    logger.info(f"  CSV: {args.output_dir / 'synthesis_results.csv'}")
    logger.info("="*60)


if __name__ == "__main__":
    main()