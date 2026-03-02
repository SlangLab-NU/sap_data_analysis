import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from jiwer import wer
import torch
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser(description="Calculate WER for reference and synthetic audio")
    
    parser.add_argument(
        "--synthesis-results",
        type=Path,
        default=Path("styletts2_synthesis/generated_audio/synthesis_results.csv"),
        help="Path to synthesis results CSV"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("styletts2_synthesis/synthesis_wer_comparison.csv"),
        help="Output CSV path"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="stt_en_conformer_ctc_large",
        help="NeMo ASR model name"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for ASR inference"
    )
    
    return parser.parse_args()


def normalize_text(text):
    """Normalize text for fair WER calculation."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = ' '.join(text.split())
    return text.strip()


def load_nemo_model(model_name):
    """Load NVIDIA NeMo ASR model."""
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        logger.error("NeMo not installed")
        return None
    
    logger.info(f"Loading NeMo model: {model_name}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    asr_model.to(device)
    asr_model.eval()
    
    logger.info("✓ Model loaded successfully")
    return asr_model


def calculate_wer_for_audio(synthesis_df, asr_model, batch_size):
    """Calculate WER for both reference and synthetic audio."""
    
    # Filter successful syntheses
    valid_df = synthesis_df[synthesis_df['status'] == 'success'].copy()
    
    logger.info(f"Processing {len(valid_df)} successful syntheses...")
    
    if len(valid_df) == 0:
        logger.warning("No successful syntheses to process")
        return pd.DataFrame()
    
    # Collect audio files
    reference_files = valid_df['reference_audio'].tolist()
    synthetic_files = valid_df['synthetic_audio'].tolist()
    
    # Transcribe reference audio
    logger.info("Transcribing reference audio...")
    reference_predictions = []
    for i in tqdm(range(0, len(reference_files), batch_size), desc="Reference"):
        batch = reference_files[i:i+batch_size]
        try:
            preds = asr_model.transcribe(batch)
            reference_predictions.extend(preds)
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            for audio in batch:
                try:
                    pred = asr_model.transcribe([audio])
                    reference_predictions.extend(pred)
                except:
                    reference_predictions.append("")
    
    # Transcribe synthetic audio
    logger.info("Transcribing synthetic audio...")
    synthetic_predictions = []
    for i in tqdm(range(0, len(synthetic_files), batch_size), desc="Synthetic"):
        batch = synthetic_files[i:i+batch_size]
        try:
            preds = asr_model.transcribe(batch)
            synthetic_predictions.extend(preds)
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            for audio in batch:
                try:
                    pred = asr_model.transcribe([audio])
                    synthetic_predictions.extend(pred)
                except:
                    synthetic_predictions.append("")
    
    # Calculate WER
    logger.info("Calculating WER scores...")
    results = []
    
    for idx, row in valid_df.iterrows():
        ref_pred = reference_predictions[results.__len__()]
        synth_pred = synthetic_predictions[results.__len__()]
        
        ground_truth = row['prompt_text']
        gt_norm = normalize_text(ground_truth)
        
        # Calculate reference WER
        if ref_pred:
            ref_pred_norm = normalize_text(ref_pred)
            ref_wer = wer(gt_norm, ref_pred_norm)
        else:
            ref_wer = None
        
        # Calculate synthetic WER
        if synth_pred:
            synth_pred_norm = normalize_text(synth_pred)
            synth_wer = wer(gt_norm, synth_pred_norm)
        else:
            synth_wer = None
        
        # Calculate improvement
        if ref_wer is not None and synth_wer is not None:
            wer_improvement = ref_wer - synth_wer
            percent_improvement = (wer_improvement / ref_wer * 100) if ref_wer > 0 else 0
        else:
            wer_improvement = None
            percent_improvement = None
        
        results.append({
            'speaker_id': row['speaker_id'],
            'etiology': row['etiology'],
            'prompt_text': ground_truth,
            'reference_prediction': ref_pred,
            'synthetic_prediction': synth_pred,
            'reference_wer': round(ref_wer, 4) if ref_wer is not None else None,
            'synthetic_wer': round(synth_wer, 4) if synth_wer is not None else None,
            'wer_improvement': round(wer_improvement, 4) if wer_improvement is not None else None,
            'percent_improvement': round(percent_improvement, 2) if percent_improvement is not None else None
        })
    
    return pd.DataFrame(results)


def print_statistics(wer_df):
    """Print summary statistics."""
    if len(wer_df) == 0:
        logger.warning("No WER data to analyze")
        return
    
    valid = wer_df.dropna(subset=['reference_wer', 'synthetic_wer'])
    
    if len(valid) == 0:
        logger.warning("No valid WER comparisons")
        return
    
    logger.info("\n" + "="*60)
    logger.info("WER COMPARISON STATISTICS")
    logger.info("="*60)
    logger.info(f"Total speakers: {len(wer_df)}")
    logger.info(f"Valid comparisons: {len(valid)}")
    
    logger.info(f"\nReference Audio (Dysarthric):")
    logger.info(f"  Mean WER: {valid['reference_wer'].mean():.2%}")
    logger.info(f"  Median WER: {valid['reference_wer'].median():.2%}")
    logger.info(f"  Range: {valid['reference_wer'].min():.2%} - {valid['reference_wer'].max():.2%}")
    
    logger.info(f"\nSynthetic Audio (StyleTTS2):")
    logger.info(f"  Mean WER: {valid['synthetic_wer'].mean():.2%}")
    logger.info(f"  Median WER: {valid['synthetic_wer'].median():.2%}")
    logger.info(f"  Range: {valid['synthetic_wer'].min():.2%} - {valid['synthetic_wer'].max():.2%}")
    
    logger.info(f"\nWER Improvement:")
    logger.info(f"  Mean: {valid['wer_improvement'].mean():.4f} ({valid['percent_improvement'].mean():.2f}%)")
    logger.info(f"  Median: {valid['wer_improvement'].median():.4f}")
    
    # Breakdown
    improved = len(valid[valid['wer_improvement'] > 0])
    worse = len(valid[valid['wer_improvement'] < 0])
    same = len(valid[valid['wer_improvement'] == 0])
    
    logger.info(f"\n  Synthetic better: {improved} ({improved/len(valid)*100:.1f}%)")
    logger.info(f"  Synthetic worse: {worse} ({worse/len(valid)*100:.1f}%)")
    logger.info(f"  Same: {same}")
    
    # By etiology
    logger.info(f"\nBy Etiology:")
    for etiology in valid['etiology'].unique():
        data = valid[valid['etiology'] == etiology]
        logger.info(f"  {etiology}:")
        logger.info(f"    Count: {len(data)}")
        logger.info(f"    Ref WER: {data['reference_wer'].mean():.2%}")
        logger.info(f"    Synth WER: {data['synthetic_wer'].mean():.2%}")
        logger.info(f"    Improvement: {data['percent_improvement'].mean():.2f}%")


def main():
    args = get_args()
    
    logger.info("="*60)
    logger.info("WER Comparison: Reference vs Synthetic Audio")
    logger.info("="*60)
    
    # Load synthesis results
    synthesis_df = pd.read_csv(args.synthesis_results)
    logger.info(f"Loaded {len(synthesis_df)} synthesis results")
    
    # Load NeMo model
    asr_model = load_nemo_model(args.model_name)
    if not asr_model:
        return
    
    # Calculate WER
    wer_df = calculate_wer_for_audio(synthesis_df, asr_model, args.batch_size)
    
    # Save results
    wer_df.to_csv(args.output, index=False)
    logger.info(f"\n✓ Results saved to: {args.output}")
    
    # Print statistics
    print_statistics(wer_df)
    
    logger.info("\n" + "="*60)
    logger.info("Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()