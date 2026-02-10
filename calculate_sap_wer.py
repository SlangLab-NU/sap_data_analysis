import json
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from jiwer import wer
import torch

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description="Calculate WER for SAP speakers using NeMo ASR")
    
    parser.add_argument(
        "--sap-dir",
        type=Path,
        default=Path("VallE/egs/sap/extracted"),
        help="Path to extracted SAP data"
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("/scratch/lewis.jor"),
        help="Base working directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sap_speaker_wer.csv"),
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
        default=16,
        help="Batch size for ASR inference"
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=None,
        help="Maximum speakers to process (for testing)"
    )
    
    return parser.parse_args()


def load_nemo_model(model_name):
    """Load NVIDIA NeMo ASR model."""
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        logger.error("NeMo not installed. Install with: pip install nemo_toolkit[asr]")
        return None
    
    logger.info(f"Loading NeMo model: {model_name}")
    
    # Check if GPU available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load pretrained model
    asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name)
    asr_model.to(device)
    asr_model.eval()
    
    logger.info(f"✓ Model loaded successfully")
    return asr_model


def get_speaker_metadata(speaker_dir):
    """Extract metadata from speaker's JSON file."""
    json_files = list(speaker_dir.glob("*.json"))
    
    if not json_files:
        return None
    
    with open(json_files[0], 'r') as f:
        data = json.load(f)
    
    return {
        'speaker_id': data.get('Contributor ID', 'Unknown'),
        'etiology': data.get('Etiology', 'Unknown'),
        'files': data.get('Files', [])
    }


def calculate_speaker_wer(speaker_dir, asr_model, batch_size=16):
    """
    Calculate WER for a single speaker.
    Returns: (speaker_id, etiology, num_utterances, average_wer)
    """
    # Get metadata
    metadata = get_speaker_metadata(speaker_dir)
    if not metadata:
        logger.warning(f"No metadata found for {speaker_dir.name}")
        return None
    
    speaker_id = metadata['speaker_id']
    etiology = metadata['etiology']
    files_data = metadata['files']
    
    # Collect audio files and ground truth transcripts
    audio_paths = []
    ground_truths = []
    
    for file_entry in files_data:
        filename = file_entry.get('Filename', '')
        
        # Get ground truth transcript
        prompt = file_entry.get('Prompt', {})
        transcript = prompt.get('Transcript', '')
        
        if not transcript:
            continue
        
        # Find corresponding audio file
        audio_path = speaker_dir / filename
        if audio_path.exists():
            audio_paths.append(str(audio_path))
            ground_truths.append(transcript.strip())
    
    if not audio_paths:
        logger.warning(f"No audio files found for speaker {speaker_id}")
        return None
    
    # Run ASR in batches
    logger.info(f"Processing {len(audio_paths)} utterances for speaker {speaker_id}")
    
    predictions = []
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i+batch_size]
        batch_preds = asr_model.transcribe(batch_paths)
        predictions.extend(batch_preds)
    
    # Calculate WER for each utterance
    wer_scores = []
    for gt, pred in zip(ground_truths, predictions):
        try:
            # Normalize text (lowercase, strip)
            gt_norm = gt.lower().strip()
            pred_norm = pred.lower().strip()
            
            # Calculate WER
            utterance_wer = wer(gt_norm, pred_norm)
            wer_scores.append(utterance_wer)
        except Exception as e:
            logger.warning(f"Error calculating WER: {e}")
            continue
    
    # Calculate average WER
    if wer_scores:
        avg_wer = sum(wer_scores) / len(wer_scores)
    else:
        avg_wer = None
    
    return {
        'Speaker_ID': speaker_id,
        'Etiology': etiology,
        'Num_Utterances': len(wer_scores),
        'Average_WER': avg_wer
    }


def process_dataset(sap_dir, dataset_type, asr_model, batch_size, max_speakers=None):
    """Process all speakers in a dataset (DEV or TRAIN)."""
    dataset_dir = sap_dir / dataset_type
    
    if not dataset_dir.exists():
        logger.warning(f"Dataset directory not found: {dataset_dir}")
        return []
    
    speaker_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    if max_speakers:
        speaker_dirs = speaker_dirs[:max_speakers]
    
    logger.info(f"\nProcessing {len(speaker_dirs)} speakers from {dataset_type}")
    
    results = []
    
    for speaker_dir in tqdm(speaker_dirs, desc=f"Processing {dataset_type}"):
        try:
            result = calculate_speaker_wer(speaker_dir, asr_model, batch_size)
            if result:
                result['Dataset'] = dataset_type
                results.append(result)
        except Exception as e:
            logger.error(f"Error processing {speaker_dir.name}: {e}")
            continue
    
    return results


def save_results(results, output_path):
    """Save WER results to CSV."""
    if not results:
        logger.warning("No results to save")
        return
    
    df = pd.DataFrame(results)
    
    # Sort by WER (ascending)
    df = df.sort_values('Average_WER')
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    
    logger.info(f"\n✓ Results saved to {output_path}")
    
    # Print statistics
    logger.info("\nWER STATISTICS:")
    logger.info("="*60)
    logger.info(f"Total speakers processed: {len(df)}")
    logger.info(f"Average WER: {df['Average_WER'].mean():.2%}")
    logger.info(f"Median WER: {df['Average_WER'].median():.2%}")
    logger.info(f"Min WER: {df['Average_WER'].min():.2%}")
    logger.info(f"Max WER: {df['Average_WER'].max():.2%}")
    
    # Statistics by etiology
    logger.info("\nWER BY ETIOLOGY:")
    logger.info("="*60)
    etiology_stats = df.groupby('Etiology')['Average_WER'].agg(['count', 'mean', 'median'])
    for etiology, stats in etiology_stats.iterrows():
        logger.info(f"{etiology}:")
        logger.info(f"  Speakers: {int(stats['count'])}")
        logger.info(f"  Mean WER: {stats['mean']:.2%}")
        logger.info(f"  Median WER: {stats['median']:.2%}")
    
    # Top 10 best (lowest WER)
    logger.info("\nTOP 10 MOST INTELLIGIBLE SPEAKERS (Lowest WER):")
    for i, row in df.head(10).iterrows():
        logger.info(f"  {row['Speaker_ID'][:8]}... ({row['Etiology']}): {row['Average_WER']:.2%}")
    
    # Bottom 10 (highest WER)
    logger.info("\nTOP 10 LEAST INTELLIGIBLE SPEAKERS (Highest WER):")
    for i, row in df.tail(10).iterrows():
        logger.info(f"  {row['Speaker_ID'][:8]}... ({row['Etiology']}): {row['Average_WER']:.2%}")


def main():
    args = get_args()
    
    logger.info("="*60)
    logger.info("SAP Speaker WER Calculation using NVIDIA NeMo")
    logger.info("="*60)
    
    # Construct full SAP directory path
    sap_dir = args.working_dir / args.sap_dir
    
    # Load NeMo ASR model
    asr_model = load_nemo_model(args.model_name)
    if not asr_model:
        return
    
    # Process both datasets
    all_results = []
    
    # Process DEV
    dev_results = process_dataset(sap_dir, "DEV", asr_model, args.batch_size, args.max_speakers)
    all_results.extend(dev_results)
    
    # Process TRAIN
    train_results = process_dataset(sap_dir, "TRAIN", asr_model, args.batch_size, args.max_speakers)
    all_results.extend(train_results)
    
    # Save results
    save_results(all_results, args.output)
    
    logger.info("\n" + "="*60)
    logger.info("Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()