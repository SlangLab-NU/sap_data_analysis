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


def get_already_processed_speakers(output_path):
    """Load already processed speakers from existing CSV."""
    if not output_path.exists():
        return set()
    
    try:
        df = pd.read_csv(output_path)
        processed = set(df['Speaker_ID'].tolist())
        logger.info(f"Found {len(processed)} already processed speakers in {output_path}")
        return processed
    except Exception as e:
        logger.warning(f"Could not read existing results: {e}")
        return set()


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


def process_dataset(sap_dir, dataset_type, asr_model, logger, batch_size, log_every, 
                    processed_speakers, max_speakers=None):
    """Process all speakers in a dataset (DEV or TRAIN)."""
    dataset_dir = sap_dir / dataset_type
    
    if not dataset_dir.exists():
        logger.warning(f"Dataset directory not found: {dataset_dir}")
        return []
    
    speaker_dirs = [d for d in dataset_dir.iterdir() if d.is_dir()]
    
    # Filter out already processed speakers
    speaker_dirs_to_process = []
    for speaker_dir in speaker_dirs:
        metadata = get_speaker_metadata(speaker_dir)
        if metadata and metadata['speaker_id'] not in processed_speakers:
            speaker_dirs_to_process.append(speaker_dir)
    
    logger.info(f"Total speakers in {dataset_type}: {len(speaker_dirs)}")
    logger.info(f"Already processed: {len(speaker_dirs) - len(speaker_dirs_to_process)}")
    logger.info(f"Remaining to process: {len(speaker_dirs_to_process)}")
    
    if max_speakers:
        speaker_dirs_to_process = speaker_dirs_to_process[:max_speakers]
    
    results = []
    
    for speaker_dir in tqdm(speaker_dirs_to_process, desc=f"Processing {dataset_type}"):
        try:
            result = calculate_speaker_wer(speaker_dir, asr_model, logger, batch_size, log_every)
            if result:
                result['Dataset'] = dataset_type
                results.append(result)
                
                # INCREMENTAL SAVE - append to CSV after each speaker
                append_result_to_csv(result, args.output)
                
        except Exception as e:
            logger.error(f"Error processing {speaker_dir.name}: {e}")
            continue
    
    return results


def append_result_to_csv(result, output_path):
    """Append a single result to CSV (creates file if doesn't exist)."""
    df_new = pd.DataFrame([result])
    
    if output_path.exists():
        # Append to existing
        df_new.to_csv(output_path, mode='a', header=False, index=False)
    else:
        # Create new with header
        df_new.to_csv(output_path, mode='w', header=True, index=False)


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
    global args
    args = get_args()
    
    logger.info("="*60)
    logger.info("SAP Speaker WER Calculation using NVIDIA NeMo")
    logger.info("="*60)
    logger.info(f"Detailed logs: {args.log_file}")
    
    # Check for already processed speakers
    processed_speakers = get_already_processed_speakers(args.output)
    
    # Construct full SAP directory path
    sap_dir = args.working_dir / args.sap_dir
    
    # Load NeMo ASR model
    asr_model = load_nemo_model(args.model_name, logger)
    if not asr_model:
        return
    
    # Process both datasets (incremental saves happen inside)
    all_results = []
    
    # Process DEV
    dev_results = process_dataset(sap_dir, "DEV", asr_model, logger, args.batch_size, 
                                  args.log_every, processed_speakers, args.max_speakers)
    all_results.extend(dev_results)
    
    # Process TRAIN
    train_results = process_dataset(sap_dir, "TRAIN", asr_model, logger, args.batch_size, 
                                   args.log_every, processed_speakers, args.max_speakers)
    all_results.extend(train_results)
    
    # Final save with sorting and statistics
    # Re-read the full CSV (including previously processed speakers)
    if args.output.exists():
        final_df = pd.read_csv(args.output)
        save_results(final_df.to_dict('records'), args.output, logger)
    else:
        save_results(all_results, args.output, logger)
    
    logger.info("\n" + "="*60)
    logger.info("Complete!")
    logger.info(f"Results in: {args.output}")
    logger.info(f"Detailed logs: {args.log_file}")
    logger.info("="*60)


if __name__ == "__main__":
    main()