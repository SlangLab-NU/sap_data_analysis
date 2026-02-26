import json
import pandas as pd
from pathlib import Path
import argparse
from tqdm import tqdm
import logging
from jiwer import wer
import torch
import re
import soundfile as sf
import numpy as np

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
        "--log-file",
        type=Path,
        default=Path("sap_wer_detailed.log"),
        help="Detailed log file with examples"
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
    parser.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log example every N utterances"
    )
    
    return parser.parse_args()


def setup_logging(log_file):
    """Setup logging to both console and file."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def load_nemo_model(model_name, logger):
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


def load_and_convert_to_mono(audio_path):
    """Load audio and convert to mono if stereo."""
    try:
        data, sr = sf.read(audio_path)
        
        # Check if stereo
        if len(data.shape) > 1 and data.shape[1] > 1:
            # Convert to mono by averaging channels
            data = np.mean(data, axis=1)
        
        # Check for NaN/Inf
        if not np.isfinite(data).all():
            return None, None, "Audio contains NaN or Inf"
        
        return data, sr, None
        
    except Exception as e:
        return None, None, str(e)


def validate_audio_file(audio_path):
    """Check if audio file is valid and finite."""
    try:
        data, sr = sf.read(audio_path)
        
        # Check for NaN or Inf values
        if not np.isfinite(data).all():
            return False, "Audio contains NaN or Inf values"
        
        # Check if too long (> 5 minutes)
        duration = len(data) / sr
        if duration > 300:  # 5 minutes
            return False, f"Audio too long: {duration:.1f}s"
        
        return True, None
        
    except Exception as e:
        return False, str(e)


def normalize_text(text):
    """
    Normalize text for fair WER calculation.
    Removes punctuation, extra spaces, and lowercases.
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


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
        return processed
    except Exception as e:
        return set()


def append_result_to_csv(result, output_path):
    """Append a single result to CSV (creates file if doesn't exist)."""
    df_new = pd.DataFrame([result])
    
    if output_path.exists():
        # Append to existing
        df_new.to_csv(output_path, mode='a', header=False, index=False)
    else:
        # Create new with header
        df_new.to_csv(output_path, mode='w', header=True, index=False)


def calculate_speaker_wer(speaker_dir, asr_model, logger, batch_size=16, log_every=10):
    """Calculate WER for a single speaker."""
    metadata = get_speaker_metadata(speaker_dir)
    if not metadata:
        logger.warning(f"No metadata found for {speaker_dir.name}")
        return None
    
    speaker_id = metadata['speaker_id']
    etiology = metadata['etiology']
    files_data = metadata['files']
    
    # Collect valid audio files
    audio_paths = []
    ground_truths = []
    skipped = 0
    
    # Create temp directory for converted files
    temp_dir = speaker_dir / "temp_mono"
    temp_dir.mkdir(exist_ok=True)
    
    for file_entry in files_data:
        filename = file_entry.get('Filename', '')
        prompt = file_entry.get('Prompt', {})
        transcript = prompt.get('Transcript', '')
        
        if not transcript:
            continue
        
        audio_path = speaker_dir / filename
        if not audio_path.exists():
            continue
        
        # Load and convert to mono
        data, sr, error = load_and_convert_to_mono(audio_path)
        
        if error:
            logger.warning(f"Skipping {filename}: {error}")
            skipped += 1
            continue
        
        # Save as mono in temp directory
        temp_path = temp_dir / filename
        sf.write(temp_path, data, sr)
        
        audio_paths.append(str(temp_path))
        ground_truths.append(transcript.strip())
    
    if skipped > 0:
        logger.warning(f"Skipped {skipped} problematic files for speaker {speaker_id}")
    
    if not audio_paths:
        logger.warning(f"No valid audio files for speaker {speaker_id}")
        # Clean up temp dir
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        return None
    
    tqdm.write(f"\nProcessing speaker {speaker_id[:12]}... ({etiology}) - {len(audio_paths)} valid utterances")
    logger.info(f"Processing {len(audio_paths)} utterances for speaker {speaker_id} ({etiology})")
    
    # Run ASR in batches
    predictions = []
    
    for i in range(0, len(audio_paths), batch_size):
        batch_paths = audio_paths[i:i+batch_size]
        
        try:
            batch_preds = asr_model.transcribe(batch_paths)
            predictions.extend(batch_preds)
        except Exception as e:
            logger.error(f"Batch failed: {e}")
            # Process individually as fallback
            for path in batch_paths:
                try:
                    pred = asr_model.transcribe([path])
                    predictions.extend(pred)
                except:
                    predictions.append("")
    
    # Clean up temp directory
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Calculate WER (rest stays the same...)
    wer_scores = []
    for idx, (gt, pred) in enumerate(zip(ground_truths, predictions)):
        if not pred:
            continue
            
        try:
            gt_norm = normalize_text(gt)
            pred_norm = normalize_text(pred)
            utterance_wer = wer(gt_norm, pred_norm)
            wer_scores.append(utterance_wer)
            
            if idx % log_every == 0:
                log_msg = (
                    f"\n  Example {idx + 1}/{len(ground_truths)}:\n"
                    f"    Ground Truth (norm): {gt_norm}\n"
                    f"    ASR Output   (norm): {pred_norm}\n"
                    f"    WER:                 {utterance_wer:.2%}\n"
                )
                logger.info(log_msg)
                if idx % (log_every * 3) == 0:
                    tqdm.write(log_msg)
                
        except Exception as e:
            logger.warning(f"Error calculating WER: {e}")
            continue
    
    if wer_scores:
        avg_wer = sum(wer_scores) / len(wer_scores)
    else:
        avg_wer = None
    
    summary = f"✓ Speaker {speaker_id[:12]}... complete: {len(wer_scores)}/{len(audio_paths)} utterances, Avg WER = {avg_wer:.2%if avg_wer else 'N/A'}"
    logger.info(summary)
    tqdm.write(summary)
    
    return {
        'Speaker_ID': speaker_id,
        'Etiology': etiology,
        'Num_Utterances': len(wer_scores),
        'Num_Failed': skipped,
        'Average_WER': avg_wer
    }


def process_dataset(sap_dir, dataset_type, asr_model, logger, batch_size, log_every, 
                    output_path, processed_speakers, max_speakers=None):
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
    
    logger.info(f"\n{dataset_type} Dataset:")
    logger.info(f"  Total speakers: {len(speaker_dirs)}")
    logger.info(f"  Already processed: {len(speaker_dirs) - len(speaker_dirs_to_process)}")
    logger.info(f"  Remaining: {len(speaker_dirs_to_process)}")
    
    if max_speakers:
        speaker_dirs_to_process = speaker_dirs_to_process[:max_speakers]
        logger.info(f"  Limited to: {max_speakers} speakers")
    
    results = []
    
    for speaker_dir in tqdm(speaker_dirs_to_process, desc=f"Processing {dataset_type}"):
        try:
            result = calculate_speaker_wer(speaker_dir, asr_model, logger, batch_size, log_every)
            if result:
                result['Dataset'] = dataset_type
                results.append(result)
                
                # INCREMENTAL SAVE
                append_result_to_csv(result, output_path)
                
        except Exception as e:
            logger.error(f"Error processing {speaker_dir.name}: {e}")
            continue
    
    return results


def save_results(results, output_path, logger):
    """Save WER results to CSV with sorting and statistics."""
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
    
    # Setup logging
    logger = setup_logging(args.log_file)
    
    logger.info("="*60)
    logger.info("SAP Speaker WER Calculation using NVIDIA NeMo")
    logger.info("="*60)
    logger.info(f"Detailed logs: {args.log_file}")
    logger.info(f"Output CSV: {args.output}")
    
    # Check for already processed speakers
    processed_speakers = get_already_processed_speakers(args.output)
    if processed_speakers:
        logger.info(f"Found {len(processed_speakers)} already processed speakers")
    
    # Construct full SAP directory path
    sap_dir = args.working_dir / args.sap_dir
    
    # Load NeMo ASR model
    asr_model = load_nemo_model(args.model_name, logger)
    if not asr_model:
        return
    
    # Process both datasets (incremental saves happen inside)
    all_results = []
    
    # Process DEV
    dev_results = process_dataset(
        sap_dir, "DEV", asr_model, logger, 
        args.batch_size, args.log_every, args.output,
        processed_speakers, args.max_speakers
    )
    all_results.extend(dev_results)
    
    # Process TRAIN
    train_results = process_dataset(
        sap_dir, "TRAIN", asr_model, logger,
        args.batch_size, args.log_every, args.output,
        processed_speakers, args.max_speakers
    )
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