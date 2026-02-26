import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
import re
import numpy as np
from scipy import stats

def get_args():
    parser = argparse.ArgumentParser(description="Analyze SAP WER results")
    
    parser.add_argument(
        "--wer-csv",
        type=Path,
        default=Path("sap_speaker_wer.csv"),
        help="WER results CSV"
    )
    parser.add_argument(
        "--wer-log",
        type=Path,
        default=Path("sap_wer_detailed.log"),
        help="WER detailed log file"
    )
    parser.add_argument(
        "--ratings-csv",
        type=Path,
        default=Path("VallE/egs/sap/analysis/ratings/speaker_ratings_combined.csv"),
        help="Speaker ratings CSV"
    )
    parser.add_argument(
        "--working-dir",
        type=Path,
        default=Path("/scratch/lewis.jor"),
        help="Base working directory"
    )
    parser.add_argument(
        "--sap-scripts-dir",
        type=Path,
        default=Path("/scratch/lewis.jor/sap_scripts/sap_data_analysis"),
        help="Base working directory"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("wer_analysis"),
        help="Output directory for plots and results"
    )
    
    return parser.parse_args()


def extract_error_speakers(log_file):
    """
    Parse log file and extract all speakers that had errors.
    Returns DataFrame with speaker_id, etiology, error_type, error_message.
    """
    error_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ERROR - (.+)'
    )
    processing_pattern = re.compile(
        r'Processing \d+ utterances for speaker ([a-f0-9-]+) \(([^)]+)\)'
    )
    
    errors = []
    current_speaker = None
    current_etiology = None
    
    with open(log_file, 'r') as f:
        for line in f:
            # Check if this line identifies a speaker being processed
            proc_match = processing_pattern.search(line)
            if proc_match:
                current_speaker = proc_match.group(1)
                current_etiology = proc_match.group(2)
            
            # Check if this is an error line
            error_match = error_pattern.search(line)
            if error_match:
                timestamp = error_match.group(1)
                error_msg = error_match.group(2).strip()
                
                # Categorize error type
                if "Audio buffer is not finite" in error_msg:
                    error_type = "Corrupted Audio"
                elif "stack expects each tensor to be equal size" in error_msg:
                    error_type = "Variable Length Batch"
                elif "Output shape mismatch" in error_msg:
                    error_type = "Stereo Audio"
                elif "Failed to transcribe" in error_msg:
                    error_type = "Transcription Failed"
                elif "Error processing" in error_msg:
                    error_type = "Speaker Processing Failed"
                    # Extract speaker ID from error message
                    speaker_match = re.search(r'Error processing ([a-f0-9-]+):', error_msg)
                    if speaker_match:
                        current_speaker = speaker_match.group(1)
                else:
                    error_type = "Other"
                
                errors.append({
                    'timestamp': timestamp,
                    'speaker_id': current_speaker,
                    'etiology': current_etiology,
                    'error_type': error_type,
                    'error_message': error_msg
                })
    
    df = pd.DataFrame(errors)
    
    # Get unique speakers with errors
    if len(df) > 0:
        speaker_errors = df.groupby(['speaker_id', 'etiology', 'error_type']).size().reset_index(name='count')
        return df, speaker_errors
    else:
        return pd.DataFrame(), pd.DataFrame()


def create_wer_rating_correlation(wer_df, ratings_df, output_dir):
    """
    Create scatter plot correlating WER with speaker ratings.
    Different shapes for DEV/TRAIN, colors for etiology.
    """
    # Merge WER and ratings data
    merged = wer_df.merge(
        ratings_df[['Speaker_ID', 'Average_Rating', 'Number_of_Ratings']], 
        on='Speaker_ID', 
        how='inner'
    )
    
    # Remove rows with missing data
    merged = merged.dropna(subset=['Average_WER', 'Average_Rating'])
    
    if len(merged) == 0:
        print("No overlapping speakers between WER and ratings")
        return
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get unique etiologies for color mapping
    etiologies = merged['Etiology'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(etiologies)))
    etiology_colors = dict(zip(etiologies, colors))
    
    # Plot DEV (circles) and TRAIN (triangles)
    for dataset, marker in [('DEV', 'o'), ('TRAIN', '^')]:
        data = merged[merged['Dataset'] == dataset]
        
        for etiology in etiologies:
            etiology_data = data[data['Etiology'] == etiology]
            
            if len(etiology_data) > 0:
                ax.scatter(
                    etiology_data['Average_Rating'],
                    etiology_data['Average_WER'],
                    c=[etiology_colors[etiology]],
                    marker=marker,
                    s=100,
                    alpha=0.6,
                    label=f"{etiology} ({dataset})",
                    edgecolors='black',
                    linewidth=0.5
                )
    
    # Calculate and plot correlation line
    if len(merged) > 2:
        correlation = merged['Average_Rating'].corr(merged['Average_WER'])
        
        # Fit line
        z = np.polyfit(merged['Average_Rating'], merged['Average_WER'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(merged['Average_Rating'].min(), merged['Average_Rating'].max(), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2, 
                label=f'Correlation: {correlation:.3f}')
    
    ax.set_xlabel('Average Clinical Rating', fontsize=12, fontweight='bold')
    ax.set_ylabel('Word Error Rate (WER)', fontsize=12, fontweight='bold')
    ax.set_title('Correlation: Clinical Ratings vs ASR Performance', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'wer_rating_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved correlation plot: {output_dir / 'wer_rating_correlation.png'}")
    print(f"  Pearson correlation: {correlation:.3f}")
    print(f"  Speakers analyzed: {len(merged)}")
    
    return merged, correlation


def create_wer_by_etiology_barplot(wer_df, output_dir):
    """
    Create bar plot showing mean WER by etiology with std dev error bars.
    """
    # Remove None values
    df_valid = wer_df[wer_df['Average_WER'].notna()]
    
    # Calculate statistics by etiology
    etiology_stats = df_valid.groupby('Etiology')['Average_WER'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('count', 'count')
    ]).reset_index()
    
    # Sort by mean WER
    etiology_stats = etiology_stats.sort_values('mean')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x_pos = np.arange(len(etiology_stats))
    bars = ax.bar(
        x_pos, 
        etiology_stats['mean'],
        yerr=etiology_stats['std'],
        capsize=5,
        color='steelblue',
        alpha=0.7,
        edgecolor='black',
        linewidth=1.5
    )
    
    # Customize
    ax.set_xlabel('Etiology', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Word Error Rate (WER)', fontsize=12, fontweight='bold')
    ax.set_title('ASR Performance by Etiology', fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(etiology_stats['Etiology'], rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # Add value labels on bars
    for i, (bar, row) in enumerate(zip(bars, etiology_stats.itertuples())):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width()/2., 
            height,
            f'{row.mean:.1%}\n(n={int(row.count)})',
            ha='center', 
            va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    
    plt.tight_layout()
    plt.savefig(output_dir / 'wer_by_etiology.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved etiology bar plot: {output_dir / 'wer_by_etiology.png'}")
    
    return etiology_stats


def main():
    args = get_args()
    
    # Setup output directory
    output_dir = args.working_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("SAP WER Analysis")
    print("="*60)
    
    # 1. Extract error speakers from log
    print("\n1. Extracting error speakers from log...")
    log_path = args.sap_scripts_dir / args.wer_log
    
    if log_path.exists():
        all_errors, speaker_errors = extract_error_speakers(log_path)
        
        if len(speaker_errors) > 0:
            # Save to CSV
            error_output = output_dir / 'speakers_with_errors.csv'
            speaker_errors.to_csv(error_output, index=False)
            print(f"✓ Found {len(speaker_errors)} speakers with errors")
            print(f"  Saved to: {error_output}")
            
            # Print summary
            print("\nError Type Summary:")
            error_summary = all_errors['error_type'].value_counts()
            for error_type, count in error_summary.items():
                print(f"  {error_type}: {count}")
        else:
            print("  No errors found in log")
    else:
        print(f"  Log file not found: {log_path}")
    
    # 2. Load WER results
    print("\n2. Loading WER results...")
    wer_path = args.sap_scripts_dir / args.wer_csv
    wer_df = pd.read_csv(wer_path)
    print(f"✓ Loaded {len(wer_df)} speakers")
    print(f"  Valid WER scores: {wer_df['Average_WER'].notna().sum()}")
    
    # 3. Create etiology bar plot
    print("\n3. Creating WER by etiology bar plot...")
    etiology_stats = create_wer_by_etiology_barplot(wer_df, output_dir)
    
    # 4. Create correlation plot with ratings
    print("\n4. Creating WER vs ratings correlation plot...")
    ratings_path = args.working_dir / args.ratings_csv
    
    if ratings_path.exists():
        ratings_df = pd.read_csv(ratings_path)
        merged, correlation = create_wer_rating_correlation(wer_df, ratings_df, output_dir)
        
        # Save merged data
        merged_output = output_dir / 'wer_ratings_merged.csv'
        merged.to_csv(merged_output, index=False)
        print(f"  Merged data saved to: {merged_output}")
    else:
        print(f"  Ratings file not found: {ratings_path}")
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()